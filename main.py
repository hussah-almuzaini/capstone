# ✅ UPDATED main.py — Full Integration with upload.html
import os
import json
import base64
import uuid
import tempfile
from io import BytesIO
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import cv2
import torch

import azure.cognitiveservices.speech as speechsdk
import google.generativeai as genai
from transformers import BlipProcessor, BlipForConditionalGeneration
load_dotenv()
# إعداد BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# إعداد Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

gemini_model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# إعداد FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    return FileResponse("static/index.html")

def generate_caption(image):
    inputs = processor(image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs)
        return processor.decode(output[0], skip_special_tokens=True)

def caption_from_base64(img_base64: str) -> str:
    try:
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(BytesIO(img_bytes)).convert("RGB").resize((384, 384))
        return generate_caption(image)
    except Exception:
        return "no caption"

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = None

            if isinstance(payload, dict) and payload.get("type") == "frame":
                _, img_str = payload["data"].split(",", 1)
                caption = caption_from_base64(img_str)

                # Gemini ترجمة
                prompt = f"""
                ترجم الوصف التالي إلى اللغة العربية فقط بدون أي إضافات أخرى:

                {caption}
                """
                try:
                    response = gemini_model.generate_content(prompt)
                    arabic = getattr(response, "text", "").strip() or "❌ لم يتم الترجمة"
                except Exception as e:
                    print("❌ ترجمة فشلت:", e)
                    arabic = "❌ لم يتم الترجمة"

                await ws.send_text(json.dumps({
                    "type": "caption",
                    "text": caption,
                    "arabic": arabic
                }))
                continue

            # أوامر نصية أخرى
            msg = data.strip().lower()
            if "بصير" in msg:
                await ws.send_text("intro")
                continue
            cleaned = msg.replace("ال", " ").strip()
            if "بحث" in cleaned:
                await ws.send_text("redirect:/static/search.html")
            elif "بث" in cleaned:
                await ws.send_text("redirect:/static/stream.html")
            elif "رفع" in cleaned or "ملف" in cleaned:
                await ws.send_text("redirect:/static/upload.html")
            elif "الرئيسية" in cleaned:
                await ws.send_text("redirect:/")
            else:
                await ws.send_text("حاول مرة أخرى")
    except Exception as e:
        print("⚠️ WebSocket Error:", e)

@app.post("/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    try:
        captions = []
        suffix = file.filename.split(".")[-1]
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}")
        temp.write(await file.read())
        temp.close()

        if "image" in file.content_type:
            image = Image.open(temp.name).convert("RGB").resize((384, 384))
            captions.append(generate_caption(image))

        elif "video" in file.content_type:
            cap = cv2.VideoCapture(temp.name)
            frame_skip = 30
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_skip == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb).resize((384, 384))
                    captions.append(generate_caption(pil_img))
                frame_count += 1
            cap.release()

        os.unlink(temp.name)
        return {"captions": captions}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/translate")
async def translate_caption(request: Request):
    data = await request.json()
    english_text = data.get("text", "").strip()
    mode = data.get("mode", "translate")
    prompt = f"""
   أنت خبير في تلخيص أوصاف الفيديو إطارًا بإطار في سرد ​​متماسك وموجز. هدفك هو كتابة فقرة قصيرة وسلسة تصف بدقة الأنشطة الرئيسية وتسلسل أحداث الفيديو بلغة سليمة.

الرجاء:
- تجميع الأفعال المتكررة بشكل طبيعي.
- استخدام مفردات دقيقة ووصفية.
- الحفاظ على أسلوب محايد وغني بالمعلومات.
- تجنب الكلمات غير الضرورية مثل "أساسًا" أو "أحيانًا" أو المصطلحات غير الملائمة إلا إذا كانت ضرورية للسياق.

إليك قائمة التعليقات التوضيحية على مستوى الإطار:

ثم قدمها باللغة العربية.
    {english_text}
    """
    try:
        response = gemini_model.generate_content(prompt)
        arabic = getattr(response, "text", "").strip() or "❌ لم يتم الترجمة"
        return {"translated": arabic}

    except Exception as e:
        print("❌ خطأ أثناء الترجمة:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/speak")
async def speak_text(text: str = Form(...)):
    speech_key = os.getenv("AZURE_SPEECH_KEY")    
    service_region = "eastus"

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_language = "ar-SA"
    speech_config.speech_synthesis_voice_name = "ar-SA-HamedNeural"
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        filename = f"output_audio.wav"
        with open(filename, "wb") as f:
            f.write(result.audio_data)
        return FileResponse(filename, media_type="audio/wav", filename=filename)
    else:
        cancellation = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
        return JSONResponse(
            content={"error": str(result.reason), "details": f"{cancellation.reason}: {cancellation.error_details}"},
            status_code=500
        )

@app.post("/news")
async def news_caption(request: Request):
    prompt =f"""
انت ناشر احدث الاخبار في السعودية الرجاء قولها لشخص بشكل    و محترف 

 عطني النص فقط بدون اي تنسيق للنص نفسه وبدون اي علامات ترقيم ة *   """
    try:
        response = gemini_model.generate_content(prompt)
        news = getattr(response, "text", "").strip() or "❌ لم يتم توليد الخبر"
        return {"news": news}

    except Exception as e:
        print("❌ خطأ أثناء توليد الخبر:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)