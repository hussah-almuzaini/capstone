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

gemini_model = genai.GenerativeModel("models/gemini-2.0-flash-lite")

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

from pathlib import Path

@app.post("/analyze_from_static")
def analyze_from_static():
    try:
        # Accepted extensions (video + image)
        valid_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png"]

        # Collect valid files only
        candidate_files = [
            f for f in Path("static/videos").iterdir()
            if f.suffix.lower() in valid_extensions and f.is_file
        ]

        if not candidate_files:
            return JSONResponse(content={"error": "❌ لا يوجد ملفات صورة أو فيديو في مجلد static"}, status_code=400)
        def get_file_time(file: Path):
            try:
                return file.stat().st_mtime  # Creation time
            except:
                return file.stat().st_mtime 
        # Get latest image or video
        latest_file = max(candidate_files, key=get_file_time)
        file_path = str(latest_file)
        print(f"✅ Processing file: {file_path}")

        # Check extension type
        if latest_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image = Image.open(file_path).convert("RGB").resize((384, 384))
            caption = generate_caption(image)
            return {"captions": [caption]}

        elif latest_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return JSONResponse(content={"error": "❌ لم يتم فتح الفيديو"}, status_code=400)

            captions = []
            frame_skip = 90
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_skip == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb).resize((384, 384))
                    captions.append(generate_caption(image))
                frame_count += 1
            cap.release()
            return {"captions": captions}
        else:
            return JSONResponse(content={"error": "📂 نوع الملف غير مدعوم."}, status_code=400)

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
    if not speech_key:
        return JSONResponse(content={"error": "Missing Azure speech key"}, status_code=500)
    
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
    
@app.get("/depth-live")
async def depth_page():
    return FileResponse("static/search.html")

# uvicorn main:app --reload
# ...existing code...
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
from ultralytics import YOLO
import numpy as np
from gtts import gTTS

import shutil

# ...existing code...
@app.websocket("/ws/depth-live")
async def ws_depth_live(ws: WebSocket):
    await ws.accept()
    try:
        yolo_model = YOLO("yolov8n.pt")
        depth_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        depth_model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").eval()
        frame_width = 320
        detection_target = "person"  # Default
        frame_count = 0  # To track frames
        name_translation = {
            "person": "شخص",
            "chair": "كرسي",
            "handbag": "حقيبة",
            "backpack": "حقيبة",
            "dining table": "طاولة",
            "tv": "تلفاز",
            "tvmonitor": "تلفاز",
            "laptop": "حاسوب محمول",
            "bottle": "زجاجة"
        }
        # last_spoken = {}
        # cooldown_seconds = 5
        while True:
            data = await ws.receive_text()
            # Check if this is a target selection message
            try:
                payload = json.loads(data)
                if isinstance(payload, dict) and "target" in payload:
                    detection_target = payload["target"]
                    continue
            except Exception:
                pass  # Not JSON, treat as image

            if data.startswith("data:image"):
                frame_count += 1
                if frame_count % 60 != 0:
                    continue
                img_data = data.split(",")[1]
                img_bytes = base64.b64decode(img_data)
                image = Image.open(BytesIO(img_bytes)).convert("RGB")
                frame_width = image.width

                # 1. Object Detection
                frame_np = np.array(image)
                results = yolo_model(frame_np)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                names = results[0].names

                # 2. Depth Estimation
                inputs = depth_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    depth_outputs = depth_model(**inputs)
                post_depth = depth_processor.post_process_depth_estimation(
                    depth_outputs, source_sizes=[(image.height, image.width)]
                )
                depth_map = post_depth[0]["predicted_depth"].squeeze().cpu().numpy()

                # 3. Object-wise Depth with Direction
                descriptions = []
                # import time

                # current_time = time.time()
                for box, cls_id in zip(boxes, class_ids):
                    object_name = names[cls_id]
                    # Only detect the selected target
                    if object_name.lower() != detection_target.lower():
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.width, x2), min(image.height, y2)
                    # pad = 10  # expand by 10 pixels
                    # x1 = max(0, x1 - pad)
                    # y1 = max(0, y1 - pad)
                    # x2 = min(image.width - 1, x2 + pad)
                    # y2 = min(image.height - 1, y2 + pad)
                    if depth_map is not None:
                        object_depth = depth_map[y1:y2, x1:x2]
                        if object_depth.size > 0:
                            median_depth = np.median(object_depth)
                            center_x = (x1 + x2) / 2
                            relative_x = center_x / frame_width
                            if relative_x < 0.2:
                                direction = "إلى أقصى يسارك"
                            elif relative_x < 0.4:
                                direction = "إلى يسارك قليلاً"
                            elif relative_x < 0.6:
                                direction = "أمامك"
                            elif relative_x < 0.8:
                                direction = "إلى يمينك قليلاً"
                            else:
                                direction = "إلى أقصى يمينك"
                            # Check cooldown
                            # Use a unique key for each object and direction
                            # key = f"{object_name}:{direction}"
                            # last_time = last_spoken.get(key, 0)
                            
                            # if current_time - last_time < cooldown_seconds:
                            #     continue
                            # # Update last spoken time
                            # last_spoken[key] = current_time
                            arabic_name = name_translation.get(object_name.lower(), object_name)
                            descriptions.append(f"{arabic_name} على بعد {median_depth:.1f} متر {direction}")

                if not descriptions:
                    await ws.send_text(json.dumps({"description": "لم يتم الكشف عن الهدف المطلوب."}))
                    continue

                # Compose and speak
                # here update it to azure tts
                sentence = ". ".join(descriptions)
                speech_key = os.getenv("AZURE_SPEECH_KEY")
                service_region = os.getenv("AZURE_REGION", "eastus")

                speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
                speech_config.speech_synthesis_language = "ar-SA"
                speech_config.speech_synthesis_voice_name = "ar-SA-HamedNeural"
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

                result = synthesizer.speak_text_async(sentence).get()

                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    audio_path = f"static/output_{uuid.uuid4().hex}.wav"
                    with open(audio_path, "wb") as f:
                        f.write(result.audio_data)
                else:
                    print("❌ Azure speech failed.")
                    audio_path = None
                if audio_path:
                    await ws.send_text(json.dumps({
                            "description": sentence,
                            "audio_url": f"/{audio_path}"
                        }))
                else:
                        await ws.send_text(json.dumps({
                            "description": sentence
                        }))

    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))
        await ws.close()
# ...existing code...