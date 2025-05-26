from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import base64
from io import BytesIO

# تحميل نموذج BLIP-1 الأخف
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to("cuda" if torch.cuda.is_available() else "cpu")

def caption_from_base64(img_base64: str) -> str:
    try:
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(BytesIO(img_bytes)).convert("RGB").resize((384, 384))

        inputs = processor(image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except:
        return "no caption"
