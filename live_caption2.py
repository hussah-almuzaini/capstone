from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import cv2
import torch

# تحميل BLIP-2
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

cap = cv2.VideoCapture(0)
frame_skip = 5
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img).resize((364, 364))

    inputs = processor(images=pil_img, text="Describe the image", return_tensors="pt").to(model.device)

    try:
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=30)
            caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    except:
        caption = 'no caption'

    cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Caption (BLIP-2)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
