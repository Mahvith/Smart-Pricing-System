from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

image = Image.open("images/train/01SCsYMIKjL.jpg")
prompt = "Get features like country: origin of brand, quality: premium/local/standard, expiry: days,months,years,N/A"

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
