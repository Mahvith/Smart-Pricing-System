import os
from google import genai
from google.genai import types
from pydantic import BaseModel
import requests
import pandas as pd

class Features(BaseModel):
    country: str
    quality: str
    expiry: str

client = genai.Client(api_key="AIzaSyBz-HUG4HHuJd8VNmmL0uH2Z8EEstSVO7w")


# image_path = "https://m.media-amazon.com/images/I/51mo8htwTHL.jpg"
# image_bytes = requests.get(image_path).content
# image = types.Part.from_bytes(
#   data=image_bytes, mime_type="image/jpeg"
# )
csv = pd.read_csv("dataset/train.csv")
my_file=""
uploaded_file_names={}
count = 0
for link in csv['image_link']:
    response = requests.get(link)
    filename = os.path.basename(link)
    uploaded_file = client.files.upload(file=f"images/train/{filename}")
    uploaded_file_names[filename] = uploaded_file.name
    count+=1
    if count>=10:
        break
    # my_file = client.files.get(name = uploaded_file.name)
    # break


prompt = "Get features like country (origin of brand), quality (premium/local/standard), expiry (food items expire quickly, preservative added food last for months, objects doesnt have expiry)"

# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents="Explain how AI works in a few words"
# )

for key in uploaded_file_names.keys:
    file = client.files.get(name = uploaded_file[key])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config={
            "response_mime_type": "application/json",
            "response_schema": list[Features],
        },
        contents=[file,prompt],
    )

    print(response.text)
