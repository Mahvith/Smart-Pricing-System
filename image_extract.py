import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError
import requests
import pandas as pd
import time
import dotenv

class Features(BaseModel):
    country: str
    quality: str
    expiry: str

client = genai.Client(api_key=dotenv.get_key(".env", "API_KEY"))


# image_path = "https://m.media-amazon.com/images/I/51mo8htwTHL.jpg"
# image_bytes = requests.get(image_path).content
# image = types.Part.from_bytes(
#   data=image_bytes, mime_type="image/jpeg"
# )
csv = pd.read_csv("dataset/train.csv")
# my_file=""
# uploaded_file_names={}
# count = 0
# for link in csv['image_link']:
#     response = requests.get(link)
#     filename = os.path.basename(link)
#     uploaded_file = client.files.upload(file=f"images/train/{filename}")
#     uploaded_file_names[filename] = uploaded_file.name
#     count+=1
#     if count>=10:
#         break
    # my_file = client.files.get(name = uploaded_file.name)
    # break


prompt = "Get features like country: origin of brand, quality :premium/local/standard, expiry : days,months,years,N/A"

# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents="Explain how AI works in a few words"
# )
results = []
# count=0
for link in csv["image_link"]:
    # count+=1
    # if count>=10:
    #     break
    try:
        filename = os.path.basename(link)
        local_path = f"images/train/{filename}"
        image_bytes=None
        with open(local_path, 'rb') as f:
            image_bytes = f.read()

        # Download image if not already
        # if not os.path.exists(local_path):
        #     r = requests.get(link, timeout=10)
        #     r.raise_for_status()
        #     with open(local_path, "wb") as f:
        #         f.write(r.content)

        # Upload to Gemini
        # uploaded_file = client.files.upload(file=local_path)

        # Generate JSON response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config={
                "response_mime_type": "application/json",
                "response_schema": Features,
            },
            contents=[types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                prompt
            ],
        )

        # Parse response text
        try:
            data = json.loads(response.text)
            parsed = Features(**data)
            results.append({
                "image_link": link,
                "country": parsed.country,
                "quality": parsed.quality,
                "expiry": parsed.expiry,
            })
            print(f"✅ {filename} → {parsed}")
            time.sleep(0.1)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"⚠️ Parse error for {filename}: {e}")
            results.append({
                "image_link": link,
                "country": None,
                "quality": None,
                "expiry": None,
            })
        

    except Exception as e:
        print(f"❌ Failed for {link}: {e}")
        results.append({
            "image_link": link,
            "country": None,
            "quality": None,
            "expiry": None,
        })

# --- Save results ---
df = pd.DataFrame(results)
df.to_csv("dataset/image_features.csv", index=False)
print("\n✅ Saved to dataset/train_with_features.csv")
