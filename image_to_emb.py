import os
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm  # progress bar (optional but useful)
# Load model and processor
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

# Load image
# )
results = []
csv = pd.read_csv("dataset/train.csv")

for index,row in tqdm(csv.iterrows(), desc="Embedding images"):
    try:
        filename = os.path.basename(row['image_link'])
        local_path = f"images/train/{filename}"
        image = Image.open(local_path)

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
            results.append({
                "sample_id": row['sample_id'],
                "embeddings": embeddings
            })
    except Exception as e:
        print(f"❌ Failed for {row['sample_id']}: {e}")

df = pd.DataFrame(results)
df.to_csv("dataset/image_embeddings.csv", index=False)