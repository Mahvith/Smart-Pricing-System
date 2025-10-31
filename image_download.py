import pandas as pd
import os
import requests
from tqdm import tqdm  # progress bar (optional but useful)

os.makedirs("images/test", exist_ok=True)

csv = pd.read_csv("dataset/test.csv")

for link in tqdm(csv["image_link"], desc="Downloading images"):
    try:
        filename = os.path.basename(link)
        filepath = f"images/test/{filename}"

        # Skip if already downloaded
        if os.path.exists(filepath):
            continue

        # Download with timeout and stream
        response = requests.get(link, timeout=10, stream=True)
        response.raise_for_status()  # Raise exception for 4xx/5xx

        # Write to file safely
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {link}: {e}")
