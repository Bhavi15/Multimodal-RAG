import os
import pickle
import uuid
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_core.documents import Document

# Directories
IMAGE_DIR = "data/images"
OUTPUT_FILE = "data/image_docs.pkl"

# Load BLIP model & processor (CPU only)
model_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id)

image_docs = []

for img_file in os.listdir(IMAGE_DIR):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(IMAGE_DIR, img_file)
        image = Image.open(img_path).convert("RGB")

        # Prepare input for BLIP
        inputs = processor(images=image, return_tensors="pt").to("cpu")

        # Generate caption
        output_ids = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)

        # Save as Document
        image_docs.append(
            Document(
                page_content=caption,
                metadata={
                    "id": str(uuid.uuid4()),
                    "type": "image",
                    "source": img_file,
                },
            )
        )
        print(f"✅ Captioned: {img_file} -> {caption}")

# Save all image documents
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(image_docs, f)

print(f"✅ All image captions saved to {OUTPUT_FILE}")
