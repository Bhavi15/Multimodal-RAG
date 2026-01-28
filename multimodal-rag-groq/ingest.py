import os
import uuid
import logging
import re
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document

logging.getLogger("pdfminer").setLevel(logging.ERROR)

PDF_DIR = "data/pdfs"
IMAGE_DIR = "data/images"

os.makedirs(IMAGE_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("**", "")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

documents = []

for pdf in os.listdir(PDF_DIR):
    if not pdf.endswith(".pdf"):
        continue

    print(f"📄 Processing {pdf}")

    elements = partition_pdf(
        filename=os.path.join(PDF_DIR, pdf),
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=IMAGE_DIR,
    )

    for el in elements:
        if hasattr(el, "text") and el.text.strip():
            documents.append(
                Document(
                    page_content=clean_text(el.text),
                    metadata={
                        "id": str(uuid.uuid4()),
                        "type": "text",
                        "source": pdf,
                    },
                )
            )

print(f"✅ Extracted {len(documents)} text chunks")

# Save raw text docs for next step
import pickle
with open("data/raw_docs.pkl", "wb") as f:
    pickle.dump(documents, f)
