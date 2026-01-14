"""
Fast PDF preprocessing using PyMuPDF.
- Extracts text, images, and tables
- Stores ALL text chunks in a single file
- Stores tables as separate files
- Stores images as separate files
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image
import io
from loguru import logger

from config.settings import settings


# -----------------------------
# Data Model
# -----------------------------
@dataclass
class DocumentChunk:
    content: str
    chunk_type: str  # "text" | "table" | "image"
    page_number: int
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    file_path: Optional[str] = None


# -----------------------------
# PDF Processor
# -----------------------------
class FastPDFProcessor:
    """Fast PDF processing using PyMuPDF."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        extract_images: bool = True,
        extract_tables: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.image_counter = 0

        # Single text output file
        self.text_output_file = settings.TEXT_DIR / "all_text_chunks.txt"
        self.text_output_file.parent.mkdir(parents=True, exist_ok=True)

        # Clear file at start
        self.text_output_file.write_text("", encoding="utf-8")

    # -----------------------------
    # Public API
    # -----------------------------
    def process_directory(self, directory: Path = settings.PDF_DIR) -> List[DocumentChunk]:
        logger.info(f"Scanning PDF directory: {directory}")
        all_chunks: List[DocumentChunk] = []

        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")

        for pdf_path in pdf_files:
            try:
                all_chunks.extend(self.process_pdf(pdf_path))
            except Exception as e:
                logger.exception(f"Failed processing {pdf_path.name}: {e}")

        logger.info(f"Total chunks extracted: {len(all_chunks)}")
        return all_chunks

    def process_pdf(self, pdf_path: Path) -> List[DocumentChunk]:
        logger.info(f"Processing PDF: {pdf_path.name}")
        chunks: List[DocumentChunk] = []

        doc = fitz.open(pdf_path)

        for page_number, page in enumerate(doc, start=1):
            chunks.extend(self._extract_text(page, page_number, pdf_path.stem))

            if self.extract_images:
                chunks.extend(self._extract_images(page, page_number, pdf_path.stem))

            if self.extract_tables:
                chunks.extend(self._extract_tables(page, page_number, pdf_path.stem))

        doc.close()
        return chunks

    # -----------------------------
    # Text Extraction (ONE FILE)
    # -----------------------------
    def _extract_text(self, page, page_number: int, doc_name: str) -> List[DocumentChunk]:
        text = page.get_text("text").strip()
        if not text:
            return []

        words = text.split()
        chunks: List[DocumentChunk] = []

        current_words = []
        current_length = 0
        chunk_index = 1

        for word in words:
            word_len = len(word) + 1

            if current_length + word_len > self.chunk_size and current_words:
                chunk_text = " ".join(current_words)
                chunks.append(
                    self._append_text_chunk(
                        chunk_text, doc_name, page_number, chunk_index
                    )
                )

                overlap_count = int(len(current_words) * (self.chunk_overlap / self.chunk_size))
                current_words = current_words[-overlap_count:] if overlap_count > 0 else []
                current_length = sum(len(w) + 1 for w in current_words)
                chunk_index += 1

            current_words.append(word)
            current_length += word_len

        if current_words:
            chunk_text = " ".join(current_words)
            chunks.append(
                self._append_text_chunk(
                    chunk_text, doc_name, page_number, chunk_index
                )
            )

        return chunks

    def _append_text_chunk(
        self, text: str, doc_name: str, page_number: int, chunk_index: int
    ) -> DocumentChunk:
        chunk_id = f"{doc_name}_page{page_number}_text{chunk_index}"

        with self.text_output_file.open("a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"CHUNK_ID   : {chunk_id}\n")
            f.write(f"SOURCE     : {doc_name}\n")
            f.write(f"PAGE       : {page_number}\n")
            f.write(f"CHAR_COUNT : {len(text)}\n")
            f.write("-" * 80 + "\n")
            f.write(text + "\n")

        return DocumentChunk(
            content=text,
            chunk_type="text",
            page_number=page_number,
            metadata={
                "source": doc_name,
                "page": page_number,
                "char_count": len(text),
            },
            chunk_id=chunk_id,
            file_path=str(self.text_output_file),
        )

    # -----------------------------
    # Image Extraction
    # -----------------------------
    def _extract_images(self, page, page_number: int, doc_name: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        images = page.get_images(full=True)

        for img in images:
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]

                self.image_counter += 1
                chunk_id = f"{doc_name}_page{page_number}_img{self.image_counter}"
                image_path = settings.IMAGE_DIR / f"{chunk_id}.png"

                image = Image.open(io.BytesIO(image_bytes))
                image.save(image_path)

                chunks.append(
                    DocumentChunk(
                        content=str(image_path),
                        chunk_type="image",
                        page_number=page_number,
                        metadata={
                            "source": doc_name,
                            "page": page_number,
                            "format": base_image.get("ext", "unknown"),
                        },
                        chunk_id=chunk_id,
                        file_path=str(image_path),
                    )
                )

            except Exception as e:
                logger.error(f"Image extraction failed on page {page_number}: {e}")

        return chunks

    # -----------------------------
    # Table Extraction
    # -----------------------------
    def _extract_tables(self, page, page_number: int, doc_name: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        blocks = page.get_text("blocks")
        table_index = 1

        for block in blocks:
            text = block[4].strip()
            if not text:
                continue

            if "\t" in text or "|" in text or text.count("  ") > 3:
                chunk_id = f"{doc_name}_page{page_number}_table{table_index}"
                file_path = settings.TABLE_DIR / f"{chunk_id}.txt"
                file_path.write_text(text, encoding="utf-8")

                chunks.append(
                    DocumentChunk(
                        content=text,
                        chunk_type="table",
                        page_number=page_number,
                        metadata={
                            "source": doc_name,
                            "page": page_number,
                            "bbox": block[:4],
                        },
                        chunk_id=chunk_id,
                        file_path=str(file_path),
                    )
                )

                table_index += 1

        return chunks


# -----------------------------
# Standalone Run
# -----------------------------
def main():
    logger.info("Starting PDF preprocessing")

    processor = FastPDFProcessor()
    chunks = processor.process_directory()

    print("\n" + "=" * 60)
    print("PREPROCESSING RESULTS")
    print("=" * 60)
    print(f"Total chunks : {len(chunks)}")
    print(f"Text chunks  : {len([c for c in chunks if c.chunk_type == 'text'])}")
    print(f"Image chunks : {len([c for c in chunks if c.chunk_type == 'image'])}")
    print(f"Table chunks : {len([c for c in chunks if c.chunk_type == 'table'])}")
    print(f"\nText output file: {processor.text_output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
