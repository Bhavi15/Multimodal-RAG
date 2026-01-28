import asyncio
import base64
import json
from pathlib import Path
from typing import List, Dict
import pickle
from openai import OpenAI
from src.preprocessing import DocumentChunk
from config.settings import settings

class MultimodalSummarizer:
    """Summarize image chunks using Vision model, store text chunks as-is."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.vision_model = "gpt-4o-mini"

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def summarize_image(self, chunk: DocumentChunk, delay: float = 0.5) -> Dict:
        """Summarize a single image chunk with throttling and logging."""
        try:
            b64_image = self.encode_image(chunk.content)

            def sync_call():
                return self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a medical image analyst. Describe key clinical features."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this medical image in detail, focus on key features and relevance."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                            ]
                        }
                    ],
                    max_tokens=200
                )

            response = await asyncio.to_thread(sync_call)
            summary = response.choices[0].message.content.strip()

            # Print log
            print(f"[Page {chunk.page_number}] Image chunk summarized: {summary[:100]}...")

            await asyncio.sleep(delay)  # Throttle

            return {
                "chunk_id": chunk.chunk_id,
                "chunk_type": "image",
                "original_content": getattr(chunk, "text_content", ""),  # any extracted text
                "summary": summary,
                "page_number": chunk.page_number,
                "metadata": chunk.metadata
            }

        except Exception as e:
            print(f"Error summarizing image {chunk.content}: {e}")
            return {
                "chunk_id": chunk.chunk_id,
                "chunk_type": "image",
                "original_content": getattr(chunk, "text_content", ""),
                "summary": f"Error: {e}",
                "page_number": chunk.page_number,
                "metadata": chunk.metadata
            }

    async def process_chunks(self, chunks: List[DocumentChunk], delay: float = 0.5) -> List[Dict]:
        """Process all chunks: store text as-is, summarize images."""
        results = []
        for chunk in chunks:
            if chunk.chunk_type == "text":
                print(f"[Page {chunk.page_number}] Text chunk stored as-is: {chunk.content[:100]}...")
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": "text",
                    "original_content": chunk.content,
                    "summary": chunk.content,  # same as original
                    "page_number": chunk.page_number,
                    "metadata": chunk.metadata
                })
            elif chunk.chunk_type == "image":
                result = await self.summarize_image(chunk, delay)
                results.append(result)
        return results


async def main():
    # Load chunks
    chunks_file = settings.DATA_DIR / "chunks.pkl"
    with open(chunks_file, "rb") as f:
        chunks: List[DocumentChunk] = pickle.load(f)

    summarizer = MultimodalSummarizer()
    processed_chunks = await summarizer.process_chunks(chunks, delay=0.5)

    # Save all chunks (text as-is + summarized images)
    output_file = settings.DATA_DIR / "processed_chunks.json"
    output_file.write_text(json.dumps(processed_chunks, indent=2), encoding="utf-8")
    print(f"\nAll chunks saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
