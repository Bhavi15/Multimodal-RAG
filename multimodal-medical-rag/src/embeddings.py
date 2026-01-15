"""
Vector embeddings and FAISS vector store management.
Creates embeddings ONLY from stored processed_chunks.json
"""

from typing import List, Dict, Optional
import uuid
import json
import pickle
from pathlib import Path

from loguru import logger
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import settings


class VectorStoreManager:
    """Manages vector embeddings, FAISS storage, and document retrieval."""

    def __init__(self, api_key: str = settings.OPENAI_API_KEY):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=api_key
        )
        self.vectorstore: Optional[FAISS] = None

        # Stores full content keyed by doc_id
        self.doc_store: Dict[str, Dict] = {}

    # --------------------------------------------------
    # DOCUMENT CREATION
    # --------------------------------------------------

    def create_documents(self, summaries: List[Dict]) -> List[Document]:
        """Convert stored summaries into LangChain Documents."""
        documents: List[Document] = []

        for item in summaries:
            doc_id = str(uuid.uuid4())

            doc = Document(
                page_content=item["summary"],  # embeddings ONLY from summary
                metadata={
                    "id": doc_id,
                    "chunk_id": item["chunk_id"],
                    "type": item["chunk_type"],  # image | table | text
                    "page_number": item.get("page_number"),
                    "source": item.get("metadata", {}).get("source", "unknown"),
                }
            )

            documents.append(doc)

            # Persist full content separately
            self.doc_store[doc_id] = {
                "original_content": item["original_content"],
                "summary": item["summary"],
                "type": item["chunk_type"],
                "page_number": item.get("page_number"),
                "metadata": item.get("metadata", {})
            }

        logger.info(f"Created {len(documents)} documents for embedding")
        return documents

    # --------------------------------------------------
    # VECTORSTORE BUILD / LOAD
    # --------------------------------------------------

    def build_vectorstore(self, documents: List[Document]) -> FAISS:
        logger.info("Building FAISS vectorstore...")

        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        logger.info(f"Vectorstore built with {len(documents)} documents")
        return self.vectorstore

    def save_vectorstore(self, path: Path = settings.FAISS_INDEX_DIR):
        if not self.vectorstore:
            raise RuntimeError("Vectorstore not initialized")

        path.mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(str(path))

        with open(path / "doc_store.pkl", "wb") as f:
            pickle.dump(self.doc_store, f)

        logger.info(f"Vectorstore saved to {path}")

    def load_vectorstore(self, path: Path = settings.FAISS_INDEX_DIR):
        self.vectorstore = FAISS.load_local(
            str(path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        doc_store_path = path / "doc_store.pkl"
        if doc_store_path.exists():
            with open(doc_store_path, "rb") as f:
                self.doc_store = pickle.load(f)

        logger.info("Vectorstore loaded successfully")
        return self.vectorstore

    # --------------------------------------------------
    # 🔑 SEARCH API (USED BY AGENTS)
    # --------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Document]:
        """
        Unified search interface used by:
        - RetrievalAgent
        - DeepResearchAgent
        """

        if not self.vectorstore:
            raise RuntimeError("Vectorstore not loaded")

        if filter_type:
            return self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"type": filter_type}
            )

        return self.vectorstore.similarity_search(query, k=k)

    def get_full_content(self, doc_id: str) -> Optional[Dict]:
        """Retrieve full original content from doc_store."""
        return self.doc_store.get(doc_id)

# --------------------------------------------------
# MAIN (EMBEDDING PIPELINE)
# --------------------------------------------------

def main():
    logger.info("Starting embeddings from stored summaries...")

    summaries_path = settings.DATA_DIR / "processed_chunks.json"

    if not summaries_path.exists():
        logger.error(f"Summaries file not found: {summaries_path}")
        return

    with open(summaries_path, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    manager = VectorStoreManager()

    documents = manager.create_documents(summaries)
    manager.build_vectorstore(documents)
    manager.save_vectorstore()

    logger.info("Embeddings pipeline completed successfully")


if __name__ == "__main__":
    main()
