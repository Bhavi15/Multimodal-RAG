from loguru import logger
from typing import List

from langchain_core.documents import Document
from src.embeddings import VectorStoreManager


class RetrievalAgent:
    """
    Lightweight retrieval agent.
    Uses FAISS vectorstore ONLY (no LLM).
    """

    def __init__(self, vectorstore_manager: VectorStoreManager):
        self.vs = vectorstore_manager

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        try:
            logger.info("Running FAISS similarity search...")
            docs = self.vs.search(query=query, k=k)
            logger.info(f"Retrieved {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
