from loguru import logger
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from src.embeddings import VectorStoreManager


class DeepResearchAgent:
    """
    Breaks a complex query into sub-queries
    and retrieves evidence for each.
    """

    def __init__(self, vectorstore_manager: VectorStoreManager):
        self.vs = vectorstore_manager
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    def _generate_subqueries(self, query: str) -> List[str]:
        prompt = f"""
Break the following medical research question into 3 focused sub-questions:

Question:
{query}

Return only the sub-questions as bullet points.
"""
        response = self.llm.invoke(prompt)
        lines = response.content.split("\n")
        return [l.strip("- ").strip() for l in lines if l.strip()]

    def research(self, query: str) -> List[Document]:
        try:
            logger.info(f"Starting deep research: {query}")
            subqueries = self._generate_subqueries(query)

            all_docs: List[Document] = []

            for sq in subqueries:
                logger.info(f"Researching sub-query: {sq}")
                docs = self.vs.search(sq, k=4)
                all_docs.extend(docs)

            logger.info(f"Deep research collected {len(all_docs)} documents")
            return all_docs

        except Exception as e:
            logger.error(f"Error in deep research: {e}")
            return []
