from loguru import logger
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


class QAAgent:
    """
    Generates final answer using retrieved context.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    def answer(self, query: str, documents: List[Document]) -> str:
        logger.info(f"Answering question: {query}")

        context = "\n\n".join(
            doc.page_content for doc in documents[:6]
        )

        prompt = f"""
You are a medical AI assistant.

Context:
{context}

Question:
{query}

Answer clearly, accurately, and safely.
"""

        response = self.llm.invoke(prompt)
        return response.content
