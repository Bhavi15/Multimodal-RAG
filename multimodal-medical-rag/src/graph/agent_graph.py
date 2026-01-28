from loguru import logger

from src.embeddings import VectorStoreManager
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.deep_research_agent import DeepResearchAgent
from src.agents.qa_agent import QAAgent


class MultiAgentGraph:
    def __init__(self, vectorstore_manager: VectorStoreManager):
        self.retrieval_agent = RetrievalAgent(vectorstore_manager)
        self.deep_agent = DeepResearchAgent(vectorstore_manager)
        self.qa_agent = QAAgent()

    def route(self, query: str) -> str:
        """
        Simple router:
        - short factual → quick
        - long analytical → deep
        """
        return "deep" if len(query.split()) > 12 else "quick"

    def run(self, query: str) -> str:
        mode = self.route(query)
        logger.info(f"Query routed to {mode} mode")

        if mode == "quick":
            docs = self.retrieval_agent.retrieve(query)
        else:
            docs = self.deep_agent.research(query)

        return self.qa_agent.answer(query, docs)


# --------------------------------------------------
# MAIN (USER INPUT)
# --------------------------------------------------

def main():
    manager = VectorStoreManager()
    manager.load_vectorstore()  # 🔑 load FAISS once

    graph = MultiAgentGraph(manager)

    print("\nMedical RAG Assistant (type 'exit' to quit)\n")

    while True:
        query = input("Query: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        answer = graph.run(query)
        print("\nAnswer:\n", answer)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
