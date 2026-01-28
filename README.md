# Multimodal RAG for Medical Documents

This project implements a multimodal Retrieval-Augmented Generation (RAG) system for medical documents, combining text and image understanding. It demonstrates how AI can process complex medical data, answer queries, and provide context-aware insights, making it an excellent showcase for AI/ML applications.

**Two Approaches**

**1. Single LLM with GROQ API**
This approach uses a single LLM powered by GROQ, combined with LLaVA and BLIP for image captioning. PDFs are processed to extract text and images, summarized intelligently, and stored in a FAISS vector database. Users can ask questions through a Streamlit interface, receiving AI-generated answers with relevant images. It is fast, CPU-friendly, and ideal for quick prototypes.

**2. Multi-Agent System with LangGraph**
This approach uses a modular, agent-based design with OpenAI GPT models. Separate agents handle text processing, image understanding, and retrieval-augmented generation, working together to answer complex queries. It is scalable, highly modular, and suitable for production-grade AI systems that require advanced reasoning.

**Features**

**PDF & Image Processing:** Extracts and summarizes content, captions medical images.

**Multimodal Retrieval:** Searches both text and images using semantic embeddings.

**Contextual Q&A:** Provides precise answers based on document content and related images.

**Interactive Interface:** Streamlit-based UI for easy querying and visualization.

**Scalable Architecture:** Supports both lightweight single-LLM and advanced multi-agent systems.

**Tech Stack**

Python, HuggingFace Transformers, LLaVA, BLIP, FAISS, LangChain, LangGraph, Streamlit, GROQ API, OpenAI GPT API, and vector embeddings.

**💻 Quick Start
**
Clone the repository:

git clone https://github.com/yourusername/Multimodal_RAG.git


**Install dependencies:**

cd Multimodal_RAG/approach1  # or approach2
pip install -r requirements.txt


**Configure API keys in .env:**

Approach 1: GROQ_API_KEY=your_groq_api_key
Approach 2: OPENAI_API_KEY=your_openai_api_key

**Run the app:**

streamlit run app.py   # Approach 1
python query_agents.py  # Approach 2

**📁 Project Structure**
Multimodal_RAG/
│
├── approach1/   # Single LLM + GROQ API
├── approach2/   # Multi-agent LangGraph system
├── data/        # PDFs, images, FAISS vector store
└── README.md    # Project overview

