
import os
import base64
from dotenv import load_dotenv
from langchain_classic.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

FAISS_DIR = "data/faiss_index"

# Load embeddings (CPU-safe)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Load FAISS vectorstore
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

# Initialize LLM with GROQ API key
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY")
)

def image_to_base64(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def answer(question: str, image_path: str = None) -> str:
    # Retrieve top 3 relevant documents
    docs = vectorstore.similarity_search(question, k=3)

    # Build context
    context = ""
    for d in docs:
        if d.metadata.get("type") == "text":
            context += d.metadata.get("original_content", "") + "\n"

    # Optional image (currently just converted to base64)
    image_base64 = image_to_base64(image_path)

    # Prepare prompt
    prompt = ChatPromptTemplate.from_template(
        "You are a medical assistant specialized in skin diseases.\n\n"
        "Context:\n{text}\n\n"
        "Question: {question}\n"
        "Answer accurately. If unsure, say 'I don't know'."
    )

    # Run LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text=context, question=question)

# Example usage
if __name__ == "__main__":
    q = "What is melanoma?"
    img = None  # Optional image path
    answer_text = answer(q, image_path=img)
    print("Answer:\n", answer_text)
