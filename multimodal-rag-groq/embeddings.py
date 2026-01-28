import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

with open("data/summarized_docs.pkl", "rb") as f:
    text_docs = pickle.load(f)

with open("data/image_docs.pkl", "rb") as f:
    image_docs = pickle.load(f)

docs = text_docs + image_docs

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)
db.save_local("data/faiss_index")

print("✅ FAISS index saved")
