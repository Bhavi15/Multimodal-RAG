import os, pickle
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # loads GROQ_API_KEY from .env

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY not set in your .env file!")

# Initialize Groq LLM with API key
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=GROQ_API_KEY
)

prompt = ChatPromptTemplate.from_template(
    "Summarize the following medical text clearly:\n{text}"
)

with open("data/raw_docs.pkl", "rb") as f:
    docs = pickle.load(f)

summaries = []

for d in docs:
    summary = llm.invoke(prompt.format(text=d.page_content)).content
    d.page_content = summary
    summaries.append(d)

with open("data/summarized_docs.pkl", "wb") as f:
    pickle.dump(summaries, f)

print("✅ Text summarized")
