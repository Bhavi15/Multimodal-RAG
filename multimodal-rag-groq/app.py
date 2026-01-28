import streamlit as st
from rag import answer  # The function from rag.py
from PIL import Image
import io

st.set_page_config(page_title="📄 Multimodal RAG (CPU-Safe)", layout="wide")

st.title("📄 Multimodal RAG (Text + Image)")

st.markdown(
    """
This app allows you to ask questions about your medical PDFs and optionally provide an image.
The system will use the pre-generated FAISS embeddings and a LLM (Groq) to answer your question.
"""
)

# Text input for question
question = st.text_input("Ask a question about your PDFs:")

# Optional image upload
uploaded_file = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])

# Display uploaded image
image_path = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    # Save temporarily for passing to answer function
    image_path = f"temp_uploaded_image.{uploaded_file.name.split('.')[-1]}"
    image.save(image_path)

# Submit button
if st.button("Get Answer") and question:
    with st.spinner("Generating answer..."):
        try:
            # Call your rag.py function
            result = answer(question, image_path=image_path)
        except Exception as e:
            st.error(f"Error: {e}")
            result = None

    if result:
        st.success("✅ Answer:")
        st.write(result)

# Clean up temp file if exists
import os
if image_path and os.path.exists(image_path):
    os.remove(image_path)
