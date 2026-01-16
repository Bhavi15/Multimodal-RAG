"""Streamlit interface for Multimodal Agentic RAG chatbot."""

import streamlit as st
from loguru import logger

from src.embeddings import VectorStoreManager
from src.graph.agent_graph import MultiAgentGraph


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# STYLES
# --------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.vectorstore_loaded = False
        st.session_state.vectorstore_manager = None
        st.session_state.agent_graph = None
        st.session_state.messages = []


@st.cache_resource
def load_vectorstore():
    try:
        manager = VectorStoreManager()
        manager.load_vectorstore()
        logger.info("Vectorstore loaded successfully")
        return manager
    except Exception as e:
        st.error(f"Failed to load vectorstore: {e}")
        return None


def display_message(role: str, content: str):
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"

    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {role.capitalize()}</strong><br>
        {content}
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# MAIN APP
# --------------------------------------------------

def main():
    init_session_state()

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown("## 🏥 Medical RAG Assistant")
        st.markdown("---")

        if not st.session_state.vectorstore_loaded:
            with st.spinner("Loading medical knowledge base..."):
                st.session_state.vectorstore_manager = load_vectorstore()
                if st.session_state.vectorstore_manager:
                    st.session_state.agent_graph = MultiAgentGraph(
                        st.session_state.vectorstore_manager
                    )
                    st.session_state.vectorstore_loaded = True

        if st.session_state.vectorstore_loaded:
            st.success("✅ Knowledge base loaded")
        else:
            st.error("❌ Knowledge base not loaded")
            st.stop()

        st.markdown("---")
        st.markdown("### Controls")

        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # ---------------- MAIN UI ----------------
    st.markdown(
        '<p class="main-header">Medical Knowledge Assistant</p>',
        unsafe_allow_html=True
    )

    # Display chat history
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])

    # Chat input
    query = st.chat_input(
        "Ask a medical or dermatology-related question..."
    )

    if query:
        # Store user message
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )
        display_message("user", query)

        # Generate response
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.agent_graph.run(query)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                display_message("assistant", answer)

            except Exception as e:
                logger.error(f"Chat error: {e}")
                st.error("Something went wrong while generating the answer.")


if __name__ == "__main__":
    main()
