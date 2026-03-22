import streamlit as st
import os
from backend.tpu_engine import TPULLMEngine
from backend.rag_pipeline import ResearchRAG

# Ensure data directories exist
os.makedirs("./data/docs", exist_ok=True)


# Cache the heavy TPU initialization so it doesn't reload on every UI click
@st.cache_resource
def load_backend():
    engine = TPULLMEngine()
    rag = ResearchRAG(engine)
    return rag


rag_system = load_backend()

st.title("🧠 Local TPU Research Assistant")
st.markdown("Upload documents and chat with Qwen2.5 powered by Google Cloud TPU v5e.")

# Sidebar for File Uploads
with st.sidebar:
    st.header("Document Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF to the database", type=["pdf"])

    if uploaded_file is not None:
        file_path = os.path.join("./data/docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing and embedding document..."):
            rag_system.process_and_store_document(file_path)
        st.success("Document added to knowledge base!")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking (TPU Inference)..."):
            response = rag_system.query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})