import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Papeer - Research Paper Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Papeer: Research Paper Assistant")
st.markdown("Upload research papers and chat with them using **Google Gemini 1.5-flash** and LangGraph.")

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"

# Sidebar for configuration and uploads
with st.sidebar:
    st.header("⚙️ Setup & Upload")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY missing in .env file!")
    else:
        st.success("✅ Google API Key loaded")
        
    st.divider()
    st.subheader("📁 Upload Research PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        docs_dir = os.path.join(os.path.dirname(__file__), "documents")
        os.makedirs(docs_dir, exist_ok=True)
        file_path = os.path.join(docs_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"Saved {uploaded_file.name}!")
        
        if st.button("🚀 Process Paper"):
            with st.spinner("Processing document chunks & local vector embeddings..."):
                try:
                    from backend.vector_store import add_paper
                    from backend.paper_loader import load_and_split_pdf
                    
                    # Use our paper loader utility
                    chunked_docs = load_and_split_pdf(file_path, uploaded_file.name)
                    add_paper(chunked_docs, st.session_state.session_id)
                    
                    st.success("Paper successfully indexed!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")

# Main Chat Interface
st.subheader("💬 Chat with your Research Papers")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your research papers..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and retrieving context..."):
            try:
                from backend.rag_graph import app_graph
                
                # Invoke the LangGraph workflow
                initial_state = {"query": query, "documents": [], "generation": "", "route": ""}
                result = app_graph.invoke(initial_state)
                
                response = result.get("generation", "I couldn't generate a response.")
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)