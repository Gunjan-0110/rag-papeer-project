import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# Initialize free, local Hugging Face embeddings (No API keys or cloud calls needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_store(session_id: str = "default_session"):
    """Get or create a Chroma vector store for a specific session."""
    session_db_dir = os.path.join(DB_DIR, session_id)
    os.makedirs(session_db_dir, exist_ok=True)
    
    return Chroma(
        persist_directory=session_db_dir,
        embedding_function=embeddings
    )

def add_paper(chunks, session_id: str = "default_session"):
    """Add processed paper chunks to the session vector store."""
    vector_store = get_vector_store(session_id)
    vector_store.add_documents(chunks)
    return True

def search(query: str, session_id: str = "default_session", k: int = 4):
    """Search the session vector store for relevant document chunks."""
    vector_store = get_vector_store(session_id)
    return vector_store.similarity_search(query, k=k)