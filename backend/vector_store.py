import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

load_dotenv()

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "embedding_cache")

# Initialize base Google free embeddings
base_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Setup Cache-backed embeddings to avoid redundant API calls
underlying_fs = LocalFileStore(CACHE_DIR)
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings, underlying_fs, namespace="google_embedding_001"
)

def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}"

def get_vectorstore(session_id: str) -> Chroma:
    """Returns a session-isolated local Chroma vector store using cached Google embeddings."""
    session_db_dir = os.path.join(DB_DIR, get_collection_name(session_id))
    
    vector_store = Chroma(
        collection_name=get_collection_name(session_id),
        embedding_function=cached_embeddings,
        persist_directory=session_db_dir
    )
    return vector_store

def add_paper(docs, session_id: str) -> None:
    """Adds document chunks to the session's local vector store."""
    store = get_vectorstore(session_id)
    store.add_documents(docs)

def search(query: str, session_id: str, k: int = 4):
    """Performs similarity search against the local store."""
    store = get_vectorstore(session_id)
    return store.similarity_search(query, k=k)