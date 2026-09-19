import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

EMBEDDING_DIM = 768  # Google embedding-001 dimension
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "documents")

# Initialize Google free embeddings
base_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}"

def get_vectorstore(session_id: str) -> Chroma:
    """Returns a session-isolated local Chroma vector store using Google embeddings."""
    session_db_dir = os.path.join(DB_DIR, get_collection_name(session_id))
    
    vector_store = Chroma(
        collection_name=get_collection_name(session_id),
        embedding_function=base_embeddings,
        persist_directory=session_db_dir
    )
    return vector_store

def add_paper(docs, session_id: str) -> None:
    """Adds document chunks to the session's local vector store."""
    store = get_vectorstore(session_id)
    store.add_documents(docs)

def search(query: str, session_id: str, k: int = 4):
    """Performs similarity search using Google embeddings against the local store."""
    store = get_vectorstore(session_id)
    return store.similarity_search(query, k=k)