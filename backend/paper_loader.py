import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_and_split_pdf(file_path: str, title: str) -> list[Document]:
    """Loads a PDF file from disk and splits it into manageable text chunks with metadata."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")
        
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(raw_docs)
    
    # Attach title metadata to each chunk
    for doc in chunked_docs:
        doc.metadata["title"] = title
        
    return chunked_docs