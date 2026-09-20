import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    ArxivLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path: str, title: str) -> List[Document]:
    """Loads a local PDF, TXT, MD, or MARKDOWN file from disk and chunks it."""
    lower_path = file_path.lower()
    
    if lower_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif lower_path.endswith(('.txt', '.md', '.markdown')):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        loader = PyPDFLoader(file_path) # Default fallback
    
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(raw_docs)
    
    for doc in chunked_docs:
        doc.metadata["title"] = title
    return chunked_docs

def load_from_url(url: str) -> List[Document]:
    """Loads documents from web URLs and chunks them."""
    loader = WebBaseLoader(url)
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(raw_docs)
    for doc in chunked_docs:
        doc.metadata["title"] = url
    return chunked_docs

def load_from_arxiv(query: str) -> List[Document]:
    """Loads papers directly from ArXiv ID or query string."""
    loader = ArxivLoader(query=query, load_max_docs=1)
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(raw_docs)
    for doc in chunked_docs:
        doc.metadata["title"] = f"ArXiv: {query}"
    return chunked_docs