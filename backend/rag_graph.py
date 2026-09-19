import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from backend.models import GraphState
from backend.vector_store import search
from backend.btw_handler import search_web

load_dotenv()

# Initialize free-tier Google Gemini 1.5 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def retrieve_node(state: GraphState):
    """Retrieves relevant document chunks from the local Chroma vector store."""
    query = state.query
    docs = search(query, session_id="default_session", k=3)
    doc_texts = [doc.page_content for doc in docs]
    return {"documents": doc_texts}

def web_search_node(state: GraphState):
    """Performs a live web search using Tavily if the query requires external info/verification."""
    query = state.query
    web_result = search_web(query)
    return {"documents": [web_result]}

def router_node(state: GraphState):
    """Determines whether to query local docs or use web search based on content."""
    query = state.query.lower()
    # Simple heuristic router matching project logic: if asking for latest web info or claim verification
    if "latest" in query or "news" in query or "verify" in query or "internet" in query:
        return "web_search"
    return "retrieve"

def generate_node(state: GraphState):
    """Generates an answer using Google Gemini based on the collected context."""
    query = state.query
    documents = state.documents
    context = "\n\n".join(documents)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research assistant. Answer the user's question accurately using ONLY the provided context below.\n\nContext:\n{context}"),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query})
    return {"generation": response.content}

def create_rag_graph():
    """Builds and compiles the advanced LangGraph workflow with search routing."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    
    # Conditional routing from entry
    workflow.set_conditional_entry_point(
        router_node,
        {
            "retrieve": "retrieve",
            "web_search": "web_search"
        }
    )
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

app_graph = create_rag_graph()