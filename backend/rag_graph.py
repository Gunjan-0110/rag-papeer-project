import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from backend.models import GraphState, llm
from backend.vector_store import search
from backend.btw_handler import handle_btw_query
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

load_dotenv()

def extract_clean_text(response):
    """Safely extracts a clean plain-text string from any complex response or list format."""
    content = response.content if hasattr(response, "content") else response
    
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        raw_text = "".join(text_parts) if text_parts else str(content)
    elif isinstance(content, dict):
        raw_text = content.get("text", str(content))
    else:
        raw_text = str(content)
        
    return raw_text.replace("\\n", "\n")

def router_node(state: GraphState):
    """Hybrid router: Uses fast rules for clear intents (0 API calls) and returns a plain string route."""
    query = state.query.strip()
    query_lower = query.lower()
    session_id = getattr(state, "session_id", "default_session") or "default_session"
    
    # Check if this session has active loaded documents
    has_docs = False
    try:
        from streamlit import session_state
        loaded_docs = session_state.get("loaded_docs_dict", {}).get(session_id, [])
        has_docs = len(loaded_docs) > 0
    except Exception:
        pass

    # 1. Instant Rule-Based Routing (Returns plain strings)
    if query.startswith("/") or "off-topic" in query_lower:
        return "command_a"
    elif any(kw in query_lower for kw in ["verify", "claim", "superseded"]):
        return "verify_claim"
    elif any(kw in query_lower for kw in ["latest", "recent", "update", "nasa"]) and not has_docs:
        return "web_fallback"
    elif any(kw in query_lower for kw in ["hello", "hi", "who are you"]) and not has_docs:
        return "direct_answer"
    
    # If documents are loaded and query mentions document/file terms or is a general question, prioritize local retrieval
    if has_docs:
        return "retrieve"

    # 2. LLM Classifier Fallback (Returns plain string)
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Classify the user query into exactly one of: 'command_a', 'verify_claim', 'web_fallback', or 'retrieve'. "
         "Output ONLY the category name."),
        ("human", "{query}")
    ])
    
    try:
        chain = classification_prompt | llm
        response = chain.invoke({"query": query})
        intent = extract_clean_text(response).strip().lower()
        if intent in ["command_a", "verify_claim", "web_fallback", "retrieve"]:
            return intent
    except Exception:
        pass
        
    # Default fallback string
    return "retrieve"

def direct_answer_node(state: GraphState):
    response = llm.invoke(f"Answer directly: {state.query}")
    clean_text = extract_clean_text(response)
    return {"generation": clean_text, "route": "direct_answer"}

def command_a_node(state: GraphState):
    """Ephemeral side-channel node for off-topic questions. Delegates directly to btw_handler.py."""
    query = state.query.strip()
    
    # Get stream response from your centralized handler
    stream_response = handle_btw_query(query)
    
    # Consume the stream generator into a clean text string for the graph state
    full_content = "".join([
        chunk.content for chunk in stream_response 
        if hasattr(chunk, "content")
    ])
    
    return {
        "generation": f"🔒 **CommandA Side-Channel (Ephemeral)**\n\n{full_content}",
        "route": "command_a",
        "documents": []
    }

def retrieve_node(state: GraphState):
    # Robust session_id extraction
    if isinstance(state, dict):
        active_session = state.get("session_id", "default_session")
        route = state.get("route", "retrieve")
        query_str = state.get("query", "")
    else:
        active_session = getattr(state, "session_id", "default_session") or "default_session"
        route = getattr(state, "route", "retrieve")
        query_str = getattr(state, "query", "")

    query_lower = query_str.lower()
    
    if route == "web_fallback" or any(kw in query_lower for kw in ["latest", "recent", "update", "nasa", "after"]):
        api_key = os.getenv("TAVILY_API_KEY")
        web_texts = []
        if api_key:
            try:
                tool = TavilySearchResults(max_results=3, tavily_api_key=api_key)
                results = tool.invoke({"query": query_str})
                for res in results:
                    snippet = res.get("content", "").strip()
                    url = res.get("url", "").strip()
                    if snippet:
                        web_texts.append(f"Snippet: {snippet}\nURL: {url}")
            except Exception as e:
                web_texts = [f"Web search failed: {e}"]
        return {"documents": web_texts if web_texts else ["Web search unavailable."], "route": "web_fallback"}

    # Check local RAG using the exact session_id where the paper was uploaded
    docs = search(query_str, session_id=active_session, k=4)
    
    if not docs:
        # Fallback to web if local search returns nothing
        api_key = os.getenv("TAVILY_API_KEY")
        web_texts = []
        if api_key:
            try:
                tool = TavilySearchResults(max_results=3, tavily_api_key=api_key)
                results = tool.invoke({"query": query_str})
                for res in results:
                    snippet = res.get("content", "").strip()
                    url = res.get("url", "").strip()
                    if snippet:
                        web_texts.append(f"Snippet: {snippet}\nURL: {url}")
            except Exception as e:
                web_texts = [f"Web search failed: {e}"]
        
        fallback_docs = web_texts if web_texts else ["No local documents found in this session, and web search is unavailable."]
        return {"documents": fallback_docs, "route": "web_fallback"}
    
    doc_texts = [doc.page_content for doc in docs]
    return {"documents": doc_texts, "route": "retrieve"}

def verify_claim_node(state: GraphState):
    web_api_key = os.getenv("TAVILY_API_KEY")
    verification_texts = []
    
    # 1. Track 1: Web Search (Tavily)
    if web_api_key:
        try:
            tool = TavilySearchResults(max_results=3, tavily_api_key=web_api_key)
            results = tool.invoke({"query": f"Verify claim against latest research: {state.query}"})
            for res in results:
                snippet = res.get("content", "").strip()
                url = res.get("url", "").strip()
                if snippet:
                    verification_texts.append(f"[Web Evidence] Snippet: {snippet}\nURL: {url}")
        except Exception as e:
            verification_texts.append(f"Web verification search failed: {e}")

    # 2. Track 2: ArXiv Pre-prints Search
    try:
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=1500)
        arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        arxiv_results = arxiv_tool.run(state.query)
        if arxiv_results:
            verification_texts.append(f"[ArXiv Research Evidence]\n{arxiv_results}")
    except Exception as e:
        verification_texts.append(f"ArXiv verification search failed: {e}")

    # Fallback if both fail
    final_docs = verification_texts if verification_texts else ["Verification search unavailable."]
    
    return {"documents": final_docs, "route": "verify_claim"}

def generate_node(state: GraphState):
    context = "\n\n".join(state.documents)
    route = getattr(state, "route", "retrieve")
    
    if route in ["web_fallback", "verify_claim"]:
        system_prompt = (
            "You are an advanced research assistant. Answer the user's question clearly and accurately "
            "using the provided web search snippets.\n\n"
            "At the end of your response, add a '### Sources' section where you list the relevant "
            "sources as clean, clickable Markdown links using the format: [Source Title or URL](URL).\n\n"
            "Web Search Context:\n{context}"
        )
    else:
        system_prompt = (
            "You are an expert research assistant. Answer accurately based strictly on the "
            "provided PDF context:\n\nContext:\n{context}"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": state.query})
    clean_text = extract_clean_text(response)
    return {"generation": clean_text}

def create_rag_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("command_a", command_a_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("verify_claim", verify_claim_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_conditional_entry_point(
        router_node,
        {
            "command_a": "command_a",
            "direct_answer": "direct_answer",
            "retrieve": "retrieve",
            "web_fallback": "retrieve",
            "verify_claim": "verify_claim"
        }
    )
    
    workflow.add_edge("command_a", END)
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("verify_claim", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

app_graph = create_rag_graph()