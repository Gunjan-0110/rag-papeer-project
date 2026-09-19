import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from backend.models import GraphState, llm
from backend.vector_store import search
from langchain_community.tools.tavily_search import TavilySearchResults

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
    """Routes user queries between direct answers, local RAG retrieval, and claim verification."""
    query = state.query.lower()
    if "verify" in query or "claim" in query or "superseded" in query:
        return "verify_claim"
    elif "hello" in query or "hi" in query or "who are you" in query:
        return "direct_answer"
    else:
        return "retrieve"

def direct_answer_node(state: GraphState):
    response = llm.invoke(f"Answer directly: {state.query}")
    clean_text = extract_clean_text(response)
    return {"generation": clean_text, "route": "direct_answer"}

def retrieve_node(state: GraphState):
    active_session = getattr(state, "session_id", "default_session") or "default_session"
    query_lower = state.query.lower()
    
    # Check local RAG first
    docs = search(state.query, session_id=active_session, k=4)
    
    # If local search finds nothing, or if user asks for latest/recent/web info, use Tavily web search
    if not docs or "latest" in query_lower or "after" in query_lower or "recent" in query_lower or "update" in query_lower or "nasa" in query_lower:
        api_key = os.getenv("TAVILY_API_KEY")
        web_texts = []
        if api_key:
            try:
                tool = TavilySearchResults(max_results=3, tavily_api_key=api_key)
                results = tool.invoke({"query": state.query})
                for res in results:
                    snippet = res.get("content", "").strip()
                    url = res.get("url", "").strip()
                    if snippet:
                        web_texts.append(f"Snippet: {snippet}\nURL: {url}")
            except Exception as e:
                web_texts = [f"Web search failed: {e}"]
        
        fallback_docs = web_texts if web_texts else ["Web search unavailable."]
        return {"documents": fallback_docs, "route": "retrieve_with_web_fallback"}
    
    doc_texts = [doc.page_content for doc in docs]
    return {"documents": doc_texts, "route": "retrieve"}

def verify_claim_node(state: GraphState):
    api_key = os.getenv("TAVILY_API_KEY")
    web_texts = []
    if api_key:
        try:
            tool = TavilySearchResults(max_results=3, tavily_api_key=api_key)
            results = tool.invoke({"query": f"Verify claim against latest research: {state.query}"})
            for res in results:
                snippet = res.get("content", "").strip()
                url = res.get("url", "").strip()
                if snippet:
                    web_texts.append(f"Snippet: {snippet}\nURL: {url}")
        except Exception as e:
            web_texts = [f"Verification search failed: {e}"]
    return {"documents": web_texts if web_texts else ["Verification search unavailable."], "route": "verify_claim"}

def generate_node(state: GraphState):
    context = "\n\n".join(state.documents)
    route = getattr(state, "route", "retrieve")
    
    if route in ["retrieve_with_web_fallback", "verify_claim"]:
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
    
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("verify_claim", verify_claim_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_conditional_entry_point(
        router_node,
        {
            "direct_answer": "direct_answer",
            "retrieve": "retrieve",
            "verify_claim": "verify_claim"
        }
    )
    
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("verify_claim", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

app_graph = create_rag_graph()