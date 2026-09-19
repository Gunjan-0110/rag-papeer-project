import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from backend.models import GraphState
from backend.vector_store import search
from langchain_community.tools.tavily_search import TavilyAnswer

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

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
    return {"generation": response.content, "route": "direct_answer"}

def retrieve_node(state: GraphState):
    docs = search(state.query, session_id="default_session", k=4)
    doc_texts = [doc.page_content for doc in docs]
    return {"documents": doc_texts, "route": "retrieve"}

def verify_claim_node(state: GraphState):
    api_key = os.getenv("TAVILY_API_KEY")
    web_result = "Web verification search unavailable."
    if api_key:
        try:
            tool = TavilyAnswer(max_results=3, tavily_api_key=api_key)
            web_result = str(tool.invoke({"query": f"Verify claim against latest research: {state.query}"}))
        except Exception as e:
            web_result = f"Verification search failed: {e}"
    return {"documents": [web_result], "route": "verify_claim"}

def generate_node(state: GraphState):
    context = "\n\n".join(state.documents)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research assistant. Answer accurately based on the context provided:\n\nContext:\n{context}"),
        ("human", "{query}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": state.query})
    return {"generation": response.content}

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