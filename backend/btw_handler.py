import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilyAnswer

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

def handle_btw_query(query: str):
    """Handles off-topic (/btw) queries with optional Tavily web search, bypassing session history."""
    api_key = os.getenv("TAVILY_API_KEY")
    context = ""
    if api_key and ("latest" in query.lower() or "news" in query.lower() or "search" in query.lower()):
        try:
            tavily_tool = TavilyAnswer(max_results=3, tavily_api_key=api_key)
            context = str(tavily_tool.invoke({"query": query}))
        except Exception:
            pass

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant answering an off-topic (/btw) question. Web context if available:\n{context}"),
        ("human", "{query}")
    ])
    chain = prompt | llm
    return chain.stream({"context": context, "query": query})