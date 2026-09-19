import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilyAnswer

load_dotenv()

def search_web(query: str) -> str:
    """Uses Tavily's free tier to search the web for claim verification or out-of-context queries."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Tavily API key is missing. Web search is unavailable."
        
    try:
        tavily_tool = TavilyAnswer(max_results=3, tavily_api_key=api_key)
        result = tavily_tool.invoke({"query": query})
        return str(result)
    except Exception as e:
        return f"Web search failed: {e}"