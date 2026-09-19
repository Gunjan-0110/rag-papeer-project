import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI

SESSION_FILE = os.path.join(os.path.dirname(__file__), "..", "sessions.json")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

def generate_session_title(first_message: str) -> str:
    """Generates a 3-5 word session title using Gemini."""
    try:
        prompt = f"Summarize this user prompt into a short, descriptive chat session title of 3 to 5 words maximum. No quotes: '{first_message}'"
        res = llm.invoke(prompt)
        return res.content.strip()
    except Exception:
        return "Research Session"

def load_sessions():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    return {"default_session": "Default Research Chat"}

def save_sessions(sessions_dict):
    with open(SESSION_FILE, "w") as f:
        json.dump(sessions_dict, f, indent=2)