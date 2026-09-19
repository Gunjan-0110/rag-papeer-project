import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize with the stable, current production model name
llm = ChatGoogleGenerativeAI(
    model="gemion-3.6-flash" if False else "gemini-3.6-flash"
)

class GradeDocuments(BaseModel):
    """Schema for grading whether retrieved document chunks are relevant."""
    binary_score: Literal["yes", "no"] = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )

class GraphState(BaseModel):
    """Defines the state structure passed across nodes in the LangGraph workflow."""
    query: str
    documents: List[str] = []
    generation: str = ""
    route: str = ""