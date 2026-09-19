from pydantic import BaseModel, Field
from typing import List, Literal

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