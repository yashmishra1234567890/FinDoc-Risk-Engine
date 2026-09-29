from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class GraphState(BaseModel):
    user_query: str
    
    # Decomposed sub-questions
    sub_questions: Optional[List[str]] = Field(default_factory=list)
    
    # Retrieved chunks for each sub-question
    retrieved_chunks: Optional[List[Dict]] = Field(default_factory=list)
    
    # Analysis results
    analysis_result: Optional[Dict] = Field(default_factory=dict)
    
    # Validation
    compliance_result: Optional[Dict] = Field(default_factory=dict)
    
    # Phase 4 numerical reasoning result (when applicable)
    numerical_result: Optional[Dict] = Field(default_factory=dict)

    # Phase 5 simple agentic retrieval
    retrieval_attempts: Optional[int] = 0
    refined_query: Optional[str] = None
    evidence_check: Optional[Dict] = Field(default_factory=dict)

    # Evidence provenance and structured verification surfaced to callers
    citations: Optional[List[Dict]] = Field(default_factory=list)
    verification: Optional[Dict] = Field(default_factory=dict)
    
    # Final answer
    final_answer: Optional[str] = None
