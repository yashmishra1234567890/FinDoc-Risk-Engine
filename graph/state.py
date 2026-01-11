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
    
    # Final answer
    final_answer: Optional[str] = None
