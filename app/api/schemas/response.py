from pydantic import BaseModel
from typing import Optional, List, Any

class Source(BaseModel):
    page_no: int
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None
    sources: List[Source] = []
    metrics: Optional[dict] = {}
    ratios: Optional[dict] = {}

