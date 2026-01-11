from pydantic import BaseModel
from typing import Optional

class QueryResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None
