from pydantic import BaseModel
from typing import Optional, List, Any

class Source(BaseModel):
    page_no: int
    snippet: str
    image_base64: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None
    sources: List[Source] = []
    metrics: Optional[dict] = {}
    ratios: Optional[dict] = {}
    compliance: Optional[dict] = {}
    market_context: Optional[dict] = {}
    federated_sources: Optional[dict] = {}
    temperature_risk: Optional[float] = None

