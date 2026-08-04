from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    stream: bool = False
    