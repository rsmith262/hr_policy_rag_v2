from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    input: str
    session_id: Optional[str] = None

class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    url: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    citations: List[Citation] = []
