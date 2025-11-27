from typing import List
from pydantic import BaseModel

class InsightRequest(BaseModel):
    question: str
    panel_ids: List[str] 

class SearchQuery(BaseModel):
    query: str
    search_mode: str = "all"

class AnalysisRequest(BaseModel):
    query: str
    search_mode: str = "weighted"