from pydantic import BaseModel
from typing import List, Optional

class InsightRequest(BaseModel):
    """
    인사이트 요약 요청 모델 (Lite 모드)
    """
    question: str
    panel_ids: List[str]

class AnalysisRequest(BaseModel):
    """
    심층 분석 요청 모델 (Pro 모드)
    """
    query: str
    search_mode: str = "weighted"