from pydantic import BaseModel
from typing import Optional, Dict, Any

# 추후 패널 상세 조회 응답을 명세화할 때 사용
class PanelResponse(BaseModel):
    panel_id: str
    structured_data: Optional[Dict[str, Any]] = None
    qpoll_data: Optional[Dict[str, Any]] = None