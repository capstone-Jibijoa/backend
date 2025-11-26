from pydantic import BaseModel
from typing import Optional, Dict, Any

class PanelResponse(BaseModel):
    panel_id: str
    structured_data: Optional[Dict[str, Any]] = None
    qpoll_data: Optional[Dict[str, Any]] = None