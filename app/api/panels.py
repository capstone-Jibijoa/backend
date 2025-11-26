from fastapi import APIRouter, Depends, HTTPException
from app.services.panel_service import PanelService

router = APIRouter()

@router.get("/{panel_id}")
async def get_panel_details(
    panel_id: str,
    service: PanelService = Depends(PanelService)
):
    """패널 상세 정보 조회"""
    try:
        return await service.get_panel_details(panel_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Panel not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))