from fastapi import APIRouter, Depends, HTTPException
from app.schemas.search import SearchQuery
from app.services.search_service import SearchService

router = APIRouter()

@router.post("/search")
async def search_panels(
    query: SearchQuery,
    service: SearchService = Depends(SearchService)
):
    """[Lite 모드] 패널 검색"""
    try:
        return await service.search_panels(query.query, query.search_mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/debug/classify")
async def debug_classify(query: SearchQuery):
    """검색어 파싱 디버깅용"""
    from llm import parse_query_intelligent
    try:
        return parse_query_intelligent(query.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))