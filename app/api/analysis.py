from fastapi import APIRouter, Depends, HTTPException
from app.schemas.analysis import AnalysisRequest, InsightRequest
from app.services.analysis_service import AnalysisService
from app.services.search_service import SearchService

router = APIRouter()

@router.post("/search-and-analyze")
async def search_and_analyze(
    req: AnalysisRequest,
    analysis_service: AnalysisService = Depends(AnalysisService),
    search_service: SearchService = Depends(SearchService)
):
    """[Pro 모드] 검색 + 심층 분석"""
    try:
        # 1. 검색 수행
        pro_info, panel_ids, classification = await search_service._perform_common_search(req.query, mode="pro")
        
        # 2. 분석 수행
        analysis_result, summary_text = await analysis_service.analyze_search_results(
            req.query, classification, panel_ids
        )
        
        # 3. 필드 준비
        display_fields = search_service._prepare_display_fields(classification, req.query)
        
        # 4. 테이블 데이터 가져오기
        table_data = await search_service.get_table_data(
            panel_ids=panel_ids, 
            display_fields=display_fields, 
            classification=classification,  # 타겟 필드 정보 전달
            limit=100
        )

        return {
            "query": req.query,
            "classification": classification,
            "display_fields": display_fields,
            "charts": analysis_result.get('charts', []),
            "search_summary": summary_text,
            "tableData": table_data, 
            "total_count": len(panel_ids),
            "mode": "pro"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/insight/summary")
async def get_insight_summary(
    req: InsightRequest,
    service: AnalysisService = Depends(AnalysisService)
):
    """선택된 패널에 대한 즉시 요약"""
    try:
        return await service.get_insight_summary(req.panel_ids, req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))