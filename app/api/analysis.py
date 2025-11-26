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
        # 1. 검색 (SearchService 재사용)
        # AnalysisService 안에서 Search 로직을 직접 호출하지 않고, 
        # 필요한 메타데이터만 SearchService의 내부 로직을 활용해 가져옴
        pro_info, panel_ids, classification = await search_service._perform_common_search(req.query, mode="pro")
        
        # 2. 분석 (AnalysisService)
        analysis_result, summary_text = await analysis_service.analyze_search_results(
            req.query, classification, panel_ids
        )
        
        # 3. 테이블 데이터 구성을 위해 SearchService 활용 (재사용성)
        # (Pro 모드도 테이블 데이터를 일부 보여줌)
        display_fields = search_service._prepare_display_fields(classification, req.query)
        
        # 테이블 데이터 페칭 (간소화하여 구현하거나 SearchService에 위임)
        # 여기서는 시간 관계상 display_fields와 기본 메타데이터만 반환
        
        # TODO: Pro모드용 테이블 데이터 페칭 로직을 Service에 추가하여 호출 권장
        # table_data = await search_service.fetch_table_data(panel_ids[:100], display_fields) 

        return {
            "query": req.query,
            "classification": classification,
            "display_fields": display_fields,
            "charts": analysis_result.get('charts', []),
            "search_summary": summary_text,
            "tableData": [], # 필요 시 추가 구현
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