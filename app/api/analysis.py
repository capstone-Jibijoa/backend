from fastapi import APIRouter, Depends, HTTPException
from app.schemas.analysis import AnalysisRequest, InsightRequest
from app.services.analysis_service import AnalysisService
from app.services.search_service import SearchService
from app.utils.common import filter_merged_panels

router = APIRouter()

@router.post("/search-and-analyze")
async def search_and_analyze(
    req: AnalysisRequest,
    analysis_service: AnalysisService = Depends(AnalysisService),
    search_service: SearchService = Depends(SearchService)
):
    """[Pro 모드] 검색 + 심층 분석"""
    try:
        # 1. 검색 수행 (여기서 388명 등 '더러운' 데이터가 넘어옴)
        pro_info, panel_ids, classification = await search_service._perform_common_search(req.query, mode="pro")
        
        # 1.5. 테이블 데이터용 필터링 (메모리 상 검증) 수행
        raw_panels_data = search_service.panel_repo.get_panels_data_from_db(panel_ids[:2000])
        
        # LLM이 추출한 필터(나이, 지역 등) 적용
        filters = classification.get('demographic_filters', {})
        if 'region_major' in filters:
             filters['region'] = filters.pop('region_major') # 필드명 보정

        filtered_data = filter_merged_panels(raw_panels_data, filters)
        
        # 필터링된 ID 리스트로 교체
        filtered_panel_ids = [p['panel_id'] for p in filtered_data]

        # 2. 분석 수행 (필터링된 데이터 사용)
        # (analyze_search_results 내부에서도 필터링을 하지만, 위에서 했으므로 중복되더라도 안전함)
        analysis_result, summary_text = await analysis_service.analyze_search_results(
            req.query, classification, filtered_panel_ids
        )
        
        # 3. 필드 준비
        display_fields = search_service._prepare_display_fields(classification, req.query)
        
        # 4. 테이블 데이터 가져오기 (필터링된 ID + 사용자 Limit 적용)
        user_limit = classification.get('limit', 100)
        
        table_data = await search_service.get_table_data(
            panel_ids=filtered_panel_ids, # [수정] filtered_panel_ids 사용
            display_fields=display_fields, 
            classification=classification,
            limit=user_limit # user_limit 적용 (기존 100 하드코딩 제거)
        )

        return {
            "query": req.query,
            "classification": classification,
            "display_fields": display_fields,
            "charts": analysis_result.get('charts', []),
            "search_summary": summary_text,
            "tableData": table_data, 
            "total_count": len(filtered_panel_ids), # [수정] 필터링된 개수 반환
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