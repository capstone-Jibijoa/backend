import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Tuple, Dict, List
from fastapi.middleware.cors import CORSMiddleware

# [최적화] import
from hybrid_logic_optimized import classify_query_keywords # 1. LLM 캐싱 적용
from search_logic import initialize_embeddings # 4. 모델 미리 로딩
from search_logic_optimized import hybrid_search_parallel as hybrid_search # 5. 검색 병렬화
from analysis_logic_optimized import analyze_search_results_optimized as analyze_search_results # 2. DB 집계 분석
from db_logic_optimized import ( # 3. Connection Pool
    log_search_query,
    get_db_connection_context,
    init_db,
    cleanup_db
)

# FastAPI 애플리케이션 초기화
app = FastAPI(title="Multi-Table Hybrid Search API v3 (Optimized & Refactored)")

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4단계: 모델 미리 로딩 함수
def preload_models():
    """애플리케이션 시작 시 모든 AI 모델을 미리 로드합니다."""
    print("\n" + "="*70)
    print("🔄 모든 AI 모델을 미리 로드합니다...")
    # 1. KURE 임베딩 모델 로드
    initialize_embeddings()
    # 2. Claude LLM 모델 로드 (테스트 호출로 초기화 유도)
    classify_query_keywords("모델 로딩 테스트")
    print("✅ 모든 AI 모델 로드 완료")
    print("="*70 + "\n")

@app.on_event("startup")
async def startup_event():
    print("🚀 FastAPI 시작...")
    init_db()
    preload_models()

@app.on_event("shutdown")
async def shutdown_event():
    print("🧹 FastAPI 종료... Connection Pool 정리")
    cleanup_db()


class SearchQuery(BaseModel):
    query: str
    search_mode: str = "all"

# (SearchResponse, AnalysisRequest 등 다른 Pydantic 모델은 동일)
class SearchResponse(BaseModel):
    query: str
    classification: dict
    results: dict
    final_panel_ids: list[str]
    summary: dict

class AnalysisRequest(BaseModel):
    query: str
    search_mode: str = "weighted"

class AnalysisResponse(BaseModel):
    query: str
    total_count: int
    main_summary: str
    charts: list[dict]

# ====================================================================
# 🚀 [리팩토링] 공통 검색 로직 함수
# ====================================================================

async def _perform_common_search(query_text: str, search_mode: str) -> Tuple[Dict, List[str], Dict]:
    """
    /search와 /search-and-analyze가 공유하는 핵심 로직
    (LLM 분류, 병렬 검색, 로그 기록, 결과 포맷팅)
    
    Returns:
        Tuple[dict, list, dict]: (response, panel_id_list, classification)
    """
    print(f"\n{'='*70}")
    print(f"🔍 공통 검색 시작: {query_text} (모드: {search_mode})")
    print(f"{'='*70}\n")
    
    # 1단계: LLM 키워드 분류 (캐싱 없음)
    print("📌 1단계: LLM 키워드 분류")
    classification = classify_query_keywords(query_text)
    
    # 1.5단계: 분류 결과에서 limit 값 추출
    user_limit = classification.get('limit')
    print(f"💡 API: 감지된 Limit 값: {user_limit}")
    
    # 2단계: 하이브리드 검색 수행 (병렬 처리)
    search_results = hybrid_search(
        classification, 
        search_mode=search_mode,
        limit=user_limit
    )
    
    # 3단계: 검색 로그 기록 (Connection Pool 사용)
    if user_limit is not None and user_limit > 0:
        total_count = len(search_results['final_panel_ids'])
    elif search_mode == "all":
        total_count = search_results['results']['union']['count']
    else:
        total_count = len(search_results['final_panel_ids']) 
    
    log_search_query(query_text, total_count)
    
    # 4단계: 응답 구성
    display_fields = []
    for kw_info in classification.get('ranked_keywords', [])[:3]:
        field = kw_info.get('field', '')
        description = kw_info.get('description', '')
        if field and description:
            display_fields.append({
                'field': field,
                'label': description,
                'priority': kw_info.get('priority', 999)
            })
    
    effective_search_mode = search_mode
    if user_limit is not None and user_limit > 0:
        effective_search_mode = "quota"

    # 차트 분석에 사용할 panel_id_list 준비
    panel_id_list = []
    
    if effective_search_mode == "all":
        response = {
            "query": query_text,
            "classification": classification,
            "display_fields": display_fields,
            "source_counts": {
                "welcome_objective_count": len(search_results['panel_id1']),
                "welcome_subjective_count": len(search_results['panel_id2']),
                "qpoll_count": len(search_results['panel_id3'])
            },
            "results": {
                "intersection": {
                    "count": search_results['results']['intersection']['count'],
                    "panel_ids": search_results['results']['intersection']['panel_ids'][:100],
                    "top_scores": {
                        str(panel_id): search_results['results']['intersection']['scores'].get(panel_id, 0)
                        for panel_id in search_results['results']['intersection']['panel_ids'][:10]
                    }
                },
                "union": {
                    "count": search_results['results']['union']['count'],
                    "panel_ids": search_results['results']['union']['panel_ids'][:100],
                    "top_scores": {
                        str(panel_id): search_results['results']['union']['scores'].get(panel_id, 0)
                        for panel_id in search_results['results']['union']['panel_ids'][:10]
                    }
                },
                "weighted": {
                    "count": search_results['results']['weighted']['count'],
                    "panel_ids": search_results['results']['weighted']['panel_ids'][:100],
                    "weights": search_results['results']['weighted']['weights'],
                    "top_scores": {
                        str(panel_id): search_results['results']['weighted']['scores'].get(panel_id, 0)
                        for panel_id in search_results['results']['weighted']['panel_ids'][:10]
                    }
                }
            },
            "summary": {
                "search_mode": search_mode,
                "search_strategy": {
                    "welcome_objective": bool(classification.get('welcome_keywords', {}).get('objective')),
                    "welcome_subjective": bool(classification.get('welcome_keywords', {}).get('subjective')),
                    "qpoll": bool(classification.get('qpoll_keywords', {}).get('keywords'))
                },
                "ranked_keywords": classification.get('ranked_keywords', [])
            },
            "final_panel_ids": search_results['results']['weighted']['panel_ids'][:100]
        }
        # 'all' 모드의 기본값은 'weighted' 결과
        panel_id_list = search_results['results']['weighted']['panel_ids']
    
    else:
        # 단일 모드 결과 반환 (quota, weighted, union, intersection)
        final_panel_ids = search_results['final_panel_ids']
        match_scores = search_results['match_scores']
        
        response = {
            "query": query_text,
            "classification": classification,
            "display_fields": display_fields,
            "source_counts": {
                "welcome_objective_count": len(search_results['panel_id1']),
                "welcome_subjective_count": len(search_results['panel_id2']),
                "qpoll_count": len(search_results['panel_id3'])
            },
            "results": {
                effective_search_mode: {
                    "count": len(final_panel_ids),
                    "panel_ids": final_panel_ids[:100],
                    "top_scores": {
                        str(panel_id): match_scores.get(panel_id, 0)
                        for panel_id in final_panel_ids[:10]
                    }
                }
            },
            "summary": {
                "total_candidates": len(final_panel_ids),
                "search_mode": effective_search_mode,
                "search_strategy": {
                    "welcome_objective": bool(classification.get('welcome_keywords', {}).get('objective')),
                    "welcome_subjective": bool(classification.get('welcome_keywords', {}).get('subjective')),
                    "qpoll": bool(classification.get('qpoll_keywords', {}).get('keywords'))
                },
                "ranked_keywords": classification.get('ranked_keywords', [])
            },
            "final_panel_ids": final_panel_ids[:100]
        }
        # 분석을 위해 최대 5000개까지 사용 (기존 로직과 동일)
        panel_id_list = final_panel_ids[:5000]
    
    print(f"✅ 공통 검색 완료")
    return response, panel_id_list, classification

# ====================================================================
# 1. 메인 검색 API
# ====================================================================

@app.post("/api/search")
async def search_panels(search_query: SearchQuery):
    """
    자연어 질의를 받아 하이브리드 검색 수행 (리팩토링 적용)
    """
    valid_modes = ["all", "weighted", "union", "intersection"]
    if search_query.search_mode not in valid_modes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search_mode. Must be one of: {valid_modes}"
        )
    
    try:
        # 🚀 공통 검색 함수 호출
        response, _, _ = await _perform_common_search(
            search_query.query, 
            search_query.search_mode
        )
        
        return response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ /api/search 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")

# ====================================================================
# 2. 검색 + 분석 통합 API (NEW!)
# ====================================================================

@app.post("/api/search-and-analyze")
async def search_and_analyze(request: AnalysisRequest):
    """
    검색 + 분석을 한 번에 수행 (리팩토링 적용)
    """
    valid_modes = ["all", "weighted", "union", "intersection"]
    if request.search_mode not in valid_modes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search_mode. Must be one of: {valid_modes}"
        )
    
    try:
        # 🚀 1. 공통 검색 함수 호출
        response, panel_id_list, classification = await _perform_common_search(
            request.query, 
            request.search_mode
        )
        
        # 🚀 2. [고유 로직] 차트 데이터 생성 (DB 집계 쿼리 사용)
        print("\n📌 5단계: 차트 데이터 생성 (최적화)")
        analysis_result, status_code = analyze_search_results(
            request.query,
            classification,
            panel_id_list
        )
        
        if status_code == 200:
            response['charts'] = analysis_result.get('charts', [])
            response['analysis_summary'] = analysis_result.get('main_summary', '')
            print(f"✅ 차트 {len(response['charts'])}개 생성 완료")
        else:
            response['charts'] = []
            response['analysis_summary'] = '차트 생성 실패'
            print(f"⚠️  차트 생성 실패")
        
        print(f"\n✅ 검색+분석 완료")
        
        return response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ /api/search-and-analyze 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")

# ====================================================================
# 3. 디버깅 API - 키워드 분류만 테스트
# ====================================================================

@app.post("/api/debug/classify")
async def debug_classify(search_query: SearchQuery):
    """
    질의를 키워드로 분류만 하고 결과를 반환 (검색은 수행하지 않음)
    """
    try:
        classification = classify_query_keywords(search_query.query)
        return {
            "query": search_query.query,
            "classification": classification
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분류 실패: {str(e)}")

# ====================================================================
# 4. 패널 상세 정보 조회 API (Connection Pool 적용)
# ====================================================================

@app.get("/api/panels/{panel_id}")
async def get_panel_details(panel_id: str):
    """
    특정 panel_id의 패널 상세 정보를 조회합니다. (Connection Pool 적용)
    """
    try:
        with get_db_connection_context() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="데이터베이스 연결 실패")
            
            cur = conn.cursor()
            
            cur.execute("""
                SELECT panel_id, structured_data
                FROM welcome_meta2 
                WHERE panel_id = %s
            """, (panel_id,))
            
            result = cur.fetchone()
            
            if not result:
                cur.close()
                raise HTTPException(status_code=404, detail=f"panel_id {panel_id}를 찾을 수 없습니다.")
            
            panel_id_value, structured_data = result
            panel_data = {"panel_id": panel_id_value}
            if isinstance(structured_data, dict):
                panel_data.update(structured_data)
            
            cur.close()
            return panel_data
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 실패: {str(e)}")

# ====================================================================
# 5. 헬스체크
# ====================================================================

@app.get("/")
def read_root():
    return {
        "service": "Multi-Table Hybrid Search & Analysis API",
        "version": "3.0 (Optimized & Refactored)",
        "status": "running",
        "optimizations_applied": [
            "DB Connection Pool (psycopg2-pool)",
            "Parallel Search (ThreadPoolExecutor)",
            "DB Aggregate Queries (analysis_logic_optimized)"
        ],
        "optimizations_excluded": [
            "Redis LLM Caching"
        ],
        "endpoints": {
            "search": "/api/search",
            "search_and_analyze": "/api/search-and-analyze (추천!)",
            "classify": "/api/debug/classify",
            "panel_detail": "/api/panels/{panel_id}",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    """시스템 상태 확인 (Connection Pool 사용)"""
    try:
        with get_db_connection_context() as conn:
            db_status = "ok" if conn else "error"
        
        return {
            "status": "healthy",
            "database": db_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }