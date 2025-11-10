import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from hybrid_logic import classify_query_keywords  # 키워드 분류 (ranked_keywords 포함)
from search_logic import hybrid_search  # 통합 검색 함수
from analysis_logic import analyze_search_results  # LLM 없는 분석 함수
from db_logic import log_search_query, get_db_connection

# FastAPI 애플리케이션 초기화
app = FastAPI(title="Multi-Table Hybrid Search API v3")


class SearchQuery(BaseModel):
    query: str
    search_mode: str = "all"


class SearchResponse(BaseModel):
    query: str
    classification: dict
    results: dict
    final_panel_ids: list[str]
    summary: dict

class AnalysisRequest(BaseModel):
    query: str
    search_mode: str = "weighted"  # 분석에는 weighted 사용 권장

class AnalysisResponse(BaseModel):
    query: str
    total_count: int
    main_summary: str
    charts: list[dict]

# ====================================================================
# 1. 메인 검색 API
# ====================================================================

@app.post("/api/search", response_model=SearchResponse)
async def search_panels(search_query: SearchQuery):
    """
    자연어 질의를 받아 Welcome/QPoll 하이브리드 검색 수행
    
    검색 모드:
    - all (기본): 교집합, 합집합, 가중치 모두 반환
    - intersection: 교집합만 (모든 조건 만족)
    - union: 합집합만 (하나라도 조건 만족)
    - weighted: 가중치 기반만 (객관식 40%, 주관식 30%, QPoll 30%)
    
    프로세스:
    1. LLM이 질의를 Welcome(객관/주관)/QPoll 키워드로 분류 + ranked_keywords 추출
    2. Welcome 객관식 → PostgreSQL 검색 (panel_id1)
    3. Welcome 주관식 → Qdrant 임베딩 검색 (panel_id2)
    4. QPoll → Qdrant 임베딩 검색 (panel_id3)
    5. 3가지 방식으로 결과 통합 및 정렬
    * 참고: 쿼리에 "NN명"이 포함되면 'limit'가 활성화되며,
      search_mode와 관계없이 'quota' 우선순위(교집합 > 가중치)로 최종 결과 반환
    """
    query_text = search_query.query
    search_mode = search_query.search_mode
    
    valid_modes = ["all", "weighted", "union", "intersection"]
    if search_mode not in valid_modes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search_mode. Must be one of: {valid_modes}"
        )
    
    try:
        print(f"\n{'='*70}")
        print(f"🔍 검색 요청: {query_text}")
        print(f"📊 검색 모드: {search_mode}")
        print(f"{'='*70}\n")
        
        # 1단계: LLM 키워드 분류 (ranked_keywords 포함)
        print("📌 1단계: LLM 키워드 분류")
        classification = classify_query_keywords(query_text)
        
        # [신규] 1.5단계: 분류 결과에서 limit 값 추출
        user_limit = classification.get('limit')
        print(f"💡 API: 감지된 Limit 값: {user_limit}")
        
        # [수정] 2단계: 하이브리드 검색 수행 (limit 인자 전달)
        search_results = hybrid_search(
            classification, 
            search_mode=search_mode,
            limit=user_limit
        )
        
        # [수정] 3단계: 검색 로그 기록 (limit 우선 확인)
        if user_limit is not None and user_limit > 0:
            total_count = len(search_results['final_panel_ids'])
        elif search_mode == "all":
            total_count = search_results['results']['union']['count']
        else:
            total_count = len(search_results['final_panel_ids']) 
        
        log_search_query(query_text, total_count)
        
        # 4단계: 응답 구성 (limit 우선 확인)
        # ✅ 프론트엔드 표시용 필드 추출 (ranked_keywords 기반)
        display_fields = []
        for kw_info in classification.get('ranked_keywords', [])[:3]:  # 상위 3개만
            field = kw_info.get('field', '')
            description = kw_info.get('description', '')
            if field and description:
                display_fields.append({
                    'field': field,
                    'label': description,
                    'priority': kw_info.get('priority', 999)
                })
        

        
        # 사용자가 'limit'를 지정했다면, search_mode가 'all'이라도 
        # 'quota' 모드(인원 수 우선순위)로 처리
        effective_search_mode = search_mode
        if user_limit is not None and user_limit > 0:
            effective_search_mode = "quota"

        # 'all' 모드이면서 'limit'가 *없을* 때만 3가지 모두 반환
        if effective_search_mode == "all":
            response = {
                "query": query_text,
                "classification": classification,
                "display_fields": display_fields,  # ✅ 추가
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
                # 'all' 모드의 기본값은 'weighted' 결과
                "final_panel_ids": search_results['results']['weighted']['panel_ids'][:100]
            }
        
        # 'limit'가 있거나('quota'), 'all'이 아닌 search_mode일 때
        else:
            # 단일 모드 결과 반환
            final_panel_ids = search_results['final_panel_ids']
            match_scores = search_results['match_scores']
            
            response = {
                "query": query_text,
                "classification": classification,
                "display_fields": display_fields,  # ✅ 추가
                "source_counts": {
                    "welcome_objective_count": len(search_results['panel_id1']),
                    "welcome_subjective_count": len(search_results['panel_id2']),
                    "qpoll_count": len(search_results['panel_id3'])
                },
                "results": {
                    # effective_search_mode는 "quota", "intersection" 등이 됨
                    effective_search_mode: {
                        "count": len(final_panel_ids),
                        # 응답 반환 시 100개로 제한 (limit이 100보다 크더라도)
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
        
        return response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")

# ====================================================================
# 2. 검색 + 분석 통합 API (NEW!)
# ====================================================================

@app.post("/api/search-and-analyze")
async def search_and_analyze(request: AnalysisRequest):
    """
    자연어 질의를 받아 검색 + 분석을 한 번에 수행합니다.
    
    search_panels()와 동일한 검색 로직 + 차트 데이터 추가
    
    프로세스:
    1. LLM 키워드 분류 (ranked_keywords 포함)
    2. 하이브리드 검색 수행 (search_panels와 동일)
    3. 검색 결과 반환 (search_panels와 동일)
    4. 차트 데이터 생성 및 추가 (신규)
    
    Returns:
        search_panels 응답 + charts 필드
    """
    query_text = request.query
    search_mode = request.search_mode
    
    valid_modes = ["all", "weighted", "union", "intersection"]
    if search_mode not in valid_modes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search_mode. Must be one of: {valid_modes}"
        )
    
    try:
        print(f"\n{'='*70}")
        print(f"🔍 검색+분석 요청: {query_text}")
        print(f"📊 검색 모드: {search_mode}")
        print(f"{'='*70}\n")
        
        # ============================================================
        # 1~4단계: search_panels()와 동일
        # ============================================================
        
        # 1단계: LLM 키워드 분류 (ranked_keywords 포함)
        print("📌 1단계: LLM 키워드 분류")
        classification = classify_query_keywords(query_text)
        
        # 1.5단계: 분류 결과에서 limit 값 추출
        user_limit = classification.get('limit')
        print(f"💡 API: 감지된 Limit 값: {user_limit}")
        
        # 2단계: 하이브리드 검색 수행 (limit 인자 전달)
        search_results = hybrid_search(
            classification, 
            search_mode=search_mode,
            limit=user_limit
        )
        
        # 3단계: 검색 로그 기록 (limit 우선 확인)
        if user_limit is not None and user_limit > 0:
            total_count = len(search_results['final_panel_ids'])
        elif search_mode == "all":
            total_count = search_results['results']['union']['count']
        else:
            total_count = len(search_results['final_panel_ids']) 
        
        log_search_query(query_text, total_count)
        
        # 4단계: 응답 구성 (search_panels와 동일)
        
        # 프론트엔드 표시용 필드 추출 (ranked_keywords 기반)
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
        
        # 사용자가 'limit'를 지정했다면 'quota' 모드로 처리
        effective_search_mode = search_mode
        if user_limit is not None and user_limit > 0:
            effective_search_mode = "quota"

        # 'all' 모드이면서 'limit'가 없을 때만 3가지 모두 반환
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
            # panel_id 리스트 (차트 생성용)
            panel_id_list = search_results['results']['weighted']['panel_ids']
        
        else:
            # 단일 모드 결과 반환
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
            # panel_id 리스트 (차트 생성용)
            panel_id_list = final_panel_ids
        
        # ============================================================
        # 5단계: 차트 데이터 생성 및 추가 (신규)
        # ============================================================
        
        print("\n📌 5단계: 차트 데이터 생성")
        analysis_result, status_code = analyze_search_results(
            query_text,
            classification,
            panel_id_list
        )
        
        if status_code == 200:
            # ✅ 차트 데이터를 response에 추가
            response['charts'] = analysis_result.get('charts', [])
            response['analysis_summary'] = analysis_result.get('main_summary', '')
            print(f"✅ 차트 {len(response['charts'])}개 생성 완료")
        else:
            # 분석 실패 시 빈 차트 반환
            response['charts'] = []
            response['analysis_summary'] = '차트 생성 실패'
            print(f"⚠️  차트 생성 실패")
        
        print(f"\n✅ 검색+분석 완료")
        
        return response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ 검색+분석 실패: {e}")
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
    ranked_keywords도 함께 반환됩니다.
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
# 4. 패널 상세 정보 조회 API
# ====================================================================

@app.get("/api/panels/{panel_id}")
async def get_panel_details(panel_id: int):
    """
    특정 panel_id의 패널 상세 정보를 조회합니다.
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="데이터베이스 연결 실패")
        
        cur = conn.cursor()
        
        # welcome_meta2 테이블에서 기본 정보 조회
        cur.execute("""
            SELECT panel_id, structured_data
            FROM welcome_meta2 
            WHERE panel_id = %s
        """, (panel_id,))
        
        result = cur.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"panel_id {panel_id}를 찾을 수 없습니다.")
        
        panel_id_value, structured_data = result
        
        # JSONB 데이터 평탄화
        panel_data = {"panel_id": panel_id_value}
        if isinstance(structured_data, dict):
            panel_data.update(structured_data)
        
        cur.close()
        return panel_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 실패: {str(e)}")
    finally:
        if conn:
            conn.close()

# ====================================================================
# 5. 헬스체크
# ====================================================================

@app.get("/")
def read_root():
    return {
        "service": "Multi-Table Hybrid Search & Analysis API",
        "version": "3.0",
        "status": "running",
        "features": [
            "LLM 기반 키워드 분류 (ranked_keywords 포함)",
            "하이브리드 검색 (PostgreSQL + Qdrant)",
            "자동 차트 데이터 생성 (LLM 없음, Python 분석)"
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
    """시스템 상태 확인"""
    try:
        conn = get_db_connection()
        db_status = "ok" if conn else "error"
        if conn:
            conn.close()
        
        return {
            "status": "healthy",
            "database": db_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }