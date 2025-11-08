import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from hybrid_logic import classify_query_keywords  # 키워드 분류 함수
from search_logic import hybrid_search  # 통합 검색 함수
from db_logic import log_search_query, get_db_connection

# FastAPI 애플리케이션 초기화
app = FastAPI(title="Multi-Table Hybrid Search API v2")

# ====================================================================
# 요청/응답 모델
# ====================================================================

class SearchQuery(BaseModel):
    query: str
    search_mode: str = "all"  # 기본값: 모든 모드 결과 반환

class SearchResponse(BaseModel):
    query: str
    classification: dict
    results: dict
    final_pids: list[int]
    summary: dict

# ====================================================================
# 1. 메인 검색 API
# ====================================================================

@app.post("/api/search", response_model=SearchResponse)
async def search_panels(search_query: SearchQuery):
    """
    자연어 질의를 받아 Welcome/QPoll 테이블에서 하이브리드 검색을 수행합니다.
    
    검색 모드:
    - all (기본): 교집합, 합집합, 가중치 모두 반환 ⭐추천
    - intersection: 교집합만 (모든 조건 만족)
    - union: 합집합만 (하나라도 조건 만족)
    - weighted: 가중치 기반만 (객관식 40%, 주관식 30%, QPoll 30%)
    
    프로세스:
    1. LLM이 질의를 Welcome(객관/주관)/QPoll 키워드로 분류
    2. Welcome 객관식 → PostgreSQL 검색 (pid1)
    3. Welcome 주관식 → Qdrant 임베딩 검색 (pid2)
    4. QPoll → Qdrant 임베딩 검색 (pid3)
    5. 3가지 방식으로 결과 통합 및 정렬
    """
    query_text = search_query.query
    search_mode = search_query.search_mode
    
    # 검색 모드 검증
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
        
        # 1단계: LLM 키워드 분류
        print("📌 1단계: LLM 키워드 분류")
        classification = classify_query_keywords(query_text)
        
        # 2단계: 하이브리드 검색 수행
        print("\n📌 2단계: 하이브리드 검색")
        search_results = hybrid_search(classification, search_mode=search_mode)
        
        # 3단계: 검색 로그 기록
        if search_mode == "all":
            total_count = search_results['results']['union']['count']
        else:
            total_count = len(search_results['final_result'])
        
        log_search_query(query_text, total_count)
        
        # 4단계: 응답 구성
        if search_mode == "all":
            # 모든 모드 결과 반환
            response = {
                "query": query_text,
                "classification": classification,
                "source_counts": {
                    "welcome_objective_count": len(search_results['pid1']),
                    "welcome_subjective_count": len(search_results['pid2']),
                    "qpoll_count": len(search_results['pid3'])
                },
                "results": {
                    "intersection": {
                        "count": search_results['results']['intersection']['count'],
                        "pids": search_results['results']['intersection']['pids'][:100],
                        "top_scores": {
                            str(pid): search_results['results']['intersection']['scores'].get(pid, 0)
                            for pid in search_results['results']['intersection']['pids'][:10]
                        }
                    },
                    "union": {
                        "count": search_results['results']['union']['count'],
                        "pids": search_results['results']['union']['pids'][:100],
                        "top_scores": {
                            str(pid): search_results['results']['union']['scores'].get(pid, 0)
                            for pid in search_results['results']['union']['pids'][:10]
                        }
                    },
                    "weighted": {
                        "count": search_results['results']['weighted']['count'],
                        "pids": search_results['results']['weighted']['pids'][:100],
                        "weights": search_results['results']['weighted']['weights'],
                        "top_scores": {
                            str(pid): search_results['results']['weighted']['scores'].get(pid, 0)
                            for pid in search_results['results']['weighted']['pids'][:10]
                        }
                    }
                },
                "summary": {
                    "search_mode": search_mode,
                    "search_strategy": {
                        "welcome_objective": bool(classification.get('welcome_keywords', {}).get('objective')),
                        "welcome_subjective": bool(classification.get('welcome_keywords', {}).get('subjective')),
                        "qpoll": bool(classification.get('qpoll_keywords', {}).get('keywords'))
                    }
                },
                "final_pids": search_results['results']['weighted']['pids'][:100]
            }
        else:
            # 단일 모드 결과 반환
            final_pids = search_results['final_result']
            match_scores = search_results['match_scores']
            
            response = {
                "query": query_text,
                "classification": classification,
                "source_counts": {
                    "welcome_objective_count": len(search_results['pid1']),
                    "welcome_subjective_count": len(search_results['pid2']),
                    "qpoll_count": len(search_results['pid3'])
                },
                "results": {
                    search_mode: {
                        "count": len(final_pids),
                        "pids": final_pids[:100],
                        "top_scores": {
                            str(pid): match_scores.get(pid, 0)
                            for pid in final_pids[:10]
                        }
                    }
                },
                "summary": {
                    "total_candidates": len(final_pids),
                    "search_mode": search_mode,
                    "search_strategy": {
                        "welcome_objective": bool(classification.get('welcome_keywords', {}).get('objective')),
                        "welcome_subjective": bool(classification.get('welcome_keywords', {}).get('subjective')),
                        "qpoll": bool(classification.get('qpoll_keywords', {}).get('keywords'))
                    }
                },
                "final_pids": final_pids[:100]
            }
        
        print(f"\n✅ 검색 완료")
        
        return response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")

# ====================================================================
# 2. 디버깅 API - 키워드 분류만 테스트
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
# 3. 패널 상세 정보 조회 API
# ====================================================================

@app.get("/api/panels/{pid}")
async def get_panel_details(pid: int):
    """
    특정 PID의 패널 상세 정보를 조회합니다.
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="데이터베이스 연결 실패")
        
        cur = conn.cursor()
        
        # Welcome 테이블에서 기본 정보 조회
        cur.execute("""
            SELECT pid, gender, birth_year, region, marital_status, 
                   income_personal_monthly, job_title_raw
            FROM welcome 
            WHERE pid = %s
        """, (pid,))
        
        result = cur.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"PID {pid}를 찾을 수 없습니다.")
        
        panel_data = {
            "pid": result[0],
            "gender": result[1],
            "birth_year": result[2],
            "region": result[3],
            "marital_status": result[4],
            "income_personal_monthly": result[5],
            "job_title": result[6]
        }
        
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
# 4. 헬스체크
# ====================================================================

@app.get("/")
def read_root():
    return {
        "service": "Multi-Table Hybrid Search API",
        "version": "2.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    """시스템 상태 확인"""
    try:
        # DB 연결 테스트
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