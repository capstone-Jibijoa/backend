import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from hybrid_logic import classify_query_keywords
from search_logic import hybrid_search
from db_logic import log_search_query, get_db_connection

app = FastAPI(title="Multi-Table Hybrid Search API v2")


class SearchQuery(BaseModel):
    query: str
    search_mode: str = "all"


class SearchResponse(BaseModel):
    query: str
    classification: dict
    results: dict
    final_panel_ids: list[str]
    summary: dict


@app.post("/api/search", response_model=SearchResponse)
async def search_panels(search_query: SearchQuery):
    """
    자연어 질의를 받아 Welcome/QPoll 하이브리드 검색 수행
    
    검색 모드:
    - all (기본): 교집합, 합집합, 가중치 모두 반환
    - intersection: 교집합만 (모든 조건 만족)
    - union: 합집합만 (하나라도 조건 만족)
    - weighted: 가중치 기반 (객관 40%, 주관 30%, QPoll 30%)
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
        # 1단계: LLM 키워드 분류
        classification = classify_query_keywords(query_text)
        
        # 2단계: 하이브리드 검색 수행
        search_results = hybrid_search(classification, search_mode=search_mode)
        
        # 3단계: 검색 로그 기록
        if search_mode == "all":
            total_count = search_results['results']['union']['count']
        else:
            total_count = len(search_results['final_panel_ids'])  
        
        log_search_query(query_text, total_count)
        
        # 4단계: 응답 구성
        if search_mode == "all":
            response = {
                "query": query_text,
                "classification": classification,
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
                    }
                },
                "final_panel_ids": search_results['results']['weighted']['panel_ids'][:100]
            }
        else:
            final_panel_ids = search_results['final_panel_ids']
            match_scores = search_results['match_scores']
            
            response = {
                "query": query_text,
                "classification": classification,
                "source_counts": {
                    "welcome_objective_count": len(search_results['panel_id1']),
                    "welcome_subjective_count": len(search_results['panel_id2']),
                    "qpoll_count": len(search_results['panel_id3'])
                },
                "results": {
                    search_mode: {
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
                    "search_mode": search_mode,
                    "search_strategy": {
                        "welcome_objective": bool(classification.get('welcome_keywords', {}).get('objective')),
                        "welcome_subjective": bool(classification.get('welcome_keywords', {}).get('subjective')),
                        "qpoll": bool(classification.get('qpoll_keywords', {}).get('keywords'))
                    }
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


@app.post("/api/debug/classify")
async def debug_classify(search_query: SearchQuery):
    """질의를 키워드로 분류만 하고 결과 반환 (검색 X)"""
    try:
        classification = classify_query_keywords(search_query.query)
        return {
            "query": search_query.query,
            "classification": classification
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분류 실패: {str(e)}")


@app.get("/api/panels/{panel_id}")
async def get_panel_details(panel_id: str):
    """특정 panel_id의 패널 상세 정보 조회"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="데이터베이스 연결 실패")
        
        cur = conn.cursor()
        
        cur.execute("""
            SELECT panel_id, gender, birth_year, region, marital_status, 
                   income_personal_monthly, job_title_raw
            FROM welcome 
            WHERE panel_id = %s
        """, (panel_id,))
        
        result = cur.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"panel_id {panel_id}를 찾을 수 없습니다.")
        
        panel_data = {
            "panel_id": result[0],
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