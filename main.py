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
    final_panel_ids: list[int]
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
    2. Welcome 객관식 → PostgreSQL 검색 (pid1)
    3. Welcome 주관식 → Qdrant 임베딩 검색 (pid2)
    4. QPoll → Qdrant 임베딩 검색 (pid3)
    5. 3가지 방식으로 결과 통합 및 정렬
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
                    },
                    "ranked_keywords": classification.get('ranked_keywords', [])
                },
                "final_panel_ids": search_results['results']['weighted']['pids'][:100]
            }
        else:
            # 단일 모드 결과 반환
            final_panel_ids = search_results['final_panel_ids']
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
                        "count": len(final_panel_ids),
                        "pids": final_panel_ids[:100],
                        "top_scores": {
                            str(pid): match_scores.get(pid, 0)
                            for pid in final_panel_ids[:10]
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

@app.post("/api/search-and-analyze", response_model=AnalysisResponse)
async def search_and_analyze(request: AnalysisRequest):
    """
    자연어 질의를 받아 검색 + 분석을 한 번에 수행합니다.
    
    프로세스:
    1. 키워드 분류 (LLM) - ranked_keywords 추출
    2. 하이브리드 검색 수행
    3. 검색 결과 분석 및 차트 데이터 생성 (LLM 없음, Python만 사용)
       - ranked_keywords 상위 2개로 차트 생성 (전체 DB 기반)
       - 높은 비율(70%↑) 필드 추가 차트 생성 (검색 결과 기반)
       - 최대 5개 차트 반환
    
    Args:
        query: 자연어 질의
        search_mode: weighted(권장) / union / intersection
    
    Returns:
        {
            "query": "...",
            "total_count": 5210,
            "main_summary": "...",
            "charts": [...]
        }
    """
    try:
        query_text = request.query
        search_mode = request.search_mode
        
        print(f"\n{'='*70}")
        print(f"🔍 검색+분석 요청: {query_text}")
        print(f"📊 검색 모드: {search_mode}")
        print(f"{'='*70}\n")
        
        # 1단계: 키워드 분류 (ranked_keywords 포함)
        print("📌 1단계: LLM 키워드 분류")
        classification = classify_query_keywords(query_text)
        
        # 2단계: 하이브리드 검색
        print("\n📌 2단계: 하이브리드 검색")
        search_results = hybrid_search(classification, search_mode=search_mode)
        
        # PID 리스트 추출
        if search_mode == "all":
            pid_list = search_results['results']['weighted']['pids']
        else:
            pid_list = search_results['final_panel_ids']
        
        # 로그 기록
        log_search_query(query_text, len(pid_list))
        
        # 3단계: 분석 수행 (LLM 없음, ranked_keywords 사용)
        print("\n📌 3단계: 결과 분석")
        analysis_result, status_code = analyze_search_results(
            query_text,
            classification,  # ranked_keywords 포함
            pid_list
        )
        
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=analysis_result.get('error', '분석 실패'))
        
        print(f"\n✅ 검색+분석 완료")
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 검색+분석 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

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
        
        # welcome_meta2 테이블에서 기본 정보 조회
        cur.execute("""
            SELECT pid, structured_data
            FROM welcome_meta2 
            WHERE pid = %s
        """, (pid,))
        
        result = cur.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"PID {pid}를 찾을 수 없습니다.")
        
        pid_value, structured_data = result
        
        # JSONB 데이터 평탄화
        panel_data = {"pid": pid_value}
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
            "panel_detail": "/api/panels/{pid}",
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
    
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔬 `/api/search-and-analyze` 핵심 로직 직접 호출 테스트 시작")
    print("="*80)

    # 1. 테스트 쿼리 및 모드 설정
    test_query = "30대 직장인이 출퇴근 시 사용하는 대중교통 관련 문항의 트렌드"
    test_mode = "weighted"
    
    try:
        # 1단계: 키워드 분류 (hybrid_logic)
        classification = classify_query_keywords(test_query)
        print(f"✅ 1단계 분류 완료. 랭크 키워드: {classification.get('ranked_keywords', [])[:3]}")

        # 2단계: 하이브리드 검색 (search_logic)
        search_results = hybrid_search(classification, search_mode=test_mode)
        pid_list = search_results['final_panel_ids']
        print(f"✅ 2단계 검색 완료. 결과 PID {len(pid_list)}개 확보.")

        # 3단계: 로그 기록
        log_search_query(test_query, len(pid_list))
        
        # 4단계: 분석 수행 (analysis_logic)
        analysis_result, status_code = analyze_search_results(
            test_query,
            classification,
            pid_list
        )

        if status_code == 200:
            print("\n✅ 4단계 분석 성공. 통합 테스트 최종 성공.")
            print(f"   - 총 결과 수: {analysis_result.get('total_count')}개")
            print(f"   - 주요 요약 (부분): {analysis_result.get('main_summary', 'N/A')[:40]}...")
            print(f"   - 차트 개수: {len(analysis_result.get('charts', []))}개")
            for i, chart in enumerate(analysis_result.get('charts', []), 1):
                print(f"\n[차트 {i}]")
                print(f"  제목: {chart.get('topic')}")
                print(f"  설명: {chart.get('description')}")
                print(f"  비율: {chart.get('ratio')}")
                print(f"  차트 데이터:")
                for chart_item in chart.get('chart_data', []):
                    print(f"    라벨: {chart_item.get('label')}")
                    print(f"    값:")
                    for key, value in chart_item.get('values', {}).items():
                        print(f"      - {key}: {value}%")
            
            print("\n" + "="*70)
            print("📋 전체 결과 JSON")
            print("="*70)
            print(json.dumps(analysis_result, indent=2, ensure_ascii=False))
        else:
            print(f"\n❌ 4단계 분석 실패. 상태 코드: {status_code}, 오류: {analysis_result.get('error', 'N/A')}")

    except Exception as e:
        print(f"\n🛑 통합 테스트 중 오류 발생: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()