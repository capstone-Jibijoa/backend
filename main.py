import os
import json
import time
import asyncio
loop = asyncio.get_running_loop()
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Tuple, Dict, List
from fastapi.middleware.cors import CORSMiddleware

from llm import classify_query_keywords
from search_helpers import initialize_embeddings
from search import hybrid_search_parallel as hybrid_search
from analysis import analyze_search_results_optimized as analyze_search_results
from db import (
    log_search_query,
    get_db_connection_context,
    init_db,
    cleanup_db,
    get_qdrant_client
)
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# 루트 로거 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Uvicorn, FastAPI 등 라이브러리 로거 레벨 설정 (필요시)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)
# --- 로깅 설정 ---

app = FastAPI(title="Multi-Table Hybrid Search API v3 (Refactored)")

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

def preload_models():
    """애플리케이션 시작 시 모든 AI 모델을 미리 로드합니다."""
    logging.info("="*70)
    logging.info("🔄 모든 AI 모델을 미리 로드합니다...")
    initialize_embeddings()
    classify_query_keywords("모델 로딩 테스트")
    try:
        classify_query_keywords("모델 로딩 테스트")
        logging.info("✅ Claude (LLM) 모델 연결 확인 완료.")
    except Exception as e:
        logging.warning(f"⚠️  Claude (LLM) 모델 연결 테스트 실패: {e}")
        logging.warning("   LLM 기능이 작동하지 않을 수 있지만, 서버는 계속 시작합니다.")
    logging.info("✅ 모든 AI 모델 로드 완료")
    logging.info("="*70)

@app.on_event("startup")
async def startup_event():
    logging.info("🚀 FastAPI 시작...")
    init_db()
    preload_models()

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("🧹 FastAPI 종료... Connection Pool 정리")
    cleanup_db()


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
    search_mode: str = "weighted"

class AnalysisResponse(BaseModel):
    query: str
    total_count: int
    main_summary: str
    charts: list[dict]

def _prepare_display_fields(classification: Dict) -> List[Dict]:
    """
    [리팩토링] ranked_keywords로부터 display_fields를 생성합니다.
    """
    display_fields_raw = []
    for kw_info in classification.get('ranked_keywords', [])[:5]:
        field = kw_info.get('field', '')
        description = kw_info.get('description', '')
        priority = kw_info.get('priority', 999)
        
        fields = [f.strip() for f in field.split(',')]
        
        for f in fields:
            if f:
                display_fields_raw.append({
                    'field': f,
                    'label': description,
                    'priority': priority
                })

    unique_display_fields_map = {}
    for item in display_fields_raw:
        if item['field'] not in unique_display_fields_map:
            unique_display_fields_map[item['field']] = item
    
    return list(unique_display_fields_map.values())


def _build_pro_mode_response(
    query_text: str,
    classification: Dict,
    search_results: Dict,
    display_fields: List[Dict],
    effective_search_mode: str
) -> Tuple[Dict, List[str]]:
    """
    Pro 모드의 복잡한 응답 본문을 생성합니다.
    """
    source_counts = {
        "welcome_objective_count": len(search_results['panel_id1']),
        "welcome_subjective_count": len(search_results['panel_id2']),
        "qpoll_count": len(search_results['panel_id3'])
    }
    
    summary = {
        "search_mode": effective_search_mode,
        "search_strategy": {
            "welcome_objective": bool(classification.get('welcome_keywords', {}).get('objective')),
            "welcome_subjective": bool(classification.get('welcome_keywords', {}).get('subjective')),
            "qpoll": bool(classification.get('qpoll_keywords', {}).get('keywords'))
        },
        "ranked_keywords": classification.get('ranked_keywords', [])
    }

    response = {
        "query": query_text,
        "classification": classification,
        "display_fields": display_fields,
        "source_counts": source_counts,
        "summary": summary,
    }

    if effective_search_mode == "all":
        # 'all' 모드는 모든 검색 결과를 포함
        response["results"] = {}
        panel_id_list = []
        for mode_name, mode_results in search_results['results'].items():
            response["results"][mode_name] = {
                "count": mode_results['count'],
                "panel_ids": mode_results['panel_ids'][:100],
                "top_scores": {
                    str(pid): mode_results['scores'].get(pid, 0)
                    for pid in mode_results['panel_ids'][:10]
                }
            }
            if 'weights' in mode_results:
                response["results"][mode_name]['weights'] = mode_results['weights']
        
        panel_id_list = search_results['results']['weighted']['panel_ids']
        response["final_panel_ids"] = panel_id_list[:100]

    else: # 'quota', 'weighted', 'union', 'intersection'
        final_panel_ids = search_results['final_panel_ids']
        match_scores = search_results['match_scores']
        
        response["results"] = {
            effective_search_mode: {
                "count": len(final_panel_ids),
                "panel_ids": final_panel_ids[:100],
                "top_scores": {str(pid): match_scores.get(pid, 0) for pid in final_panel_ids[:10]}
            }
        }
        response["final_panel_ids"] = final_panel_ids[:100]
        panel_id_list = final_panel_ids

    return response, panel_id_list

async def _perform_common_search(query_text: str, search_mode: str, mode: str) -> Tuple[Dict, List[str], Dict]:
    """
    /search와 /search-and-analyze가 공유하는 핵심 로직
    (LLM 분류, 병렬 검색, 로그 기록, 결과 포맷팅)
    """
    logging.info(f"🔍 공통 검색 시작: {query_text} (모드: {search_mode}, 실행: {mode})")
    
    # 1. LLM 키워드 분류
    classification = classify_query_keywords(query_text)
    logging.info(f"🤖 LLM 분류 결과: {classification}")
    user_limit = classification.get('limit')
    effective_search_mode = "quota" if user_limit and user_limit > 0 else search_mode
    logging.info(f"💡 API: 감지된 Limit 값: {user_limit}")

    search_results = hybrid_search(
        classification,
        search_mode,
        user_limit
    )
    
    # 3. 검색 로그 기록
    total_count = len(search_results['final_panel_ids'])
    log_search_query(query_text, total_count)
    
    # 4. 응답 구성
    display_fields = _prepare_display_fields(classification)
    panel_ids_for_analysis = search_results['final_panel_ids']
    
    # Lite 모드 응답 간소화
    if mode == "lite":
        lite_response = {
            "query": query_text,
            "classification": {
                "ranked_keywords": classification.get('ranked_keywords', []),
            },
            "display_fields": display_fields,
            "total_count": total_count,
            "final_panel_ids": panel_ids_for_analysis[:500], # 테이블 조회를 위해 최대 500개
            "effective_search_mode": effective_search_mode
        }
        logging.info("✅ 공통 검색 완료 (Lite 모드 간소화)")
        return lite_response, panel_ids_for_analysis, classification

    # Pro 모드 (기존 로직 유지)
    response, panel_id_list = _build_pro_mode_response(
        query_text,
        classification,
        search_results,
        display_fields,
        effective_search_mode
    )
    
    # 분석을 위해 최대 5000개 ID 전달
    panel_ids_for_analysis = panel_id_list[:5000]
    
    logging.info("✅ 공통 검색 완료 (Pro 모드 전체 데이터)")
    return response, panel_ids_for_analysis, classification


async def _get_ordered_table_data(
    ids_to_fetch: List[str], 
    fields_to_fetch: List[str] = None
) -> List[dict]:
    """
    DB에서 패널 데이터를 조회하되, 입력된 id 리스트 순서를 보장하여 반환합니다.
    - fields_to_fetch가 None이면 structured_data 전체를,
    - fields_to_fetch가 리스트면 해당 필드만 선택적으로 조회합니다.
    """
    if not ids_to_fetch:
        return []

    table_data = []
    try:
        with get_db_connection_context() as conn:
            if not conn:
                raise Exception("DB 연결 실패")
            
            cur = conn.cursor()
            
            # 1. SQL 쿼리 준비 (필드 선택 부분 동적 구성)
            if fields_to_fetch:
                field_selects = ", ".join([
                    f"structured_data->>'{field}' as \"{field}\""
                    for field in fields_to_fetch
                ])
                sql_query = f"SELECT panel_id, {field_selects} FROM welcome_meta2 WHERE panel_id = ANY(%s::text[])"
            else:
                sql_query = "SELECT panel_id, structured_data FROM welcome_meta2 WHERE panel_id = ANY(%s::text[])"

            # 2. DB에서 데이터 조회
            cur.execute(sql_query, (ids_to_fetch,))
            results = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

            # 3. 순서 재정렬을 위한 맵 생성
            fetched_data_map = {row[0]: row for row in results}

            # 4. 입력된 ID 순서대로 결과 재구성
            for pid in ids_to_fetch:
                if pid in fetched_data_map:
                    row_data = fetched_data_map[pid]
                    if fields_to_fetch:
                        # Lite 모드: 특정 필드만 포함된 딕셔너리 생성
                        data = {columns[i]: row_data[i] for i in range(len(columns))}
                    else:
                        # Pro 모드: structured_data 전체를 포함
                        data = row_data[1] or {} # structured_data가 null일 경우 빈 dict
                        data['panel_id'] = pid
                    table_data.append(data)
            
            cur.close()
            
    except Exception as db_e:
        logging.error(f"Table Data 조회 실패: {db_e}", exc_info=True)
    
    return table_data


@app.post("/api/search")
async def search_panels(search_query: SearchQuery):
    """
    🚀 Lite 모드: 빠른 검색 (차트 분석 없이 테이블 데이터만 반환)
    """
    logging.info(f"🚀 [Lite 모드] 빠른 검색 시작: {search_query.query}")
    
    valid_modes = ["all", "weighted", "union", "intersection"]
    if search_query.search_mode not in valid_modes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search_mode. Must be one of: {valid_modes}"
        )
    
    try:
        start_time = time.time()
        
        # 1. 공통 검색 함수 호출 (mode="lite")
        lite_response, _, _ = await _perform_common_search(
            search_query.query, 
            search_query.search_mode,
            mode="lite"
        )
        
        search_time = time.time() - start_time
        logging.info(f"⏱️  [Lite 모드] 검색 완료: {search_time:.2f}초")
        
        # 2. 테이블 데이터 조회 (리팩토링된 함수 사용)
        ids_to_fetch = lite_response['final_panel_ids']
        fields_to_fetch = [item['field'] for item in lite_response.get('display_fields', [])]
        
        db_start = time.time()
        logging.info(f"📊 [Lite 모드] 테이블 데이터 조회 시작 (최대 {len(ids_to_fetch)}개)")
        
        table_data = await _get_ordered_table_data(ids_to_fetch, fields_to_fetch)
        
        db_time = time.time() - db_start
        logging.info(f"✅ [Lite 모드] 테이블 데이터 {len(table_data)}개 조회 완료: {db_time:.2f}초")
        
        # 3. Lite 모드 최종 응답 구성
        lite_response['tableData'] = table_data
        lite_response['mode'] = "lite" 
        del lite_response['final_panel_ids'] # ID 목록은 응답에서 제거
        
        total_time = time.time() - start_time
        logging.info(f"✅ [Lite 모드] 전체 완료: {total_time:.2f}초 - 총 {lite_response['total_count']}개 결과 중 {len(table_data)}개 테이블 데이터 반환")
        
        return lite_response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"[Lite 모드] /api/search 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")


@app.post("/api/search-and-analyze")
async def search_and_analyze(request: AnalysisRequest):
    """
    📊 Pro 모드: 검색 + 인사이트 분석 (차트 + 테이블 데이터 반환)
    """
    logging.info(f"📊 [Pro 모드] 검색 + 분석 시작: {request.query}")
    
    valid_modes = ["all", "weighted", "union", "intersection"]
    if request.search_mode not in valid_modes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search_mode. Must be one of: {valid_modes}"
        )
    
    try:
        # 1. 공통 검색 함수 호출 (mode="pro")
        response, panel_id_list, classification = await _perform_common_search(
            request.query, 
            request.search_mode,
            mode="pro"
        )
        
        # 2. 차트 데이터 생성
        logging.info("📊 [Pro 모드] 차트 데이터 생성 시작")
        analysis_result, status_code = analyze_search_results(
            request.query,
            classification,
            panel_id_list
        )
        
        if status_code == 200:
            response['charts'] = analysis_result.get('charts', [])
            response['analysis_summary'] = analysis_result.get('main_summary', '')
            logging.info(f"✅ [Pro 모드] 차트 {len(response['charts'])}개 생성 완료")
        else:
            response['charts'] = []
            response['analysis_summary'] = '차트 생성 실패'
            logging.warning("[Pro 모드] 차트 생성 실패")
        
        # 3. Table Data 생성 (리팩토링된 함수 사용)
        logging.info(f"📊 [Pro 모드] Table Data 생성 시작 (패널 {len(panel_id_list)}개 대상)")
        ids_to_fetch = response['final_panel_ids'] # Pro 모드는 100개만
        
        # Pro 모드는 fields_to_fetch=None으로 전달 (전체 structured_data 조회)
        table_data = await _get_ordered_table_data(ids_to_fetch, fields_to_fetch=None)
        
        # 4. 최종 응답 구성
        response['tableData'] = table_data
        response['mode'] = 'pro'
        
        logging.info(f"✅ [Pro 모드] 차트 {len(response['charts'])}개, 테이블 데이터 {len(table_data)}개 생성 완료")
        
        return response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"[Pro 모드] /api/search-and-analyze 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")


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


async def _get_welcome_data(panel_id: str) -> Dict:
    """[리팩토링] PostgreSQL에서 Welcome 데이터를 비동기적으로 조회합니다."""
    loop = asyncio.get_running_loop()
    
    def db_call():
        with get_db_connection_context() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="DB 연결 실패")
            
            cur = conn.cursor()
            cur.execute(
                "SELECT panel_id, structured_data FROM welcome_meta2 WHERE panel_id = %s",
                (panel_id,)
            )
            result = cur.fetchone()
            cur.close()
            
            if not result:
                raise HTTPException(status_code=404, detail=f"panel_id {panel_id}를 찾을 수 없습니다.")
            
            panel_id_value, structured_data = result
            panel_data = {"panel_id": panel_id_value}
            if isinstance(structured_data, dict):
                panel_data.update(structured_data)
            return panel_data

    return await loop.run_in_executor(None, db_call)


async def _get_qpoll_data(panel_id: str) -> Dict:
    """[리팩토링] Qdrant에서 QPoll 데이터를 비동기적으로 조회합니다."""
    loop = asyncio.get_running_loop()
    qpoll_data = {"qpoll_응답_개수": 0}

    def qdrant_call():
        try:
            qdrant_client = get_qdrant_client()
            if not qdrant_client:
                logging.warning("⚠️  Qdrant 클라이언트 없음")
                return qpoll_data

            logging.info(f"🔍 QPoll 데이터 조회 시작 (panel_id: {panel_id})")
            qpoll_results, _ = qdrant_client.scroll(
                collection_name="qpoll_vectors_v2",
                scroll_filter=Filter(must=[FieldCondition(key="panel_id", match=MatchValue(value=panel_id))]),
                limit=100, with_payload=True, with_vectors=False
            )

            if qpoll_results:
                logging.info(f"✅ QPoll 응답 {len(qpoll_results)}개 발견")
                for idx, point in enumerate(qpoll_results, 1):
                    if point.payload:
                        qpoll_data[f"qpoll_{idx:03d}_질문"] = point.payload.get("question", "")
                        qpoll_data[f"qpoll_{idx:03d}_응답"] = point.payload.get("sentence", "")
                qpoll_data["qpoll_응답_개수"] = len(qpoll_results)
            else:
                logging.warning("⚠️  QPoll 응답 없음")
            
            return qpoll_data

        except Exception as qpoll_error:
            logging.error(f"❌ QPoll 조회 실패 (panel_id: {panel_id}): {qpoll_error}", exc_info=True)
            qpoll_data["qpoll_조회_오류"] = str(qpoll_error)
            return qpoll_data

    return await loop.run_in_executor(None, qdrant_call)


@app.get("/api/panels/{panel_id}")
async def get_panel_details(panel_id: str):
    """
    [개선] 특정 panel_id의 패널 상세 정보를 병렬로 조회합니다.
    - Welcome(PostgreSQL)과 QPoll(Qdrant) 데이터를 동시에 조회하여 성능 개선
    """
    try:
        logging.info(f"⚡️ 패널 상세 정보 병렬 조회 시작 (panel_id: {panel_id})")
        
        # Welcome 데이터와 QPoll 데이터를 동시에 조회
        results = await asyncio.gather(
            _get_welcome_data(panel_id),
            _get_qpoll_data(panel_id),
            return_exceptions=True  # 한쪽에서 에러가 나도 다른 쪽은 계속 진행
        )

        # 결과 취합
        panel_data, qpoll_data = {}, {}
        for result in results:
            if isinstance(result, HTTPException):
                raise result # 404 Not Found 등은 즉시 반환
            elif isinstance(result, Exception):
                raise HTTPException(status_code=500, detail=f"조회 중 내부 오류 발생: {result}")
            
            if "qpoll_응답_개수" in result:
                qpoll_data = result
            else:
                panel_data = result

        panel_data.update(qpoll_data)
        return panel_data
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"패널 상세 조회 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"조회 실패: {str(e)}")


@app.get("/")
def read_root():
    return {
        "service": "Multi-Table Hybrid Search & Analysis API",
        "version": "3.0 (Refactored)",
        "status": "running",
        "optimizations_applied": [
            "DB Connection Pool (psycopg2-pool)",
            "Parallel Search (ThreadPoolExecutor)",
            "DB Aggregate Queries (analysis_logic)"
        ],
        "endpoints": {
            "search": "/api/search (Lite)",
            "search_and_analyze": "/api/search-and-analyze (Pro)",
            "classify": "/api/debug/classify",
            "panel_detail": "/api/panels/{panel_id}",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    """시스템 상태 확인"""
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