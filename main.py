import os
import json
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Tuple, Dict, List
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

from llm import classify_query_keywords
from search import initialize_embeddings
from search import hybrid_search as hybrid_search
import asyncio
from mapping_rules import QPOLL_FIELD_TO_TEXT, VECTOR_CATEGORY_TO_FIELD
from insights import (
    analyze_search_results_optimized as analyze_search_results,
    get_field_mapping
)
from db import (
    log_search_query,
    get_db_connection_context,
    init_db,
    cleanup_db,
    get_qdrant_client
)
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from utils import FIELD_NAME_MAP

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
    # 캐시 초기화 (In-memory backend 사용)
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    logging.info("✅ 캐시 시스템 초기화 완료.")
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
    [v2] classification 결과로부터 테이블 헤더(display_fields)를 생성합니다.
    - `objective_keywords`와 `mandatory_keywords`를 우선적으로 사용합니다.
    - `get_field_mapping`을 호출하여 각 키워드에 맞는 필드와 설명을 찾습니다.
    """
    
    objective_kws, must_have_kws, preference_kws = classification.get('objective_keywords', []), classification.get('must_have_keywords', []), classification.get('preference_keywords', [])

    qpoll_keywords = []
    other_must_have = []
    other_preference = []
    
    for kw in must_have_kws:
        mapping = get_field_mapping(kw)
        if mapping and mapping.get('type') == 'qpoll':
            qpoll_keywords.append(kw)
        else:
            other_must_have.append(kw)
    
    for kw in preference_kws:
        mapping = get_field_mapping(kw)
        if mapping and mapping.get('type') == 'qpoll':
            if kw not in qpoll_keywords:
                qpoll_keywords.append(kw)
        else:
            other_preference.append(kw)
    
    header_keywords = qpoll_keywords + other_must_have + other_preference + objective_kws
    
    if not header_keywords:
        logging.warning("⚠️ _prepare_display_fields: 분석할 키워드가 없습니다. 빈 헤더 반환.")
        return []
        
    objective_field_counts = {}
    for kw in objective_kws:
        mapping = get_field_mapping(kw)
        field = mapping.get('field')
        if field and field != 'unknown':
            if field not in objective_field_counts:
                objective_field_counts[field] = 0
            objective_field_counts[field] += 1
    
    ALWAYS_INCLUDE_FIELDS = {'gender', 'birth_year'}

    single_value_fields_to_exclude = {field for field, count in objective_field_counts.items() 
                                      if count == 1 and field not in ALWAYS_INCLUDE_FIELDS}

    unique_fields = {}
    priority_counter = 0

    for keyword in header_keywords:
        if len(unique_fields) >= 5:
            break

        mapping = get_field_mapping(keyword)
        field = mapping.get('field')
        
        if keyword in objective_kws:
            if field in single_value_fields_to_exclude:
                continue

        if field and field != 'unknown' and field not in unique_fields:
            label = FIELD_NAME_MAP.get(field, mapping.get('description', field))
            unique_fields[field] = {
                'field': field,
                'label': label,
                'priority': 0 if mapping.get('type') == 'qpoll' else priority_counter
            }
            priority_counter += 1

    if len(unique_fields) < 5:
        
        all_subjective_kws = must_have_kws + preference_kws
        recommended_fields = []
        
        # 키워드와 연관된 필드 그룹 매핑
        TOPIC_TO_RELATED_FIELDS = {
            '여행': ['income_personal_monthly', 'family_size'],
            '자동차': ['car_model_raw', 'car_manufacturer_raw'],
            '직업': ['education_level', 'income_personal_monthly'],
            '가족': ['marital_status', 'children_count'],
            '소득': ['job_duty_raw', 'education_level']
        }
        
        for topic, related_fields in TOPIC_TO_RELATED_FIELDS.items():
            if any(topic in kw for kw in all_subjective_kws):
                recommended_fields.extend(related_fields)
        
        if not recommended_fields:
            recommended_fields = ['gender', 'birth_year', 'region_major']

        fields_to_augment = list(dict.fromkeys(recommended_fields))
        
        for field_key in fields_to_augment:
            if len(unique_fields) >= 5: break
            if field_key not in unique_fields:
                korean_name = FIELD_NAME_MAP.get(field_key, field_key)
                unique_fields[field_key] = {
                    'field': field_key, 'label': korean_name, 'priority': 900 + len(unique_fields)
                }

    found_categories = classification.get('found_categories', [])
    if found_categories and len(unique_fields) < 5:
        for category in found_categories:
            if len(unique_fields) >= 5:
                break
            
            fields_to_add = VECTOR_CATEGORY_TO_FIELD.get(category, [])
            for field in fields_to_add:
                if len(unique_fields) >= 5:
                    break
                if field not in unique_fields:
                    label = FIELD_NAME_MAP.get(field, field)
                    unique_fields[field] = {
                        'field': field, 
                        'label': label, 
                        'priority': 950 + len(unique_fields)
                    }

    final_result = sorted(list(unique_fields.values()), key=lambda x: x['priority'])
    return final_result
 
 
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
        "stage1_objective": search_results.get("stage_details", {}).get("stage1_objective", 0),
        "stage2_must_have": search_results.get("stage_details", {}).get("stage2_must_have", 0),
        "stage3_preference": search_results.get("stage_details", {}).get("stage3_preference", 0),
        "stage4_negative": search_results.get("stage_details", {}).get("stage4_negative", 0),
    }
    
    summary = {
        "search_mode": effective_search_mode,
        "ranked_keywords": classification.get('ranked_keywords_raw', [])
    }

    response = {
        "query": query_text,
        "classification": classification,
        "display_fields": display_fields,
        "source_counts": source_counts,
        "summary": summary,
    }

    final_panel_ids = search_results.get('final_panel_ids', [])
    total_count = search_results.get('total_count', 0)
    
    panel_id_list = final_panel_ids[:100]
    response["final_panel_ids"] = panel_id_list

    if effective_search_mode:
        response["results"] = {
            effective_search_mode: {
                "count": total_count,
                "panel_ids": panel_id_list,
            }
        }

    return response, panel_id_list

async def _perform_common_search(query_text: str, search_mode: str, mode: str) -> Tuple[Dict, List[str], Dict]:
    """
    /search와 /search-and-analyze가 공유하는 핵심 로직
    (LLM 분류, 병렬 검색, 로그 기록, 결과 포맷팅)
    """
    logging.info(f"🔍 공통 검색 시작: {query_text} (모드: {search_mode}, 실행: {mode})")
    classification = classify_query_keywords(query_text)
    user_limit = classification.get('limit')
    effective_search_mode = "quota" if user_limit and user_limit > 0 else search_mode
    
    search_results = hybrid_search(
        query=query_text,
        limit=user_limit
    )
    
    panel_id_list = search_results.get('final_panel_ids', [])
    total_count = len(panel_id_list)
    log_search_query(query_text, total_count)
    
    classification['ranked_keywords_raw'] = classification.get('objective_keywords', []) + classification.get('must_have_keywords', [])
    display_fields = _prepare_display_fields(classification)
    
    if mode == "lite":
        lite_response = {
            "query": query_text,
            "classification": classification,
            "total_count": total_count,
            "final_panel_ids": panel_id_list[:500],
            "effective_search_mode": effective_search_mode,
            "display_fields": display_fields
        }

        logging.info("✅ 공통 검색 완료 (Lite 모드 간소화)")
        return lite_response, panel_id_list, classification

    response, panel_id_list = _build_pro_mode_response(
        query_text,
        classification,
        search_results,
        display_fields,
        effective_search_mode,
    )
    
    logging.info("✅ 공통 검색 완료 (Pro 모드 전체 데이터)")
    return response, panel_id_list, classification


async def _get_ordered_welcome_data(
    ids_to_fetch: List[str], 
    fields_to_fetch: List[str] = None
) -> List[dict]:
    """
    DB에서 패널 데이터를 조회하되, 입력된 id 리스트 순서를 보장하여 반환합니다.
    """
    if not ids_to_fetch:
        return []

    table_data = []
    try:
        with get_db_connection_context() as conn:
            if not conn:
                raise Exception("DB 연결 실패")

            cur = conn.cursor()

            sql_query = "SELECT panel_id, structured_data FROM welcome_meta2 WHERE panel_id = ANY(%s::text[])"
            cur.execute(sql_query, (ids_to_fetch,))
            results = cur.fetchall()

            fetched_data_map = {row[0]: row for row in results}

            for pid in ids_to_fetch:
                if pid in fetched_data_map:
                    row_data = fetched_data_map[pid]
                    panel_id_val, structured_data_val = row_data

                    display_data = {'panel_id': panel_id_val}

                    if fields_to_fetch:
                        if isinstance(structured_data_val, dict):
                            for field in fields_to_fetch:
                                if field != 'panel_id':
                                    display_data[field] = structured_data_val.get(field)
                    else:
                        if isinstance(structured_data_val, dict):
                            display_data.update(structured_data_val)

                    table_data.append(display_data)

            cur.close()

    except Exception as db_e:
        logging.error(f"Table Data 조회 실패: {db_e}", exc_info=True)
    
    return table_data

async def _get_qpoll_responses_for_table(
    ids_to_fetch: List[str], 
    qpoll_fields: List[str]
) -> Dict[str, Dict[str, str]]:
    """
    주어진 panel_id 목록과 Q-Poll 필드 목록에 대해 Qdrant에서 응답을 조회합니다.
    Returns: {panel_id: {qpoll_field: sentence}}
    """
    if not ids_to_fetch or not qpoll_fields:
        return {}
    
    questions_to_fetch = [QPOLL_FIELD_TO_TEXT[f] for f in qpoll_fields if f in QPOLL_FIELD_TO_TEXT]
    
    if not questions_to_fetch:
        return {}

    def qdrant_call():
        qpoll_client = get_qdrant_client()
        if not qpoll_client: return {}
        
        COLLECTION_NAME = "qpoll_vectors_v2"
        
        query_filter = Filter(
            must=[
                FieldCondition(key="panel_id", match=MatchAny(any=ids_to_fetch)),
                FieldCondition(key="question", match=MatchAny(any=questions_to_fetch))
            ]
        )
        
        qpoll_results, _ = qpoll_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query_filter,
            limit=len(ids_to_fetch) * len(questions_to_fetch),
            with_payload=True, with_vectors=False
        )

        result_map = {pid: {} for pid in ids_to_fetch}
        
        text_to_field_map = {v: k for k, v in QPOLL_FIELD_TO_TEXT.items()}
        
        for point in qpoll_results:
            pid = point.payload.get("panel_id")
            question = point.payload.get("question")
            sentence = point.payload.get("sentence")

            if pid and question and sentence:
                field_key = text_to_field_map.get(question)
                if field_key:
                    result_map[pid][field_key] = sentence
                    
        return result_map

    return await asyncio.get_running_loop().run_in_executor(None, qdrant_call)

@app.post("/api/search")
@cache(expire=600) # 10분 동안 캐시
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
        
        lite_response, _, _ = await _perform_common_search(
            search_query.query, 
            search_query.search_mode,
            mode="lite"
        )
        
        search_time = time.time() - start_time
        logging.info(f"⏱️  [Lite 모드] 검색 완료: {search_time:.2f}초")
        
        ids_to_fetch = lite_response.get('final_panel_ids', [])
        display_fields = lite_response.get('display_fields', [])
        
        qpoll_fields = [item['field'] for item in display_fields if item['field'] in QPOLL_FIELD_TO_TEXT]
        welcome_fields = [item['field'] for item in display_fields if item['field'] not in QPOLL_FIELD_TO_TEXT]
        
        db_start = time.time()
        
        welcome_table_data = await _get_ordered_welcome_data(ids_to_fetch, welcome_fields)
        qpoll_responses_map = await _get_qpoll_responses_for_table(ids_to_fetch, qpoll_fields)
        
        table_data = []
        for welcome_row in welcome_table_data:
            pid = welcome_row.get('panel_id')
            if pid and pid in qpoll_responses_map:
                welcome_row.update(qpoll_responses_map[pid])
            table_data.append(welcome_row)
            
        db_time = time.time() - db_start
        logging.info(f"✅ [Lite 모드] 통합 테이블 데이터 {len(table_data)}개 조회 완료: {db_time:.2f}초")
        
        lite_response['tableData'] = table_data
        lite_response['mode'] = "lite" 
        del lite_response['final_panel_ids']
        
        total_time = time.time() - start_time
        logging.info(f"✅ [Lite 모드] 전체 완료: {total_time:.2f}초 - 총 {lite_response['total_count']}개 결과 중 {len(table_data)}개 테이블 데이터 반환")
        
        return lite_response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"[Lite 모드] /api/search 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")


@app.post("/api/search-and-analyze")
# @cache(expire=600) # 10분 동안 캐시
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
        response, panel_id_list, classification = await _perform_common_search(
            request.query, 
            request.search_mode,
            mode="pro"
        )
        
        display_fields = response.get('display_fields', [])
        
        logging.info("📊 [Pro 모드] 차트 데이터 생성 시작")
        analysis_result, status_code = analyze_search_results(
            request.query, 
            classification,
            panel_id_list[:5000]
        )
        
        if status_code == 200:
            response['charts'] = analysis_result.get('charts', [])
            response['analysis_summary'] = analysis_result.get('main_summary', '')
            logging.info(f"✅ [Pro 모드] 차트 {len(response['charts'])}개 생성 완료")
        else:
            response['charts'] = []
            response['analysis_summary'] = '차트 생성 실패'
            logging.warning("[Pro 모드] 차트 생성 실패")
        
        logging.info(f"📊 [Pro 모드] Table Data 생성 시작 (패널 {len(panel_id_list)}개 대상)")
        ids_to_fetch = response['final_panel_ids']
        
        welcome_table_data = await _get_ordered_welcome_data(ids_to_fetch, fields_to_fetch=None)
        
        qpoll_fields = [
            item['field'] for item in display_fields 
            if item['field'] in QPOLL_FIELD_TO_TEXT
        ]

        qpoll_responses_map = await _get_qpoll_responses_for_table(ids_to_fetch, qpoll_fields)

        table_data = []
        for welcome_row in welcome_table_data:
            pid = welcome_row.get('panel_id')
            if pid and pid in qpoll_responses_map:
                welcome_row.update(qpoll_responses_map[pid])
            table_data.append(welcome_row)

        response['tableData'] = table_data
        response['mode'] = 'pro'
        
        logging.info(f"✅ [Pro 모드] 차트 {len(response.get('charts', []))}개, 통합 테이블 데이터 {len(table_data)}개 생성 완료")
        
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
    """PostgreSQL에서 Welcome 데이터를 비동기적으로 조회합니다."""
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

    return await asyncio.get_running_loop().run_in_executor(None, db_call)


async def _get_qpoll_data(panel_id: str) -> Dict:
    """Qdrant에서 QPoll 데이터를 비동기적으로 조회합니다."""
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

    return await asyncio.get_running_loop().run_in_executor(None, qdrant_call)


@app.get("/api/panels/{panel_id}")
async def get_panel_details(panel_id: str):
    """
    특정 panel_id의 패널 상세 정보를 병렬로 조회합니다.
    """
    try:
        logging.info(f"⚡️ 패널 상세 정보 병렬 조회 시작 (panel_id: {panel_id})")
        
        results = await asyncio.gather(
            _get_welcome_data(panel_id),
            _get_qpoll_data(panel_id),
            return_exceptions=True
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