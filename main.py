import os
import json
import time
import logging
import re
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, Tuple, Dict, List, Callable, Awaitable, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import Response
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

from llm import parse_query_intelligent
from search_helpers import initialize_embeddings
from search import hybrid_search as hybrid_search
import asyncio
from mapping_rules import QPOLL_FIELD_TO_TEXT, QPOLL_ANSWER_TEMPLATES, VECTOR_CATEGORY_TO_FIELD
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

load_dotenv()

app = FastAPI(title="Multi-Table Hybrid Search API v3 (Refactored)")

# --- CORS Middleware ---
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

# --- Helper Functions ---
def truncate_text(value: Any, max_length: int = 30) -> str:
    if value is None: return ""
    if isinstance(value, list):
        text = ", ".join(map(str, value))
    else:
        text = str(value)
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def extract_answer_from_template(field_name: str, sentence: str) -> str:
    if not sentence: return ""
    if field_name == "ott_count":
        match = re.search(r'(\d+개|이용 안 함|없음)', sentence)
        if match: return match.group(1)
    elif field_name == "skincare_spending":
        match = re.search(r'(\d+만\s*원|\d+~\d+만\s*원|\d+원)', sentence)
        if match: return match.group(1)

    template = QPOLL_ANSWER_TEMPLATES.get(field_name)
    if template:
        try:
            pattern_str = re.escape(template)
            pattern_str = pattern_str.replace(re.escape("{answer_str}"), r"(.*?)")
            pattern_str = pattern_str.replace(r"\(이\)다", r"(?:이)?다")
            pattern_str = pattern_str.replace(r"\(으\)로", r"(?:으)?로")
            pattern_str = pattern_str.replace(r"\(가\)", r"(?:가)?")
            pattern_str = pattern_str.replace(r"\ ", r"\s*")

            match = re.search(pattern_str, sentence)
            if match:
                extracted = match.group(1)
                cleaned = re.sub(r'\([^)]*\)', '', extracted).strip()
                return truncate_text(cleaned, 20)
        except: pass

    cleaned = re.sub(r'\([^)]*\)', '', str(sentence)).strip()
    return truncate_text(cleaned, 30)

# --- Caching Setup ---
def custom_key_builder(func, namespace: str = "", *, request: Request = None, response: Response = None, **kwargs):
    if request:
        sorted_query_params = sorted(request.query_params.items())
        return ":".join([namespace, request.method.lower(), request.url.path, repr(sorted_query_params), func.__module__ + func.__name__])
    return ":".join([namespace, func.__module__ + func.__name__, repr(sorted(kwargs.items()))])

# --- Model Preloading ---
def preload_models():
    import time
    from semantic_router import router
    logging.info("="*70)
    logging.info("🔄 모든 AI 모델을 미리 로드합니다...")
    start = time.time()
    initialize_embeddings()
    try: router.find_closest_field("테스트 쿼리")
    except: pass
    logging.info(f"✅ 모델 로드 완료 ({time.time() - start:.2f}초)")

@app.on_event("startup")
async def startup_event():
    logging.info("🚀 FastAPI 시작...")
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache", key_builder=custom_key_builder)
    init_db()
    preload_models()

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("🧹 FastAPI 종료... Connection Pool 정리")
    cleanup_db()

class SearchQuery(BaseModel):
    query: str
    search_mode: str = "all"

class AnalysisRequest(BaseModel):
    query: str
    search_mode: str = "weighted"

def _prepare_display_fields(classification: Dict, chart_fields: Optional[List[str]] = None) -> List[Dict]:
    relevant_categories = {"DEMO_BASIC"}
    target_field = classification.get('target_field')
    
    if target_field and target_field != 'unknown':
        for category, fields in VECTOR_CATEGORY_TO_FIELD.items():
            if target_field in fields:
                relevant_categories.add(category)
                break

    structured_filters = classification.get('structured_filters', {}) or classification.get('demographic_filters', {})
    filter_keys = []
    if isinstance(structured_filters, dict): filter_keys = structured_filters.keys()
    elif isinstance(structured_filters, list): filter_keys = [f.get('field') for f in structured_filters if f.get('field')]

    for f_key in filter_keys:
        for category, fields in VECTOR_CATEGORY_TO_FIELD.items():
            if f_key in fields: relevant_categories.add(category)

    CATEGORY_ORDER = ["DEMO_BASIC", "FAMILY_STATUS", "JOB_EDUCATION", "INCOME_LEVEL", "TECH_OWNER", "CAR_OWNER", "DRINK_HABIT", "SMOKE_HABIT"]
    unique_fields = {}
    priority_counter = 0

    if target_field and target_field != 'unknown':
        label = FIELD_NAME_MAP.get(target_field, target_field)
        if target_field in QPOLL_FIELD_TO_TEXT: label = QPOLL_FIELD_TO_TEXT[target_field]
        unique_fields[target_field] = {'field': target_field, 'label': label, 'priority': priority_counter}
        priority_counter += 1

    for cat in CATEGORY_ORDER:
        if cat in relevant_categories:
            cat_fields = VECTOR_CATEGORY_TO_FIELD.get(cat, [])
            for field in cat_fields:
                if field not in unique_fields:
                    unique_fields[field] = {'field': field, 'label': FIELD_NAME_MAP.get(field, field), 'priority': priority_counter}
                    priority_counter += 1
    
    final_result = sorted(list(unique_fields.values()), key=lambda x: x['priority'])
    return final_result[:8]
 
async def _perform_common_search(query_text: str, search_mode: str, mode: str) -> Tuple[Dict, List[str], Dict]:
    logging.info(f"🔍 공통 검색 시작: {query_text} (모드: {search_mode}, 실행: {mode})")
    classification = parse_query_intelligent(query_text)
    user_limit = classification.get('limit')
    
    search_results = await asyncio.to_thread(hybrid_search, query=query_text, limit=user_limit)
    
    panel_id_list = search_results.get('final_panel_ids', [])
    total_count = len(panel_id_list)
    classification['target_field'] = search_results.get('target_field')
    
    if mode == "lite":
        lite_response = {
            "query": query_text,
            "classification": classification,
            "total_count": total_count,
            "final_panel_ids": panel_id_list[:500],
            "effective_search_mode": "quota",
        }
        return lite_response, panel_id_list, classification

    pro_mode_info = {
        "query": query_text,
        "classification": classification,
        "search_results": search_results,
        "effective_search_mode": "quota",
        "final_panel_ids": panel_id_list
    }
    return pro_mode_info, panel_id_list, classification

# [수정] 엄격한 필터링 제거됨! (데이터가 있으면 무조건 가져옴)
async def _get_ordered_welcome_data(ids_to_fetch: List[str], fields_to_fetch: List[str] = None) -> List[dict]:
    if not ids_to_fetch: return []
    table_data = []
    try:
        with get_db_connection_context() as conn:
            if not conn: raise Exception("DB 연결 실패")
            cur = conn.cursor()
            sql_query = "SELECT panel_id, structured_data FROM welcome_meta2 WHERE panel_id = ANY(%s::text[])"
            cur.execute(sql_query, (ids_to_fetch,))
            results = cur.fetchall()
            fetched_data_map = {row[0]: row for row in results}

            for pid in ids_to_fetch:
                if pid in fetched_data_map:
                    row_data = fetched_data_map[pid]
                    panel_id_val, structured_data_val = row_data
                    if not structured_data_val: continue

                    display_data = {'panel_id': panel_id_val}
                    # 여기서는 필터링(is_valid_row = False)을 하지 않고, 있는 그대로 담습니다.
                    if fields_to_fetch:
                        for field in fields_to_fetch:
                            if field != 'panel_id':
                                val = structured_data_val.get(field)
                                display_data[field] = truncate_text(val) 
                    else:
                        for k, v in structured_data_val.items():
                            if k != 'panel_id':
                                display_data[k] = truncate_text(v)
                    table_data.append(display_data)
            cur.close()
    except Exception as db_e:
        logging.error(f"Table Data 조회 실패: {db_e}")
    return table_data

async def _get_qpoll_responses_for_table(ids_to_fetch: List[str], qpoll_fields: List[str]) -> Dict[str, Dict[str, str]]:
    if not ids_to_fetch or not qpoll_fields: return {}
    questions_to_fetch = [QPOLL_FIELD_TO_TEXT[f] for f in qpoll_fields if f in QPOLL_FIELD_TO_TEXT]
    if not questions_to_fetch: return {}

    def qdrant_call():
        qpoll_client = get_qdrant_client()
        if not qpoll_client: return {}
        query_filter = Filter(must=[FieldCondition(key="panel_id", match=MatchAny(any=ids_to_fetch)), FieldCondition(key="question", match=MatchAny(any=questions_to_fetch))])
        qpoll_results, _ = qpoll_client.scroll(collection_name="qpoll_vectors_v2", scroll_filter=query_filter, limit=len(ids_to_fetch) * len(questions_to_fetch), with_payload=True, with_vectors=False)
        result_map = {pid: {} for pid in ids_to_fetch}
        text_to_field_map = {v: k for k, v in QPOLL_FIELD_TO_TEXT.items()}
        for point in qpoll_results:
            pid = point.payload.get("panel_id")
            question = point.payload.get("question")
            sentence = point.payload.get("sentence")
            if pid and question and sentence:
                field_key = text_to_field_map.get(question)
                if field_key:
                    core_value = extract_answer_from_template(field_key, sentence)
                    result_map[pid][field_key] = core_value
        return result_map
    return await asyncio.get_running_loop().run_in_executor(None, qdrant_call)

@app.post("/api/search")
async def search_panels(search_query: SearchQuery):
    logging.info(f"🚀 [Lite 모드] 빠른 검색 시작: {search_query.query}")
    try:
        lite_response, _, _ = await _perform_common_search(search_query.query, search_query.search_mode, mode="lite")
        ids_to_fetch = lite_response.get('final_panel_ids', [])
        
        display_fields = _prepare_display_fields(lite_response['classification'])
        welcome_fields = [item['field'] for item in display_fields if item['field'] not in QPOLL_FIELD_TO_TEXT]
        qpoll_fields = [item['field'] for item in display_fields if item['field'] in QPOLL_FIELD_TO_TEXT]
        
        welcome_table_data, qpoll_responses_map = await asyncio.gather(
            _get_ordered_welcome_data(ids_to_fetch, welcome_fields),
            _get_qpoll_responses_for_table(ids_to_fetch, qpoll_fields)
        )
        
        table_data = []
        target_field = lite_response.get('classification', {}).get('target_field')

        for welcome_row in welcome_table_data:
            pid = welcome_row.get('panel_id')
            if pid and pid in qpoll_responses_map:
                welcome_row.update(qpoll_responses_map[pid])
            
            # [수정] 필터링 로직: Target Field만 체크
            is_valid_row = True
            
            # 1. Q-Poll 타겟인 경우
            if target_field in qpoll_fields:
                val = welcome_row.get(target_field)
                if not val or str(val).strip().lower() == 'nan': is_valid_row = False
            
            # 2. Welcome 타겟인 경우 (예: drinking_experience) - DB 필터로 이미 걸러졌겠지만 한번 더 체크
            elif target_field in welcome_fields:
                val = welcome_row.get(target_field)
                if not val or str(val).strip().lower() == 'nan': is_valid_row = False

            # 나머지 필드 빈 값 처리
            for field in (welcome_fields + qpoll_fields):
                if field != target_field:
                    val = welcome_row.get(field)
                    if not val or str(val).strip().lower() == 'nan': welcome_row[field] = "-"

            if is_valid_row:
                table_data.append(welcome_row)
            
        final_limit = user_limit
        lite_response['tableData'] = table_data[:final_limit]
        lite_response['display_fields'] = display_fields
        lite_response['mode'] = "lite" 
        if 'final_panel_ids' in lite_response: del lite_response['final_panel_ids']
        return lite_response
    except Exception as e:
        logging.error(f"[Lite 모드] 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search-and-analyze")
async def search_and_analyze(request: AnalysisRequest):
    logging.info(f"📊 [Pro 모드] 검색 + 분석 시작: {request.query}")
    try:
        pro_info, panel_ids, classification = await _perform_common_search(request.query, request.search_mode, mode="pro")
        user_limit = classification.get('limit', 100)

        analysis_result, _ = await asyncio.to_thread(analyze_search_results, request.query, classification, panel_ids[:5000])
        charts = analysis_result.get('charts', []) if analysis_result else []
        
        display_fields = _prepare_display_fields(classification)
        welcome_fields = [item['field'] for item in display_fields if item['field'] not in QPOLL_FIELD_TO_TEXT]
        qpoll_fields = [item['field'] for item in display_fields if item['field'] in QPOLL_FIELD_TO_TEXT]

        fetch_limit = max(user_limit * 2, 500) 
        ids_to_fetch = panel_ids[:fetch_limit]

        welcome_table_data, qpoll_responses_map = await asyncio.gather(
            _get_ordered_welcome_data(ids_to_fetch, fields_to_fetch=welcome_fields),
            _get_qpoll_responses_for_table(ids_to_fetch, qpoll_fields)
        )

        table_data = []
        target_field = classification.get('target_field')
        
        for welcome_row in welcome_table_data:
            pid = welcome_row.get('panel_id')
            if pid and pid in qpoll_responses_map:
                welcome_row.update(qpoll_responses_map[pid])
            
            # [수정] 필터링 로직: Target Field만 체크
            is_valid_row = True
            if target_field in qpoll_fields:
                val = welcome_row.get(target_field)
                if not val or str(val).strip().lower() == 'nan': is_valid_row = False
            elif target_field in welcome_fields:
                val = welcome_row.get(target_field)
                if not val or str(val).strip().lower() == 'nan': is_valid_row = False
            
            for field in (welcome_fields + qpoll_fields):
                if field != target_field:
                    val = welcome_row.get(field)
                    if not val or str(val).strip().lower() == 'nan': welcome_row[field] = "-"

            if is_valid_row:
                table_data.append(welcome_row)
        
        response_data = {
            "query": pro_info["query"],
            "classification": classification,
            "display_fields": display_fields,
            "charts": charts,
            "tableData": table_data[:user_limit], # 100개 제한
            "total_count": len(panel_ids), 
            "mode": 'pro'
        }
        return response_data
    except Exception as e:
        logging.error(f"[Pro 모드] 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ... (debug endpoint 및 get_panel_details, read_root, health_check 등은 동일하게 유지) ...
@app.post("/api/debug/classify")
async def debug_classify(search_query: SearchQuery):
    try:
        classification = parse_query_intelligent(search_query.query)
        return {"query": search_query.query, "classification": classification}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _get_welcome_data(panel_id: str) -> Dict:
    def db_call():
        with get_db_connection_context() as conn:
            if not conn: raise HTTPException(status_code=500)
            cur = conn.cursor()
            cur.execute("SELECT panel_id, structured_data FROM welcome_meta2 WHERE panel_id = %s", (panel_id,))
            result = cur.fetchone()
            cur.close()
            if not result: raise HTTPException(status_code=404)
            pid, data = result
            p_data = {"panel_id": pid}
            if isinstance(data, dict): p_data.update(data)
            return p_data
    return await asyncio.get_running_loop().run_in_executor(None, db_call)

async def _get_qpoll_data(panel_id: str) -> Dict:
    q_data = {"qpoll_응답_개수": 0}
    def qdrant_call():
        try:
            client = get_qdrant_client()
            if not client: return q_data
            res, _ = client.scroll(
                collection_name="qpoll_vectors_v2",
                scroll_filter=Filter(must=[FieldCondition(key="panel_id", match=MatchValue(value=panel_id))]),
                limit=100, with_payload=True, with_vectors=False
            )
            if res:
                q_data["qpoll_응답_개수"] = len(res)
                txt_map = {v: k for k, v in QPOLL_FIELD_TO_TEXT.items()}
                for p in res:
                    if p.payload:
                        q = p.payload.get("question")
                        s = p.payload.get("sentence")
                        if q and s:
                            k = txt_map.get(q)
                            if k: q_data[k] = s
            return q_data
        except: return q_data
    return await asyncio.get_running_loop().run_in_executor(None, qdrant_call)

@app.get("/api/panels/{panel_id}")
async def get_panel_details(panel_id: str):
    try:
        logging.info(f"⚡️ 패널 상세 정보 병렬 조회 시작 (panel_id: {panel_id})")
        results = await asyncio.gather(_get_welcome_data(panel_id), _get_qpoll_data(panel_id), return_exceptions=True)
        panel_data, qpoll_data = {}, {}
        for result in results:
            if isinstance(result, HTTPException): raise result 
            elif isinstance(result, Exception): raise HTTPException(status_code=500, detail=f"조회 중 내부 오류 발생: {result}")
            if "qpoll_응답_개수" in result: qpoll_data = result
            else: panel_data = result
        panel_data.update(qpoll_data)
        return panel_data
    except HTTPException: raise
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