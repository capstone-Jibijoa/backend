# services.py
import logging
import re
import asyncio
import time
from typing import List, Dict, Any, Tuple, Optional
from fastapi import Request, Response, HTTPException

from repository import PanelRepository, VectorRepository
from insights import (
    get_ai_summary, 
    analyze_search_results_optimized as analyze_search_results,
    get_search_result_overview
)
from llm import parse_query_intelligent
from search_helpers import initialize_embeddings
from search import hybrid_search
from mapping_rules import (
    QPOLL_FIELD_TO_TEXT, 
    QPOLL_ANSWER_TEMPLATES, 
    VECTOR_CATEGORY_TO_FIELD,
    find_related_fields
)
from db import (
    get_db_connection_context,
    get_qdrant_client
)
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from utils import FIELD_NAME_MAP

# --- Helper Functions (유틸리티 함수) ---
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

def custom_key_builder(func, namespace: str = "", *, request: Request = None, response: Response = None, **kwargs):
    if request:
        sorted_query_params = sorted(request.query_params.items())
        return ":".join([namespace, request.method.lower(), request.url.path, repr(sorted_query_params), func.__module__ + func.__name__])
    return ":".join([namespace, func.__module__ + func.__name__, repr(sorted(kwargs.items()))])

def preload_models():
    from semantic_router import router
    logging.info("="*70)
    logging.info("🔄 모든 AI 모델을 미리 로드합니다...")
    start = time.time()
    initialize_embeddings()
    try: router.find_closest_field("테스트 쿼리")
    except: pass
    logging.info(f"✅ 모델 로드 완료 ({time.time() - start:.2f}초)")

# --- Core Logic Functions (핵심 로직) ---
def _prepare_display_fields(classification: Dict, query_text: str = "") -> List[Dict]:
    relevant_fields = {"gender", "birth_year", "region_major"} 
    target_field = classification.get('target_field')
    
    is_qpoll_search = target_field and target_field in QPOLL_FIELD_TO_TEXT
    if is_qpoll_search:
        relevant_fields.update(["job_title_raw", "education_level", "income_household_monthly"])
    
    if target_field and target_field != 'unknown':
        relevant_fields.add(target_field)
        for category, fields in VECTOR_CATEGORY_TO_FIELD.items():
            if target_field in fields:
                relevant_fields.update(fields)
                break

    structured_filters = classification.get('structured_filters', {}) or classification.get('demographic_filters', {})
    filter_keys = []
    if isinstance(structured_filters, dict): filter_keys = structured_filters.keys()
    elif isinstance(structured_filters, list): filter_keys = [f.get('field') for f in structured_filters if f.get('field')]
    relevant_fields.update(filter_keys)

    if query_text:
        dynamic_fields = find_related_fields(query_text)
        relevant_fields.update(dynamic_fields)

    final_list = []
    if target_field and target_field != 'unknown':
        label = FIELD_NAME_MAP.get(target_field, target_field)
        if target_field in QPOLL_FIELD_TO_TEXT: label = QPOLL_FIELD_TO_TEXT[target_field]
        final_list.append({'field': target_field, 'label': label})
        relevant_fields.discard(target_field)

    for field in relevant_fields:
        if field in FIELD_NAME_MAP:
            final_list.append({'field': field, 'label': FIELD_NAME_MAP[field]})
            
    return final_list[:12]

async def _perform_common_search(query_text: str, search_mode: str, mode: str) -> Tuple[Dict, List[str], Dict]:
    logging.info(f"🔍 공통 검색 시작: {query_text} (모드: {search_mode}, 실행: {mode})")
    classification = parse_query_intelligent(query_text)
    user_limit = classification.get('limit', 100)
    
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

async def _get_ordered_welcome_data(ids_to_fetch: List[str], fields_to_fetch: List[str] = None) -> List[dict]:
    if not ids_to_fetch: return []
    table_data = []
    
    # Repository 호출
    results = await asyncio.to_thread(PanelRepository.fetch_ordered_table_data, ids_to_fetch)

    for row in results:
        panel_id_val, structured_data_val = row
        if not structured_data_val: continue
        
        display_data = {'panel_id': panel_id_val}
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
        
    return table_data

async def _get_qpoll_responses_for_table(ids_to_fetch: List[str], qpoll_fields: List[str]) -> Dict[str, Dict[str, str]]:
    if not ids_to_fetch or not qpoll_fields: return {}
    questions_to_fetch = [QPOLL_FIELD_TO_TEXT[f] for f in qpoll_fields if f in QPOLL_FIELD_TO_TEXT]
    if not questions_to_fetch: return {}

    def process_qpoll():
        qpoll_results = VectorRepository.fetch_qpoll_responses(ids_to_fetch, questions_to_fetch)
        
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
        
    return await asyncio.get_running_loop().run_in_executor(None, process_qpoll)

async def _get_welcome_data(panel_id: str) -> Dict:
    result = await asyncio.to_thread(PanelRepository.fetch_panel_detail, panel_id)
    if not result:
        raise HTTPException(status_code=404, detail="패널 정보를 찾을 수 없습니다.")
    return result

async def _get_qpoll_data(panel_id: str) -> Dict:
    def process():
        q_data = {"qpoll_응답_개수": 0}
        res = VectorRepository.fetch_qpoll_for_panel(panel_id)
        
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
    return await asyncio.get_running_loop().run_in_executor(None, process)