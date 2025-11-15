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
from analysis import (
    analyze_search_results_optimized as analyze_search_results,
    QPOLL_FIELD_TO_TEXT,
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
    [수정] ranked_keywords로부터 display_fields를 생성합니다.
    - LLM의 ranked_keywords_raw를 기반으로 필드를 매핑하고 유니크하게 만듭니다.
    """
    display_fields_raw = []
    
    # [디버깅 로그 1] LLM이 반환한 원본 키워드 리스트를 가져옵니다.
    ranked_keywords = classification.get('ranked_keywords_raw', [])
    if not ranked_keywords:
        logging.warning("⚠️ _prepare_display_fields: ranked_keywords_raw가 비어있습니다. 필드 매핑 건너뜀.")
        # 키워드가 없으면 빈 리스트 반환 (이후 main.py에서 Fallback이 처리함)
        return []

    for keyword in ranked_keywords[:5]:
        
        # [수정 필요]: 이 함수는 classification 전체가 아닌 keyword 리스트를 받습니다.
        # 따라서, 여기서는 keyword(str)를 analysis.py의 get_field_mapping에 넘겨야 합니다.
        
        # NOTE: get_field_mapping 함수는 analysis.py에서 import 되어 사용 가능하다고 가정합니다.
        mapping = get_field_mapping(keyword) 
        
        field = mapping.get('field', 'unknown')
        kw_type = mapping.get('type', 'filter')
        priority = 999 # 임시 우선순위
        
        # [디버깅 로그 2] 키워드별 매핑 결과를 확인합니다.
        logging.info(f"   [DEBUG_PREP] '{keyword}' 매핑 결과: {mapping}") 

        # 매핑이 성공하고 'unknown'이 아닌 경우에만 처리
        if field != 'unknown':
            # 매핑 함수가 필드(f)를 분리하지 않고 단일 필드를 반환한다고 가정
            f = field

            # Welcome 필드는 그대로 추가
            if kw_type == 'filter':
                # 필터 타입 필드(region_major, birth_year 등)
                display_fields_raw.append({
                    'field': f,
                    # FIELD_NAME_MAP은 utils.py에서 import 되어야 합니다.
                    'label': FIELD_NAME_MAP.get(f, f), 
                    'priority': priority
                })
            # QPoll 필드는 특별히 처리
            elif kw_type == 'qpoll':
                # QPOLL_FIELD_TO_TEXT는 analysis.py에서 import 되어야 합니다.
                display_fields_raw.append({
                    'field': f, 
                    'label': QPOLL_FIELD_TO_TEXT.get(f, f), 
                    'priority': priority
                })

    unique_display_fields_map = {}
    for item in display_fields_raw:
        if item['field'] not in unique_display_fields_map:
            unique_display_fields_map[item['field']] = item
    
    final_result = list(unique_display_fields_map.values())
    logging.info(f"   [DEBUG_PREP] 최종 매핑 필드: {final_result}") 
    
    # Fallback은 main.py의 호출 함수 (search_panels, search_and_analyze)에서 처리됨
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
    classification = classify_query_keywords(query_text) # LLM 분류 결과
    display_fields = _prepare_display_fields(classification)
    panel_ids_for_analysis = search_results['final_panel_ids']
    
    # Lite 모드 응답 간소화
    if mode == "lite":
        lite_response = {
            "query": query_text,
            "classification": classification,
            "display_fields": display_fields,
            "total_count": total_count,
            "final_panel_ids": panel_ids_for_analysis[:500], # 테이블 조회를 위해 최대 500개
            "effective_search_mode": effective_search_mode
        }

        # LLM 응답 구조에 맞춰 키를 명확히 삽입합니다.
        lite_response['classification']['ranked_keywords_raw'] = classification.get('ranked_keywords_raw', [])

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


async def _get_ordered_welcome_data(
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

    # **1. fields_to_fetch가 None인지 확인하고 분기 처리**
    if fields_to_fetch is not None:
        # Lite Mode: fields_to_fetch가 리스트일 때만 필터링 수행
        welcome_fields_to_fetch = [
            f for f in fields_to_fetch if f in FIELD_NAME_MAP or f == 'panel_id'
        ]
        # 필터링 후 남은 필드가 없으면 panel_id만 가져옵니다.
        if not welcome_fields_to_fetch:
            welcome_fields_to_fetch = ['panel_id']
    else:
        # Pro Mode: fields_to_fetch가 None일 때 (전체 structured_data 조회 의도)
        welcome_fields_to_fetch = None 
        # 이 경우, 아래 쿼리 로직에서 structured_data 전체를 가져오도록 처리됩니다.

    table_data = []
    try:
        with get_db_connection_context() as conn:
            if not conn:
                raise Exception("DB 연결 실패")
            
            cur = conn.cursor()
            
            # 2. SQL 쿼리 준비
            if welcome_fields_to_fetch is not None:
                # Lite Mode (특정 필드만 조회)
                fields_for_select = [f for f in welcome_fields_to_fetch if f != 'panel_id']
                
                if fields_for_select:
                    field_selects = ", ".join([
                        f"structured_data->>'{field}' as \"{field}\""
                        for field in fields_for_select
                    ])
                    sql_query = f"SELECT panel_id, {field_selects} FROM welcome_meta2 WHERE panel_id = ANY(%s::text[])"
                else:
                    # panel_id만 남은 경우
                    sql_query = "SELECT panel_id FROM welcome_meta2 WHERE panel_id = ANY(%s::text[])"
            else:
                # Pro Mode (structured_data 전체 조회)
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
                    
                    # 3. 데이터 파싱
                    if welcome_fields_to_fetch is not None:
                        # Lite 모드: 특정 필드만 포함된 딕셔너리 생성
                        data = {columns[i]: row_data[i] for i in range(len(columns))}
                    else:
                        # Pro 모드: structured_data 전체를 포함
                        data = row_data[1] or {}
                        data['panel_id'] = pid
                    table_data.append(data)
            
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
    (테이블 데이터 병합용)
    반환 형태: {panel_id: {qpoll_field: sentence}}
    """
    if not ids_to_fetch or not qpoll_fields:
        return {}
    
    questions_to_fetch = [QPOLL_FIELD_TO_TEXT[f] for f in qpoll_fields if f in QPOLL_FIELD_TO_TEXT]
    
    if not questions_to_fetch:
        return {}

    loop = asyncio.get_running_loop()
    
    def qdrant_call():
        qpoll_client = get_qdrant_client()
        if not qpoll_client: return {}
        
        COLLECTION_NAME = "qpoll_vectors_v2" # 환경변수 사용 권장: os.getenv("QDRANT_COLLECTION_NAME")
        
        # 1. 필터 구성: 주어진 panel_id 중 하나이고, 질문 텍스트 중 하나인 경우
        query_filter = Filter(
            must=[
                FieldCondition(key="panel_id", match=MatchAny(any=ids_to_fetch)),
                FieldCondition(key="question", match=MatchAny(any=questions_to_fetch))
            ]
        )
        
        # 2. Qdrant 스크롤 (응답 수 제한)
        qpoll_results, _ = qpoll_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query_filter,
            limit=len(ids_to_fetch) * len(questions_to_fetch), # 충분한 크기로 설정
            with_payload=True, with_vectors=False
        )

        result_map = {pid: {} for pid in ids_to_fetch}
        
        # 3. 결과 파싱 및 병합
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

    return await loop.run_in_executor(None, qdrant_call)

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
        display_fields = lite_response.get('display_fields', [])
        logging.info(f"lite_response: {lite_response}")
        logging.info(f"display_fields: {display_fields}")
        
        qpoll_fields = [item['field'] for item in display_fields if item['field'] in QPOLL_FIELD_TO_TEXT]
        welcome_fields = [item['field'] for item in display_fields if item['field'] not in QPOLL_FIELD_TO_TEXT]
        
        FALLBACK_WELCOME_FIELDS = ['gender', 'birth_year', 'family_size', 'job_duty_raw']
        
        if not welcome_fields and ids_to_fetch:
            logging.warning("⚠️ Welcome 필드 누락! 기본 필드를 Fallback으로 사용합니다.")
            welcome_fields = FALLBACK_WELCOME_FIELDS

        # 2. Welcome 필드가 4개 미만인 경우, '가족 수'를 추가하여 4개를 확보
        # 단, 이미 매핑된 필드가 아닌 경우에만 추가해야 합니다.
        FIELDS_TO_AUGMENT = ['family_size', 'job_duty_raw', 'marital_status'] # 보강 후보 필드

        current_welcome_fields_set = set(welcome_fields)
        
        for field_key in FIELDS_TO_AUGMENT:
            if len(welcome_fields) >= 4:
                break
            
            if field_key not in current_welcome_fields_set:
                logging.info(f"✨ Lite 모드: 테이블 컬럼 보강을 위해 '{FIELD_NAME_MAP.get(field_key)}' 필드를 추가합니다.")
                welcome_fields.append(field_key)
                current_welcome_fields_set.add(field_key)
        
        db_start = time.time()
        
        # Welcome 데이터 조회
        welcome_table_data = await _get_ordered_welcome_data(ids_to_fetch, welcome_fields)
        
        # QPoll 데이터 조회
        qpoll_responses_map = await _get_qpoll_responses_for_table(ids_to_fetch, qpoll_fields)
        
        # 데이터 병합 (Welcome 데이터 순서 유지)
        table_data = []
        for welcome_row in welcome_table_data:
            pid = welcome_row.get('panel_id')
            if pid and pid in qpoll_responses_map:
                welcome_row.update(qpoll_responses_map[pid])
            table_data.append(welcome_row)
            
        db_time = time.time() - db_start
        logging.info(f"✅ [Lite 모드] 통합 테이블 데이터 {len(table_data)}개 조회 완료: {db_time:.2f}초")
        
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

        panel_id_list = response['final_panel_ids']

        display_fields = response.get('display_fields', [])
        
        # QPOLL_FIELD_TO_TEXT에 없는 필드가 display_fields에 하나라도 있는지 확인
        # (즉, Welcome 필드가 분류되었는지 확인)
        has_welcome_fields = any(item['field'] not in QPOLL_FIELD_TO_TEXT for item in display_fields)
        
        # Lite Mode와 동일한 Fallback 필드 정의 (필수 인구 통계 필드)
        FALLBACK_WELCOME_FIELDS = ['gender', 'birth_year', 'family_size', 'job_duty_raw']
        
        FIELDS_TO_AUGMENT = ['family_size', 'job_duty_raw', 'marital_status'] 
        
        current_display_fields_set = set(item['field'] for item in display_fields)
        fields_to_add_to_display = []
        
        # 1. LLM이 Welcome 필드를 분류하지 못한 경우, 기본 필드 4개로 대체 (헤더 보장)
        if not has_welcome_fields and panel_id_list:
            logging.warning("⚠️ Pro 모드: Welcome 필드 누락! 기본 필드를 display_fields에 Fallback으로 추가합니다.")
            
            # 기존 display_fields를 비우고 Fallback 4개로 시작
            response['display_fields'] = [] 
            current_display_fields_set = set()
            
            for field_key in FALLBACK_WELCOME_FIELDS:
                korean_name = FIELD_NAME_MAP.get(field_key, field_key) 
                response['display_fields'].append({
                    'field': field_key,
                    'label': korean_name,
                    'priority': 999 
                })
                current_display_fields_set.add(field_key)
            
        # 2. LLM이 Welcome 필드를 분류했으나 4개 미만인 경우, 보강 후보로 채움
        elif len(response['display_fields']) < 4:
             for field_key in FIELDS_TO_AUGMENT:
                if len(response['display_fields']) >= 4:
                    break
                
                if field_key not in current_display_fields_set:
                    logging.info(f"✨ Pro 모드: 테이블 컬럼 보강을 위해 '{FIELD_NAME_MAP.get(field_key)}' 필드를 추가합니다.")
                    korean_name = FIELD_NAME_MAP.get(field_key, field_key)
                    
                    response['display_fields'].append({
                        'field': field_key,
                        'label': korean_name,
                        'priority': 999 
                    })
                    current_display_fields_set.add(field_key)
                    
        display_fields = response['display_fields'] # 업데이트된 리스트를 이후 로직에서 사용
        
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
        
        # Welcome 데이터 조회 (전체 structured_data)
        welcome_table_data = await _get_ordered_welcome_data(ids_to_fetch, fields_to_fetch=None)
        
        # Q-Poll 필드 목록 생성 (Pro 모드는 검색된 키워드만 가져옵니다.)
        qpoll_fields = [
            item['field'] for item in display_fields 
            if item['field'] in QPOLL_FIELD_TO_TEXT
        ]

        # QPoll 데이터 조회
        qpoll_responses_map = await _get_qpoll_responses_for_table(ids_to_fetch, qpoll_fields)

        # 데이터 병합
        table_data = []
        for welcome_row in welcome_table_data:
            pid = welcome_row.get('panel_id')
            if pid and pid in qpoll_responses_map:
                welcome_row.update(qpoll_responses_map[pid])
            table_data.append(welcome_row)

        # 4. 최종 응답 구성
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