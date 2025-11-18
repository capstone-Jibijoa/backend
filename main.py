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
from search import initialize_embeddings
from search import hybrid_search as hybrid_search
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
    
    # [수정] must_have, preference를 우선하고 objective는 후순위로 변경
    objective_kws = classification.get('objective_keywords', [])
    must_have_kws = classification.get('must_have_keywords', [])
    preference_kws = classification.get('preference_keywords', [])
    
    # 핵심 주제(must-have, preference)를 먼저, 필터링 조건(objective)은 나중에
    header_keywords = must_have_kws + preference_kws + objective_kws

    if not header_keywords:
        logging.warning("⚠️ _prepare_display_fields: 분석할 키워드가 없습니다. 빈 헤더 반환.")
        return []
        
    # [신규] objective_keywords 중 단일 값만 갖는 필드를 식별하여 헤더에서 제외
    objective_field_counts = {}
    for kw in objective_kws:
        mapping = get_field_mapping(kw)
        field = mapping.get('field')
        if field and field != 'unknown':
            if field not in objective_field_counts:
                objective_field_counts[field] = 0
            objective_field_counts[field] += 1
    
    # 값이 하나만 있는 필드(뻔한 결과)는 제외 대상
    single_value_fields_to_exclude = {field for field, count in objective_field_counts.items() if count == 1}
    logging.info(f"✨ [Display Fields] 단일 값 필드 제외 대상: {single_value_fields_to_exclude}")

    unique_fields = {}
    priority_counter = 0

    for i, keyword in enumerate(header_keywords):
        # 이미 5개 헤더가 채워졌으면 중단
        if len(unique_fields) >= 5:
            break

        mapping = get_field_mapping(keyword)
        field = mapping.get('field')

        # [신규] 제외 대상 필드인 경우 건너뛰기 (단, objective 키워드에 대해서만 적용)
        if keyword in objective_kws and field in single_value_fields_to_exclude:
            logging.info(f"   → '{keyword}'(필드: {field})는 단일 조건이므로 헤더에서 제외")
            continue

        if field and field != 'unknown' and field not in unique_fields:
            label = mapping.get('description')
            # QPoll 필드의 경우, description이 질문 전체이므로 FIELD_NAME_MAP에서 짧은 이름으로 대체
            if mapping.get('type') == 'qpoll':
                label = FIELD_NAME_MAP.get(field, field)

            unique_fields[field] = {
                'field': field,
                'label': label,
                'priority': i # 원래 순서를 우선순위로 사용
            }
            priority_counter += 1

    # [신규 추가] 벡터 검색 결과에서 발견된 Category 기반으로 필드 보강
    found_categories = classification.get('found_categories', [])
    if found_categories:
        logging.info(f"✨ [Display Fields] 벡터 카테고리 기반 필드 보강 시작: {found_categories}")
        for category in found_categories:
            if len(unique_fields) >= 5:
                break
            
            fields_to_add = VECTOR_CATEGORY_TO_FIELD.get(category, [])
            for field in fields_to_add:
                if len(unique_fields) >= 5:
                    break
                if field not in unique_fields:
                    label = FIELD_NAME_MAP.get(field, field)
                    logging.info(f"   → 카테고리 '{category}'를 통해 '{label}' 필드 추가")
                    unique_fields[field] = {
                        'field': field, 'label': label, 'priority': 800 + len(unique_fields)
                    }

    # [수정] 컬럼 보강 로직을 이 함수로 통합
    if len(unique_fields) < 4:
        FIELDS_TO_AUGMENT = ['family_size', 'job_duty_raw', 'marital_status']
        
        for field_key in FIELDS_TO_AUGMENT:
            if len(unique_fields) >= 4:
                break
            
            if field_key not in unique_fields:
                korean_name = FIELD_NAME_MAP.get(field_key, field_key)
                logging.info(f"✨ [Display Fields] 테이블 컬럼 보강: '{korean_name}' 추가")
                unique_fields[field_key] = {
                    'field': field_key,
                    'label': korean_name,
                    'priority': 900 + len(unique_fields) # 보강 필드는 낮은 우선순위
                }

    final_result = sorted(list(unique_fields.values()), key=lambda x: x['priority'])
    logging.info(f"   [DEBUG_PREP] 최종 매핑 필드: {final_result}") 
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
    # v2 검색 결과 구조에 맞게 source_counts를 stage_details에서 가져오도록 수정
    source_counts = { 
        "stage1_objective": search_results.get("stage_details", {}).get("stage1_objective", 0),
        "stage2_must_have": search_results.get("stage_details", {}).get("stage2_must_have", 0),
        "stage3_preference": search_results.get("stage_details", {}).get("stage3_preference", 0),
        "stage4_negative": search_results.get("stage_details", {}).get("stage4_negative", 0),
    }
    
    summary = {
        "search_mode": effective_search_mode,
        # "search_strategy": { # [수정] classification 키 구조 변경 대응
        #     "welcome_objective": bool(classification.get('objective_keywords')),
        #     "welcome_subjective": bool(classification.get('vector_keywords')),
        #     "qpoll": bool(classification.get('qpoll_keywords'))
        # },
        "ranked_keywords": classification.get('ranked_keywords_raw', [])
    }

    response = {
        "query": query_text,
        "classification": classification,
        "display_fields": display_fields,
        "source_counts": source_counts,
        "summary": summary,
    }

    # v2 검색 결과 구조에 맞게 응답 포맷팅
    final_panel_ids = search_results.get('final_panel_ids', [])
    total_count = search_results.get('total_count', 0)
    
    # Pro 모드는 최대 100개의 ID만 반환
    panel_id_list = final_panel_ids[:100]
    response["final_panel_ids"] = panel_id_list

    # 'results' 필드 구조 단순화
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
    
    # 1. LLM 키워드 분류
    classification = classify_query_keywords(query_text)
    logging.info(f"🤖 LLM 분류 결과: {classification}")
    user_limit = classification.get('limit')
    effective_search_mode = "quota" if user_limit and user_limit > 0 else search_mode
    logging.info(f"💡 API: 감지된 Limit 값: {user_limit}")
    
    search_results = hybrid_search(
        query=query_text,
        limit=user_limit
    )
    
    # 3. 검색 로그 기록
    panel_id_list = search_results.get('final_panel_ids', []) # [수정] hybrid_search 결과에서 final_panel_ids 추출
    total_count = len(panel_id_list)
    log_search_query(query_text, total_count)
    
    # 4. 응답 구성
    # [수정] classification 객체에 ranked_keywords_raw 추가
    classification['ranked_keywords_raw'] = classification.get('objective_keywords', []) + classification.get('mandatory_keywords', [])
    display_fields = _prepare_display_fields(classification)
    
    if mode == "lite":
        lite_response = {
            "query": query_text,
            "classification": classification,
            "total_count": total_count,
            "final_panel_ids": panel_id_list[:500], # 테이블 조회를 위해 최대 500개
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
    return response, panel_id_list, classification # [수정] 원본 classification 반환


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

    table_data = []
    try:
        with get_db_connection_context() as conn:
            if not conn:
                raise Exception("DB 연결 실패")

            cur = conn.cursor()

            # [수정] Lite/Pro 모드 구분 없이 structured_data 전체를 조회하여 일관성 확보
            sql_query = "SELECT panel_id, structured_data FROM welcome_meta2 WHERE panel_id = ANY(%s::text[])"
            cur.execute(sql_query, (ids_to_fetch,))
            results = cur.fetchall()

            # 순서 재정렬을 위한 맵 생성
            fetched_data_map = {row[0]: row for row in results}

            # 입력된 ID 순서대로 결과 재구성
            for pid in ids_to_fetch:
                if pid in fetched_data_map:
                    row_data = fetched_data_map[pid]
                    panel_id_val, structured_data_val = row_data

                    # 최종적으로 테이블에 표시될 데이터 객체
                    display_data = {'panel_id': panel_id_val}

                    # fields_to_fetch가 제공되면 (Lite 모드), 해당 필드만 추출
                    if fields_to_fetch:
                        if isinstance(structured_data_val, dict):
                            for field in fields_to_fetch:
                                if field != 'panel_id':
                                    display_data[field] = structured_data_val.get(field)
                    # fields_to_fetch가 None이면 (Pro 모드), structured_data 전체를 병합
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
        
        # 2. 테이블 데이터 조회
        ids_to_fetch = lite_response.get('final_panel_ids', [])
        display_fields = lite_response.get('display_fields', [])
        logging.info(f"lite_response: {lite_response}")
        logging.info(f"display_fields: {display_fields}")
        
        qpoll_fields = [item['field'] for item in display_fields if item['field'] in QPOLL_FIELD_TO_TEXT]
        welcome_fields = [item['field'] for item in display_fields if item['field'] not in QPOLL_FIELD_TO_TEXT]
        
        # [최종 수정] 모든 컬럼 보강 로직을 _prepare_display_fields로 통합했으므로, 여기서는 모두 제거합니다.
        
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
        
        display_fields = response.get('display_fields', [])
        
        # [최종 수정] 모든 컬럼 보강 로직을 _prepare_display_fields로 통합했으므로, 여기서는 모두 제거합니다.
        
        # 2. 차트 데이터 생성
        logging.info("📊 [Pro 모드] 차트 데이터 생성 시작")
        analysis_result, status_code = analyze_search_results(
            request.query, 
            classification,
            panel_id_list[:5000] # 분석은 최대 5000개
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