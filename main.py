import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Tuple, Dict, List
from fastapi.middleware.cors import CORSMiddleware

# [최적화] import
from hybrid_logic_optimized import classify_query_keywords # 1. LLM 캐싱 적용
from search_logic import initialize_embeddings # 4. 모델 미리 로딩
from search_logic_optimized import hybrid_search_parallel as hybrid_search # 5. 검색 병렬화
from analysis_logic_optimized import analyze_search_results_optimized as analyze_search_results # 2. DB 집계 분석
from db_logic_optimized import ( # 3. Connection Pool
    log_search_query,
    get_db_connection_context,
    init_db,
    cleanup_db,
    get_qdrant_client
)
from qdrant_client.models import Filter, FieldCondition, MatchValue


# FastAPI 애플리케이션 초기화
app = FastAPI(title="Multi-Table Hybrid Search API v3 (Optimized & Refactored)")

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

# 4단계: 모델 미리 로딩 함수
def preload_models():
    """애플리케이션 시작 시 모든 AI 모델을 미리 로드합니다."""
    print("\n" + "="*70)
    print("🔄 모든 AI 모델을 미리 로드합니다...")
    # 1. KURE 임베딩 모델 로드
    initialize_embeddings()
    # 2. Claude LLM 모델 로드 (테스트 호출로 초기화 유도)
    classify_query_keywords("모델 로딩 테스트")
    print("✅ 모든 AI 모델 로드 완료")
    print("="*70 + "\n")

@app.on_event("startup")
async def startup_event():
    print("🚀 FastAPI 시작...")
    init_db()
    preload_models()

@app.on_event("shutdown")
async def shutdown_event():
    print("🧹 FastAPI 종료... Connection Pool 정리")
    cleanup_db()


class SearchQuery(BaseModel):
    query: str
    search_mode: str = "all"

# (SearchResponse, AnalysisRequest 등 다른 Pydantic 모델은 동일)
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

# ====================================================================
# 🚀 [리팩토링] 공통 검색 로직 함수
# ====================================================================

async def _perform_common_search(query_text: str, search_mode: str, mode: str) -> Tuple[Dict, List[str], Dict]:
    """
    /search와 /search-and-analyze가 공유하는 핵심 로직
    (LLM 분류, 병렬 검색, 로그 기록, 결과 포맷팅)
    
    Args:
        query_text (str): 검색 질의
        search_mode (str): 검색 모드 (all/weighted/union/intersection)
        mode (str): 실행 모드 ("lite" 또는 "pro")
    
    Returns:
        Tuple[dict, list, dict]: (response, panel_id_list, classification)
        - Lite 모드일 경우 response는 간소화된 딕셔너리
    """
    print(f"\n{'='*70}")
    print(f"🔍 공통 검색 시작: {query_text} (모드: {search_mode}, 실행: {mode})")
    print(f"{'='*70}\n")
    
    # 1단계: LLM 키워드 분류 (캐싱 없음)
    print("📌 1단계: LLM 키워드 분류")
    classification = classify_query_keywords(query_text)
    
    # 1.5단계: 분류 결과에서 limit 값 추출
    user_limit = classification.get('limit')
    print(f"💡 API: 감지된 Limit 값: {user_limit}")
    
    # 2단계: 하이브리드 검색 수행 (병렬 처리)
    search_results = hybrid_search(
        classification, 
        search_mode=search_mode,
        limit=user_limit
    )
    
    # 3단계: 검색 로그 기록 (Connection Pool 사용)
    if user_limit is not None and user_limit > 0:
        total_count = len(search_results['final_panel_ids'])
    elif search_mode == "all":
        total_count = search_results['results']['union']['count']
    else:
        total_count = len(search_results['final_panel_ids']) 
    
    log_search_query(query_text, total_count)
    
    # 4단계: 응답 구성
    display_fields_raw = []
    # 💡 [수정] max 5개 필드만 추출
    for kw_info in classification.get('ranked_keywords', [])[:5]:
        field = kw_info.get('field', '')
        description = kw_info.get('description', '')
        priority = kw_info.get('priority', 999)
        
        # ⭐️ [수정] 복합 필드 처리: 쉼표로 분리하여 각 필드를 추가
        fields = [f.strip() for f in field.split(',')]
        
        for f in fields:
            if f:
                # ⭐️ [추가] display_fields에는 개별 필드와 그 설명을 포함
                display_fields_raw.append({
                    'field': f,
                    'label': description, # 복합 필드라도 동일한 설명 사용
                    'priority': priority
                })

    # ⭐️ [추가] display_fields를 중복 제거 후 리스트로 변환 (프론트엔드에서 테이블 헤더로 사용)
    # 딕셔너리를 사용하여 field를 키로 중복 제거
    unique_display_fields_map = {}
    for item in display_fields_raw:
        if item['field'] not in unique_display_fields_map:
            unique_display_fields_map[item['field']] = item
    display_fields = list(unique_display_fields_map.values())

    effective_search_mode = search_mode
    if user_limit is not None and user_limit > 0:
        effective_search_mode = "quota"

    # 차트 분석에 사용할 panel_id_list 준비
    panel_id_list_all = search_results['final_panel_ids']
    
    # ⭐️ [수정] Lite 모드 응답 간소화
    if mode == "lite":
        lite_response = {
            "query": query_text,
            "classification": {
                "ranked_keywords": classification.get('ranked_keywords', [])
            },
            "display_fields": display_fields,
            "total_count": len(panel_id_list_all),
            # Lite 모드는 테이블 데이터 조회를 위해 최대 500개만 반환
            "final_panel_ids": panel_id_list_all[:500], 
            "effective_search_mode": effective_search_mode
        }
        print(f"✅ 공통 검색 완료 (Lite 모드 간소화)")
        # Lite 모드의 경우, panel_id_list_full 대신 간소화된 응답을 반환
        return lite_response, panel_id_list_all, classification

    # ====================================================================
    # Pro 모드 (기존 로직 유지)
    # ====================================================================
    panel_id_list = [] # Pro 모드 분석에 사용할 ID 리스트
    
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
        # 'all' 모드의 기본값은 'weighted' 결과
        panel_id_list = search_results['results']['weighted']['panel_ids']
    
    else:
        # 단일 모드 결과 반환 (quota, weighted, union, intersection)
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
        # 분석을 위해 최대 5000개까지 사용
        panel_id_list = final_panel_ids[:5000]
    
    print(f"✅ 공통 검색 완료 (Pro 모드 전체 데이터)")
    return response, panel_id_list, classification

# ====================================================================
# 1. 메인 검색 API
# ====================================================================

@app.post("/api/search")
async def search_panels(search_query: SearchQuery):
    """
    🚀 Lite 모드: 빠른 검색 (차트 분석 없이 테이블 데이터만 반환)
    - 검색 결과 목록만 빠르게 제공
    - 차트 데이터 생성 과정 생략으로 응답 속도 향상
    - 최소한의 필드만 조회하여 DB 부하 감소
    """
    print(f"\n{'='*70}")
    print(f"🚀 [Lite 모드] 빠른 검색 시작: {search_query.query}")
    print(f"{'='*70}\n")
    
    valid_modes = ["all", "weighted", "union", "intersection"]
    if search_query.search_mode not in valid_modes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search_mode. Must be one of: {valid_modes}"
        )
    
    try:
        import time
        start_time = time.time()
        
        # 🚀 공통 검색 함수 호출 (mode="lite" 전달)
        lite_response, panel_id_list_full, classification = await _perform_common_search(
            search_query.query, 
            search_query.search_mode,
            mode="lite" # ⭐️ mode 인자 추가
        )
        
        search_time = time.time() - start_time
        print(f"⏱️  [Lite 모드] 검색 완료: {search_time:.2f}초")
        
        # 💡 Lite 모드: tableData만 추가 (차트 분석 생략)
        table_data = []
        
        # ⭐️ [수정] final_panel_ids를 간소화된 응답에서 가져옴
        ids_to_fetch = lite_response['final_panel_ids']
        
        if ids_to_fetch and len(ids_to_fetch) > 0:
            try:
                db_start = time.time()
                print(f"📊 [Lite 모드] 테이블 데이터 조회 시작 (최대 {len(ids_to_fetch)}개)")
                
                # ✅ 최적화: display_fields만 선택적으로 조회
                # ⭐️ [수정] display_fields는 이미 간소화된 response에 있으므로 그것을 사용
                fields_to_fetch = [item['field'] for item in lite_response.get('display_fields', [])]
                
                with get_db_connection_context() as conn:
                    if conn:
                        cur = conn.cursor()
                        
                        # ✅ 최적화: 필요한 필드만 선택 (전체 조회보다 빠름)
                        if fields_to_fetch:
                            # ⭐️ [수정] 복합 필드 방지 로직 적용
                            field_selects = ", ".join([
                                f"structured_data->>'{field}' as {field}"
                                for field in fields_to_fetch
                            ])
                            sql_query = f"""
                                SELECT panel_id, {field_selects}
                                FROM welcome_meta2
                                WHERE panel_id = ANY(%s::text[])
                            """
                        else:
                            # fallback: 전체 조회
                            sql_query = """
                                SELECT panel_id, structured_data
                                FROM welcome_meta2
                                WHERE panel_id = ANY(%s::text[])
                            """
                        
                        cur.execute(sql_query, (ids_to_fetch,))
                        results = cur.fetchall()
                        
                        if fields_to_fetch:
                            # 필드 선택 모드: 검색 순서대로 테이블 데이터 생성
                            fetched_data_map = {row[0]: {fields_to_fetch[i]: row[i+1] for i in range(len(fields_to_fetch))} for row in results}
                            
                            for pid in ids_to_fetch:
                                if pid in fetched_data_map:
                                    data = {'panel_id': pid}
                                    data.update(fetched_data_map[pid])
                                    table_data.append(data)
                        else:
                            # 전체 조회 모드: 검색 순서대로 테이블 데이터 생성
                            fetched_data_map = {row[0]: row[1] for row in results}
                            
                            for pid in ids_to_fetch:
                                if pid in fetched_data_map:
                                    data = fetched_data_map[pid]
                                    if isinstance(data, dict):
                                        data['panel_id'] = pid
                                        table_data.append(data)
                                    else:
                                        table_data.append({"panel_id": pid})
                        
                        db_time = time.time() - db_start
                        print(f"✅ [Lite 모드] 테이블 데이터 {len(table_data)}개 조회 완료: {db_time:.2f}초")
                                    
            except Exception as db_e:
                print(f"❌ [Lite 모드] Table Data 조회 실패: {db_e}")
                import traceback
                traceback.print_exc()
        
        # 💥 Lite 모드 최종 응답 구성 (간소화된 response 사용)
        lite_response['tableData'] = table_data
        lite_response['mode'] = "lite" 
        
        # ⭐️ [수정] final_panel_ids는 테이블 데이터 조회를 위해 사용되었으므로, 최종 응답에서 제거
        del lite_response['final_panel_ids']
        
        total_time = time.time() - start_time
        print(f"✅ [Lite 모드] 전체 완료: {total_time:.2f}초 - 총 {lite_response['total_count']}개 결과 중 {len(table_data)}개 테이블 데이터 반환")
        
        return lite_response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ [Lite 모드] /api/search 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")

# ====================================================================
# 2. 검색 + 분석 통합 API (NEW!)
# ====================================================================

@app.post("/api/search-and-analyze")
async def search_and_analyze(request: AnalysisRequest):
    """
    📊 Pro 모드: 검색 + 인사이트 분석 (차트 + 테이블 데이터 반환)
    - 검색 결과에 대한 차트 시각화 제공
    - 테이블 데이터와 분석 결과를 함께 반환
    """
    print(f"\n{'='*70}")
    print(f"📊 [Pro 모드] 검색 + 분석 시작: {request.query}")
    print(f"{'='*70}\n")
    
    valid_modes = ["all", "weighted", "union", "intersection"]
    if request.search_mode not in valid_modes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid search_mode. Must be one of: {valid_modes}"
        )
    
    try:
        # 🚀 1. 공통 검색 함수 호출 (mode="pro" 전달)
        response, panel_id_list, classification = await _perform_common_search(
            request.query, 
            request.search_mode,
            mode="pro" # ⭐️ mode 인자 추가
        )
        
        # 🚀 2. [Pro 모드 고유] 차트 데이터 생성 (DB 집계 쿼리 사용)
        print("\n📊 [Pro 모드] 차트 데이터 생성 시작")
        analysis_result, status_code = analyze_search_results(
            request.query,
            classification,
            panel_id_list
        )
        
        if status_code == 200:
            response['charts'] = analysis_result.get('charts', [])
            response['analysis_summary'] = analysis_result.get('main_summary', '')
            print(f"✅ [Pro 모드] 차트 {len(response['charts'])}개 생성 완료")
        else:
            response['charts'] = []
            response['analysis_summary'] = '차트 생성 실패'
            print(f"⚠️  [Pro 모드] 차트 생성 실패")
        
        # ===============================================================
        # 💥 [Pro 모드] Table Data 생성을 위해 DB 조회
        # ===============================================================
        print(f"\n📊 [Pro 모드] Table Data 생성 시작 (패널 {len(panel_id_list)}개 대상)")
        table_data = []
        
        # 프론트엔드가 페이지네이션으로 100개만 보여주므로,
        # DB 부하 및 네트워크 효율성을 위해 상위 100개만 조회합니다.
        # response['final_panel_ids']는 이미 100개로 잘려있으므로 그것을 사용합니다.
        ids_to_fetch = response['final_panel_ids']
        
        if ids_to_fetch:
            try:
                with get_db_connection_context() as conn:
                    with conn.cursor() as cur:
                        # SQL IN 절을 사용
                        sql_query = """
                            SELECT panel_id, structured_data 
                            FROM welcome_meta2 
                            WHERE panel_id IN %s
                        """
                        # IN 절에 튜플 형태로 ID 리스트 전달
                        cur.execute(sql_query, (tuple(ids_to_fetch),))
                        
                        results = cur.fetchall()
                        
                        # DB에서 가져온 결과는 순서가 보장되지 않으므로,
                        # 맵을 만들어 검색 순서(ids_to_fetch)대로 재정렬합니다.
                        fetched_data_map = {row[0]: row[1] for row in results}
                        
                        # ids_to_fetch (검색 점수 순서)를 기준으로 table_data 리스트 생성
                        for pid in ids_to_fetch:
                            if pid in fetched_data_map:
                                data = fetched_data_map[pid]
                                if isinstance(data, dict):
                                    data['panel_id'] = pid # panel_id를 데이터에 포함
                                    table_data.append(data)
                                else:
                                    # structured_data가 dict가 아닌 경우 (예: null)
                                    table_data.append({"panel_id": pid})
                                    
            except Exception as db_e:
                print(f"❌ [Pro 모드] Table Data 조회 실패: {db_e}")
                # 실패해도 차트 결과는 반환하도록 table_data는 비워둠
        
        # 💥 최종 응답에 tableData와 mode 추가
        response['tableData'] = table_data
        response['mode'] = 'pro'  # 응답 모드 명시
        
        print(f"✅ [Pro 모드] 차트 {len(response['charts'])}개, 테이블 데이터 {len(table_data)}개 생성 완료")
        print(f"\n✅ [Pro 모드] 검색+분석 완료")
        
        return response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ [Pro 모드] /api/search-and-analyze 실패: {e}")
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
# 4. 패널 상세 정보 조회 API (Connection Pool 적용)
# ====================================================================

@app.get("/api/panels/{panel_id}")
async def get_panel_details(panel_id: str):
    """
    특정 panel_id의 패널 상세 정보를 조회합니다.
    - Welcome 데이터 (PostgreSQL)
    - QPoll 질문/응답 데이터 (Qdrant) - 평탄화하여 통합
    """
    try:
        # ============================================================
        # 1. PostgreSQL에서 Welcome 데이터 조회
        # ============================================================
        with get_db_connection_context() as conn:
            if not conn:
                raise HTTPException(status_code=500, detail="데이터베이스 연결 실패")
            
            cur = conn.cursor()
            
            cur.execute("""
                SELECT panel_id, structured_data
                FROM welcome_meta2 
                WHERE panel_id = %s
            """, (panel_id,))
            
            result = cur.fetchone()
            
            if not result:
                cur.close()
                raise HTTPException(
                    status_code=404, 
                    detail=f"panel_id {panel_id}를 찾을 수 없습니다."
                )
            
            panel_id_value, structured_data = result
            
            # Welcome 데이터 구성
            panel_data = {"panel_id": panel_id_value}
            if isinstance(structured_data, dict):
                panel_data.update(structured_data)
            
            cur.close()
        
        # ============================================================
        # 2. Qdrant에서 QPoll 데이터 조회 및 평탄화
        # ============================================================
        try:
            qdrant_client = get_qdrant_client()
            
            if qdrant_client:
                print(f"🔍 QPoll 데이터 조회 시작 (panel_id: {panel_id})")
                
                # Qdrant에서 panel_id로 필터링하여 검색
                qpoll_results = qdrant_client.scroll(
                    collection_name="qpoll_vectors_v2",
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="panel_id",
                                match=MatchValue(value=panel_id)
                            )
                        ]
                    ),
                    limit=100,  # 최대 100개 질문/응답
                    with_payload=True,
                    with_vectors=False  # 벡터는 불필요
                )
                
                # ✅ QPoll 데이터를 평탄화하여 panel_data에 추가
                if qpoll_results and qpoll_results[0]:  # (points, next_page_offset)
                    points = qpoll_results[0]
                    print(f"✅ QPoll 응답 {len(points)}개 발견")
                    
                    for idx, point in enumerate(points, 1):
                        if point.payload:
                            question = point.payload.get("question", "")
                            sentence = point.payload.get("sentence", "")
                            
                            # ✅ "qpoll_1_질문", "qpoll_1_응답" 형식으로 저장
                            panel_data[f"qpoll_{idx:03d}_질문"] = question
                            panel_data[f"qpoll_{idx:03d}_응답"] = sentence
                    
                    # QPoll 개수 저장
                    panel_data["qpoll_응답_개수"] = len(points)
                    print(f"✅ QPoll 데이터 {len(points)}개 평탄화 완료")
                else:
                    print("⚠️  QPoll 응답 없음")
                    panel_data["qpoll_응답_개수"] = 0
            
            else:
                print("⚠️  Qdrant 클라이언트 없음")
                panel_data["qpoll_응답_개수"] = 0
        
        except Exception as qpoll_error:
            # QPoll 조회 실패 시에도 Welcome 데이터는 반환
            print(f"❌ QPoll 조회 실패 (panel_id: {panel_id}): {qpoll_error}")
            import traceback
            traceback.print_exc()
            panel_data["qpoll_응답_개수"] = 0
            panel_data["qpoll_조회_오류"] = str(qpoll_error)
        
        return panel_data
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"조회 실패: {str(e)}")
    

# ====================================================================
# 5. 헬스체크
# ====================================================================

@app.get("/")
def read_root():
    return {
        "service": "Multi-Table Hybrid Search & Analysis API",
        "version": "3.0 (Optimized & Refactored)",
        "status": "running",
        "optimizations_applied": [
            "DB Connection Pool (psycopg2-pool)",
            "Parallel Search (ThreadPoolExecutor)",
            "DB Aggregate Queries (analysis_logic_optimized)"
        ],
        "optimizations_excluded": [
            "Redis LLM Caching"
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
    """시스템 상태 확인 (Connection Pool 사용)"""
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