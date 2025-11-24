import logging
import re
from typing import Dict, Optional, List, Set

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, MatchText
from utils import WELCOME_OBJECTIVE_FIELDS
from llm import parse_query_intelligent
from semantic_router import router
from search_helpers import (
    search_welcome_objective, 
    initialize_embeddings, 
    filter_negative_conditions, 
    embed_keywords
)
from db import get_qdrant_client
from mapping_rules import QPOLL_FIELD_TO_TEXT

STRICT_NEGATIVE_PATTERNS = {
    "ott_count": [r"0개", r"없음", r"않음", r"안 함", r"안함", r"없다"],
    "moving_stress_factor": [r"없다", r"없음", r"안 받", r"않았", r"모르겠"],
    "pet_experience": [r"없다", r"키워본 적 없다", r"비반려"],
    "summer_worry": [r"없다", r"없음", r"걱정 없다"],
    "skincare_spending": [r"0원", r"안 쓴다", r"지출 없다"],
}

def hybrid_search(query: str, limit: Optional[int] = None) -> Dict:
    """
    Semantic Search V3 (Strict + Text Filter + Optimized): 
    부정적인 답변을 텍스트 분석과 벡터 분석으로 이중 제거합니다.
    """
    try:
        logging.info(f"🚀 Semantic Search V3 (Optimized): {query}")
        
        # 1. LLM 파싱
        parsed_query = parse_query_intelligent(query) 
        
        # 긍정 조건과 부정 조건 분리
        all_conditions = parsed_query.get("semantic_conditions", [])
        positive_conditions = [c for c in all_conditions if not c.get('is_negative', False)]
        negative_conditions = [c for c in all_conditions if c.get('is_negative', False)]

        structured_filters = parsed_query.get("demographic_filters", {})
        user_limit = limit or parsed_query.get("limit", 100)

        # 의도(Intent) 파악은 긍정 조건 기준 (부정 조건은 검색어가 아님)
        intent = ""
        if positive_conditions:
            intent = positive_conditions[0].get("original_keyword", "")
        
        # 2. 라우팅 (어떤 질문/필드를 검색할지 결정)
        target_field_info = router.find_closest_field(intent)
        target_field = None
        target_desc = None
        if target_field_info:
            target_field = target_field_info['field']
            target_desc = target_field_info['description']

        # 인구통계 필드 목록 (이름만 추출)
        objective_field_names = [f[0] for f in WELCOME_OBJECTIVE_FIELDS]
        
        # 만약 현재 타겟이 인구통계 필드라면?
        if target_field in objective_field_names:
            # 다른 semantic condition 중에 Q-Poll 관련이 있는지 찾아본다
            for cond in all_conditions:
                kw = cond.get('original_keyword', '')
                # 현재 타겟이 된 키워드는 패스
                if kw == intent: continue 
                
                # 다른 키워드로 라우팅 시도
                alt_info = router.find_closest_field(kw)
                
                # QPOLL_FIELD_TO_TEXT에 있는 필드(설문)라면 교체!
                if alt_info and alt_info['field'] in QPOLL_FIELD_TO_TEXT:
                    logging.info(f"🔄 타겟 재설정: {target_field}(인구통계) -> {alt_info['field']}(설문)로 변경")
                    target_field = alt_info['field']
                    target_desc = alt_info['description']
                    intent = kw # 검색 의도 키워드도 변경
                    break

        # 3. 1차 필터링 (SQL - 인구통계)
        filtered_panel_ids = set()
        if structured_filters:
            filters_for_sql = []
            if "age_range" in structured_filters:
                filters_for_sql.append({"field": "age", "operator": "between", "value": structured_filters["age_range"]})
            for key, value in structured_filters.items():
                if key != "age_range":
                    filters_for_sql.append({"field": key, "operator": "in", "value": value})

            if target_field and target_field not in QPOLL_FIELD_TO_TEXT:
                filters_for_sql.append({"field": target_field, "operator": "not_null", "value": "check"})
            
            if filters_for_sql:
                panel_ids, _ = search_welcome_objective(filters_for_sql, attempt_name="V3_Filter_Optimized")
                filtered_panel_ids = panel_ids
        
        # 4. 2차 검색 (Vector Search)
        if filtered_panel_ids:
            vector_search_k = max(len(filtered_panel_ids), user_limit * 5)
            vector_search_k = min(vector_search_k, 1000)
        else:
            vector_search_k = max(user_limit * 5, 500)

        final_panel_ids = filtered_panel_ids
        vector_matched_ids = set() 

        is_structured_target = target_field and target_field not in QPOLL_FIELD_TO_TEXT
        
        # 조건: 정형 데이터 타겟이고 + SQL 필터로 찾은 사람이 있다면 -> 벡터 검색 안 함!
        if is_structured_target and filtered_panel_ids:
            logging.info(f"🎯 정형 데이터 타겟({target_field}) 감지 -> 벡터 검색 없이 SQL 결과({len(filtered_panel_ids)}명) 사용")
            final_panel_ids = filtered_panel_ids
            
        # 기존 벡터 검색 로직 (정형 데이터가 아니거나, SQL 결과가 없을 때만 실행)
        elif intent and target_field:
            qdrant_client = get_qdrant_client()
            embeddings = initialize_embeddings()
            query_vector = embeddings.embed_query(intent)
            
            # 컬렉션 결정
            is_welcome_collection = False
            if target_field in QPOLL_FIELD_TO_TEXT:
                collection_name = "qpoll_vectors_v2"
                target_question = QPOLL_FIELD_TO_TEXT[target_field]
                id_key_path = "panel_id"
                must_conditions = [
                    FieldCondition(key="question", match=MatchText(text=target_question))
                ]
            else:
                collection_name = "welcome_subjective_vectors"
                id_key_path = "metadata.panel_id"
                is_welcome_collection = True
                must_conditions = []
            
            # SQL 필터링된 ID가 있다면 Qdrant 필터에도 추가
            str_panel_ids = [str(pid) for pid in filtered_panel_ids]
            if filtered_panel_ids:
                must_conditions.append(
                    FieldCondition(key=id_key_path, match=MatchAny(any=str_panel_ids))
                )
            
            qdrant_filter = Filter(must=must_conditions)

            search_results = []
            try:
                search_results = qdrant_client.search(
                    collection_name=collection_name, 
                    query_vector=query_vector,
                    query_filter=qdrant_filter,
                    limit=vector_search_k,
                    with_payload=True 
                )
            except Exception as e:
                logging.error(f"❌ Qdrant 검색 실패: {e}")
                search_results = []

            # [검증 1] 텍스트 기반 정밀 필터링 (Regex Strict Mode)
            valid_hits_count = 0
            negative_patterns = STRICT_NEGATIVE_PATTERNS.get(target_field, [])
            
            # 검색 결과 순회
            for hit in search_results:
                if not hit.payload: continue
                
                # 답변 텍스트 추출
                answer_text = ""
                if is_welcome_collection:
                    answer_text = hit.payload.get('page_content', "")
                else:
                    answer_text = hit.payload.get('sentence', "")

                # 정규식 부정 패턴 검사
                is_negative = False
                for pattern in negative_patterns:
                    if re.search(pattern, answer_text):
                        is_negative = True
                        break 
                
                if is_negative:
                    continue # 결과 제외

                # ID 추출
                pid = None
                if is_welcome_collection:
                    meta = hit.payload.get('metadata', {})
                    pid = meta.get('panel_id')
                else:
                    pid = hit.payload.get('panel_id')
                
                if pid:
                    vector_matched_ids.add(pid)
                    valid_hits_count += 1
            
            logging.info(f"   ✂️ 텍스트 필터링 결과: 검색 {len(search_results)}명 -> 유효 {valid_hits_count}명")
            
            # [검증 2] 벡터 기반 부정 조건 필터링 (New Optimized Logic)
            # LLM이 파악한 부정 조건(예: '고양이 안 키우는')과 유사한 벡터를 가진 사람 제외
            if negative_conditions and vector_matched_ids:
                 neg_keywords = []
                 for nc in negative_conditions:
                     # 긍정문으로 변환된 쿼리(expanded_queries)를 사용해 유사도 검사
                     neg_keywords.extend(nc.get('expanded_queries', []))
                 
                 if neg_keywords:
                     logging.info(f"🚫 부정 조건 필터링 적용 (벡터): {neg_keywords}")
                     
                     # 부정 키워드 벡터화
                     neg_vectors = embed_keywords(neg_keywords)
                     
                     # 해당 벡터와 유사한 사람들을 찾아 현재 결과에서 제외
                     vector_matched_ids = filter_negative_conditions(
                         panel_ids=vector_matched_ids,
                         negative_keywords=neg_keywords,
                         query_vectors=neg_vectors,
                         qdrant_client=qdrant_client,
                         collection_name=collection_name,
                         threshold=0.55 # 부정 유사도 임계값 (너무 높으면 못 거르고, 너무 낮으면 다 걸러짐)
                     )
                     logging.info(f"   ✂️ 벡터 부정 필터링 후 남은 인원: {len(vector_matched_ids)}명")

            final_panel_ids = vector_matched_ids

        else:
            logging.debug("  - 의도/타겟 없음. 1차 필터 결과 사용.")
            final_panel_ids = filtered_panel_ids

        final_panel_ids_list = list(final_panel_ids)
        logging.info(f"✅ 검색 완료: {len(final_panel_ids_list)}명")

        return {
            "final_panel_ids": final_panel_ids_list,
            "total_count": len(final_panel_ids_list),
            "search_intent": intent,
            "target_field": target_field,
            "target_field_desc": target_desc
        }

    except Exception as e:
        logging.error(f"❌ hybrid_search 오류: {e}", exc_info=True)
        return {
            "final_panel_ids": [],
            "total_count": 0,
            "search_intent": "",
            "target_field": None,
            "target_field_desc": None
        }