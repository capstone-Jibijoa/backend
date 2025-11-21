import logging
import sys
import json
import re  # [추가] 정규식 사용
from typing import Dict, Optional, List

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, MatchText

from llm import parse_query_intelligent
from semantic_router import router
from search_helpers import search_welcome_objective, initialize_embeddings
from db import get_qdrant_client
from mapping_rules import QPOLL_FIELD_TO_TEXT

# [추가] 텍스트 기반 강력 필터링 패턴
# 이 단어가 포함되면 무조건 결과에서 제외합니다.
STRICT_NEGATIVE_PATTERNS = {
    "ott_count": [r"0개", r"없음", r"않음", r"안 함", r"안함", r"없다"],
    "moving_stress_factor": [r"없다", r"없음", r"안 받", r"않았", r"모르겠"],
    "pet_experience": [r"없다", r"키워본 적 없다", r"비반려"],
    "summer_worry": [r"없다", r"없음", r"걱정 없다"],
    "skincare_spending": [r"0원", r"안 쓴다", r"지출 없다"],
}

def hybrid_search(query: str, limit: Optional[int] = None) -> Dict:
    """
    Semantic Search V3 (Strict + Text Filter): 
    부정적인 답변을 텍스트 분석으로 완벽하게 제거합니다.
    """
    try:
        logging.info(f"🚀 Semantic Search V3 (Strict): {query}")
        
        # ... (LLM 파싱, 라우터, 1차 필터링 로직은 기존과 동일) ...
        # (1. LLM ~ 3. DB 필터링까지 기존 코드 그대로 유지)
        parsed_query = parse_query_intelligent(query) 
        structured_filters = parsed_query.get("demographic_filters", {})
        semantic_conditions = parsed_query.get("semantic_conditions", [])
        intent = ""
        if semantic_conditions:
            intent = semantic_conditions[0].get("original_keyword", "")
        user_limit = limit or parsed_query.get("limit", 100)
        
        target_field_info = router.find_closest_field(intent)
        target_field = None
        target_desc = None
        if target_field_info:
            target_field = target_field_info['field']
            target_desc = target_field_info['description']

        filtered_panel_ids = set()
        if structured_filters:
            # ... (search_helpers 호출 로직 그대로) ...
            filters_for_sql = []
            if "age_range" in structured_filters:
                filters_for_sql.append({"field": "age", "operator": "between", "value": structured_filters["age_range"]})
            for key, value in structured_filters.items():
                if key != "age_range":
                    filters_for_sql.append({"field": key, "operator": "in", "value": value})
            panel_ids, _ = search_welcome_objective(filters_for_sql, attempt_name="V3_Filter")
            filtered_panel_ids = panel_ids
        
        # 4. 벡터 검색 (Stage 2)
        # 필터링으로 많이 잘려나갈 것을 대비해 충분히 많이 가져옵니다. (5배 -> 10배)
        if filtered_panel_ids:
            vector_search_k = max(len(filtered_panel_ids), user_limit * 5)
            vector_search_k = min(vector_search_k, 1000)
        else:
            vector_search_k = max(user_limit * 5, 3000)

        final_panel_ids = filtered_panel_ids
        vector_matched_ids = set() 

        if intent and target_field:
            qdrant_client = get_qdrant_client()
            embeddings = initialize_embeddings()
            query_vector = embeddings.embed_query(intent)
            
            is_welcome_collection = False
            if target_field in QPOLL_FIELD_TO_TEXT:
                collection_name = "qpoll_vectors_v2"
                target_question = QPOLL_FIELD_TO_TEXT[target_field]
                id_key_path = "panel_id"
                is_welcome_collection = False
                must_conditions = [
                    FieldCondition(key="question", match=MatchText(text=target_question))
                ]
            else:
                collection_name = "welcome_subjective_vectors"
                id_key_path = "metadata.panel_id"
                is_welcome_collection = True
                must_conditions = []
            
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

            # ======================================================================
            # [핵심 수정] 텍스트 기반 정밀 필터링 (Hard Filtering)
            # 벡터가 아무리 비슷하다고 해도, 텍스트에 '없음'이 있으면 버립니다.
            # ======================================================================
            valid_hits_count = 0
            negative_patterns = STRICT_NEGATIVE_PATTERNS.get(target_field, [])
            
            for hit in search_results:
                if not hit.payload: continue
                
                # 1. 답변 텍스트 가져오기 (Q-Poll: sentence, Welcome: page_content or field)
                answer_text = ""
                if is_welcome_collection:
                    # Welcome 데이터는 구조에 따라 다름 (일반적으로 page_content에 문장 있음)
                    answer_text = hit.payload.get('page_content', "")
                    # 만약 page_content가 없고 특정 필드라면 해당 필드 값 확인 (필요시 로직 추가)
                else:
                    answer_text = hit.payload.get('sentence', "")

                # 2. [검사] 부정 패턴이 포함되어 있는지 확인
                is_negative = False
                for pattern in negative_patterns:
                    if re.search(pattern, answer_text):
                        is_negative = True
                        break # 하나라도 걸리면 아웃
                
                if is_negative:
                    # 부정 답변이므로 건너뜀 (결과에 포함 X)
                    continue

                # 3. ID 추출 및 저장
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
            
            # [Strict Mode] 강제 채우기 없음 (사용자가 원치 않음)
            final_panel_ids = vector_matched_ids

        else:
            logging.debug("  - 의도/타겟 없음. 1차 필터 결과 사용.")
            final_panel_ids = filtered_panel_ids

        final_panel_ids_list = list(final_panel_ids)[:user_limit]
        logging.info(f"✅ 검색 완료: {len(final_panel_ids_list)}명 (정밀 필터 적용됨)")

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