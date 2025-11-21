import os
import re
import logging
import threading
from typing import List, Set, Optional, Dict, Tuple
from datetime import datetime
from collections import defaultdict
from functools import lru_cache
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, SearchParams
from langchain_huggingface import HuggingFaceEmbeddings

from db import get_db_connection_context
from mapping_rules import CATEGORY_MAPPING, get_field_mapping

# [수정] LLM 필드명 -> 실제 DB 필드명 매핑 (여기에 household_size 추가!)
FIELD_ALIAS_MAP = {
    "household_size": "family_size",  # 1인 가구 -> 가족 수
    "age": "birth_year",              # 나이 -> 출생연도
    "job": "job_title_raw",           # 직업
    "region": "region_major"          # 지역
}

# 값 변환이 필요한 필드를 위한 매핑
VALUE_TRANSLATION_MAP = {
    'gender': {
        '남성': 'M', '여성': 'F', '남자': 'M', '여자': 'F',
    },
    'marital_status': {
        '미혼': '미혼', '싱글': '미혼', '기혼': '기혼', '결혼': '기혼', '이혼': '이혼', '돌싱': '이혼'
    }
}

ARRAY_FIELDS = [
    "drinking_experience",
    "owned_electronics",
    "smoking_experience",
    "smoking_brand",
    "e_cigarette_experience"
]

def build_sql_from_structured_filters(filters: List[Dict]) -> Tuple[str, List]:
    """
    JSONB 데이터 타입에 맞춰 정확한 SQL WHERE 절을 생성합니다.
    (속도 최적화를 위해 TRIM 제거 및 필드 매핑 적용)
    """
    if not filters:
        return "", []

    conditions = []
    params = []
    CURRENT_YEAR = datetime.now().year

    for f in filters:
        raw_field = f.get("field")
        operator = f.get("operator")
        value = f.get("value")

        if not raw_field or not operator:
            continue

        # [핵심 1] 필드명 매핑 적용 (household_size -> family_size)
        field = FIELD_ALIAS_MAP.get(raw_field, raw_field)

        # --- [특수 처리] 나이 계산 (Age -> birth_year) ---
        # raw_field가 'age'이거나 매핑된 field가 'birth_year'인 경우
        if field == "birth_year" or raw_field == "age":
            if operator == "between" and isinstance(value, list) and len(value) == 2:
                age_start, age_end = value
                birth_year_end = CURRENT_YEAR - age_start
                birth_year_start = CURRENT_YEAR - age_end
                # 인덱스 활용을 위해 형변환
                conditions.append(f"(structured_data->>'birth_year')::int BETWEEN %s AND %s")
                params.extend([birth_year_start, birth_year_end])
            continue
        
        # --- [값 변환] 매핑된 값으로 변환 ---
        final_value = value
        if field in VALUE_TRANSLATION_MAP:
            if isinstance(value, list):
                final_value = [VALUE_TRANSLATION_MAP[field].get(v, v) for v in value]
            else:
                final_value = VALUE_TRANSLATION_MAP[field].get(value, value)

        # --- [분기 1] JSON 배열(List) 필드 처리 ---
        if field in ARRAY_FIELDS:
            if operator == "in" and isinstance(final_value, list):
                placeholders = ','.join(['%s'] * len(final_value))
                conditions.append(f"structured_data->'{field}' ?| array[{placeholders}]")
                params.extend(final_value)
            elif operator == "eq":
                conditions.append(f"structured_data->'{field}' ? %s")
                params.append(str(final_value))

        # --- [분기 2] 숫자형 필드 처리 ---
        elif field in ["children_count"]:
            field_sql = f"(structured_data->>'{field}')::numeric"
            if operator == "between" and isinstance(final_value, list) and len(final_value) == 2:
                conditions.append(f"{field_sql} BETWEEN %s AND %s")
                params.extend(final_value)
            elif operator == "gte":
                conditions.append(f"{field_sql} >= %s")
                params.append(final_value)
            elif operator == "lte":
                conditions.append(f"{field_sql} <= %s")
                params.append(final_value)
            elif operator == "eq":
                conditions.append(f"{field_sql} = %s")
                params.append(final_value)

        # --- [분기 3] 일반 문자열 필드 처리 ---
        else:
            # [핵심 2] 속도 최적화를 위해 TRIM() 제거 -> structured_data->>'field'
            field_sql = f"structured_data->>'{field}'"

            # family_size(1인 가구 등)는 숫자 '1' 검색 시 '1명', '1인' 등을 모두 찾아야 함
            if field == "family_size":
                if isinstance(final_value, list):
                    or_conditions = []
                    for v in final_value:
                        or_conditions.append(f"{field_sql} ILIKE %s")
                        params.append(f"%{v}%")
                    conditions.append(f"({' OR '.join(or_conditions)})")
                else:
                    conditions.append(f"{field_sql} ILIKE %s")
                    params.append(f"%{final_value}%")

            elif operator == "eq":
                # job_title_raw 등은 부분 일치(ILIKE)가 안전
                if field in ["job_title_raw", "job_duty_raw"]:
                     conditions.append(f"{field_sql} ILIKE %s")
                     params.append(f"%{final_value}%")
                else:
                     # 그 외(지역, 성별 등)는 정확 일치 (인덱스 활용 최적화)
                     conditions.append(f"{field_sql} = %s")
                     params.append(str(final_value))

            elif operator == "in" and isinstance(final_value, list) and final_value:
                str_values = [str(v) for v in final_value]
                placeholders = ','.join(['%s'] * len(str_values))
                conditions.append(f"{field_sql} IN ({placeholders})")
                params.extend(str_values)
                
            elif operator == "like":
                conditions.append(f"{field_sql} ILIKE %s")
                params.append(f"%{final_value}%")

    if not conditions:
        return "", []

    where_clause = " WHERE " + " AND ".join(conditions)
    return where_clause, params


def search_welcome_objective(
    filters: List[Dict],
    attempt_name: str = "구조화"
) -> Tuple[Set[str], Set[str]]:
    """
    Stage 1: LLM 필터 -> PostgreSQL 필터링
    """
    if not filters:
        logging.info(f"   Welcome {attempt_name}: 필터 없음")
        return set(), set()

    try:
        with get_db_connection_context() as conn:
            if not conn:
                return set(), set()

            cur = conn.cursor()
            where_clause, params = build_sql_from_structured_filters(filters)

            if not where_clause:
                return set(), set()

            query = f"SELECT panel_id FROM welcome_meta2 {where_clause}"
            
            logging.info(f"  (SQL) 실행: {query}")
            logging.info(f"  (SQL) 파라미터: {params}")

            cur.execute(query, tuple(params))
            
            results = {str(row[0]) for row in cur.fetchall()}
            cur.close()

            logging.info(f"  (SQL) 📈 1단계 필터링 결과: {len(results)}명")

        return results, set()

    except Exception as e:
        logging.error(f"   Welcome {attempt_name} 검색 실패: {e}", exc_info=True)
        return set(), set()


def search_preference_conditions(
    preference_keywords: List[str],
    query_vectors: List[List[float]],
    qdrant_client: QdrantClient,
    collection_name: str,
    candidate_panel_ids: Set[str],
    threshold: float = 0.45,
    top_k_per_keyword: int = 500
) -> Tuple[List[tuple], List[str]]:
    if not preference_keywords or not query_vectors or not candidate_panel_ids:
        return ([(pid, 0.0) for pid in candidate_panel_ids], [])
    try:
        candidate_list = list(candidate_panel_ids)
        qdrant_filter = Filter(must=[FieldCondition(key="panel_id", match=MatchAny(any=candidate_list))])
        panel_scores: Dict[str, float] = {pid: 0.0 for pid in candidate_panel_ids}
        found_categories: List[str] = []
        for i, (keyword, vector) in enumerate(zip(preference_keywords, query_vectors)):
            search_results = qdrant_client.search(
                collection_name=collection_name, query_vector=vector, query_filter=qdrant_filter,
                limit=top_k_per_keyword, with_payload=True, score_threshold=threshold
            )
            for result in search_results:
                pid = result.payload.get('panel_id')
                category = result.payload.get('category', None)
                if not pid and 'metadata' in result.payload:
                    pid = result.payload['metadata'].get('panel_id')
                    if not category: category = result.payload['metadata'].get('category', None)
                if pid and str(pid) in panel_scores:
                    panel_scores[str(pid)] = max(panel_scores[str(pid)], result.score)
                    if category: found_categories.append(category)
        sorted_results = sorted(panel_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results, list(set(found_categories))
    except Exception as e:
        logging.error(f"Preference 검색 실패: {e}")
        return ([(pid, 0.0) for pid in candidate_panel_ids], [])

def filter_negative_conditions(
    panel_ids: Set[str],
    negative_keywords: List[str],
    query_vectors: List[List[float]],
    qdrant_client: QdrantClient,
    collection_name: str,
    threshold: float = 0.50
) -> Set[str]:
    if not negative_keywords or not query_vectors or not panel_ids: return panel_ids
    try:
        panel_ids_to_exclude = set()
        for vector in query_vectors:
            search_results = qdrant_client.search(
                collection_name=collection_name, query_vector=vector, limit=5000,
                with_payload=True, score_threshold=threshold
            )
            for result in search_results:
                pid = result.payload.get('panel_id')
                if not pid and 'metadata' in result.payload: pid = result.payload['metadata'].get('panel_id')
                if pid: panel_ids_to_exclude.add(str(pid))
        return panel_ids - panel_ids_to_exclude
    except Exception as e:
        logging.error(f"Negative 필터링 실패: {e}")
        return panel_ids

@lru_cache(maxsize=None)
def initialize_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="nlpai-lab/KURE-v1", model_kwargs={'device': 'cpu'})
    except Exception as e:
        logging.error(f"임베딩 로드 실패: {e}")
        raise

def embed_keywords(keywords: List[str]) -> List[List[float]]:
    if not keywords: return []
    try:
        return initialize_embeddings().embed_documents(keywords)
    except Exception as e:
        logging.error(f"임베딩 실패: {e}")
        return []

def find_negative_answer_ids(
    candidate_ids: Set[str],
    target_field: str,
    collection_name: str,
    is_welcome_collection: bool = False,
    threshold: float = 0.82 # 유사도 0.82 이상이면 '부정 답변'으로 간주
) -> Set[str]:
    """
    후보군 중에서 '없다', '모르겠다' 등 부정적인 답변을 한 패널의 ID를 찾습니다.
    """
    from mapping_rules import NEGATIVE_ANSWER_KEYWORDS
    
    # 1. 해당 필드에 대한 부정 키워드가 없으면 스킵
    negative_keywords = NEGATIVE_ANSWER_KEYWORDS.get(target_field)
    if not negative_keywords or not candidate_ids:
        return set()

    try:
        client = QdrantClient(url=os.getenv("QDRANT_HOST"))
        embeddings = initialize_embeddings()
        
        # 2. 부정 키워드 벡터화 (예: "스트레스 없다"의 벡터)
        # 여러 키워드 중 하나라도 걸리면 되므로 각각 검색
        negative_vectors = embeddings.embed_documents(negative_keywords)
        
        ids_to_exclude = set()
        id_key_path = "metadata.panel_id" if is_welcome_collection else "panel_id"
        
        # 3. 후보군(candidate_ids) 내에서만 검색하도록 필터 설정
        search_filter = Filter(
            must=[
                FieldCondition(key=id_key_path, match=MatchAny(any=list(candidate_ids)))
            ]
        )
        
        # 4. 각 부정 키워드에 대해 검색 실행
        for neg_vec in negative_vectors:
            hits = client.search(
                collection_name=collection_name,
                query_vector=neg_vec,
                query_filter=search_filter,
                limit=len(candidate_ids), # 후보군 전체 검사
                score_threshold=threshold, # [중요] 이 점수보다 높으면 '부정 답변'으로 간주
                with_payload=[id_key_path]
            )
            
            for hit in hits:
                if is_welcome_collection:
                    pid = hit.payload.get('metadata', {}).get('panel_id')
                else:
                    pid = hit.payload.get('panel_id')
                
                if pid:
                    ids_to_exclude.add(pid)
        
        if ids_to_exclude:
            logging.info(f"   🚫 부정 답변 필터링: {len(ids_to_exclude)}명 제외됨 (키워드: {negative_keywords})")
            
        return ids_to_exclude

    except Exception as e:
        logging.error(f"부정 필터링 실패: {e}")
        return set()