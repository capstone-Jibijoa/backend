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
from mapping_rules import (
    CATEGORY_MAPPING, 
    get_field_mapping,
    FIELD_ALIAS_MAP, 
    VALUE_TRANSLATION_MAP,
    FUZZY_MATCH_FIELDS, 
    ARRAY_FIELDS
)

def build_sql_from_structured_filters(filters: List[Dict]) -> Tuple[str, List]:
    """
    JSONB 데이터 타입에 맞춰 정확한 SQL WHERE 절을 생성합니다.
    (소득 범위 스마트 매핑 포함)
    """
    if not filters:
        return "", []

    conditions = []
    params = []
    CURRENT_YEAR = datetime.now().year

    # 소득 카테고리 정의
    INCOME_RANGES = [
        (0, 999999, "월 100만원 미만"),
        (1000000, 1999999, "월 100~199만원"),
        (2000000, 2999999, "월 200~299만원"),
        (3000000, 3999999, "월 300~399만원"),
        (4000000, 4999999, "월 400~499만원"),
        (5000000, 5999999, "월 500~599만원"),
        (6000000, 6999999, "월 600~699만원"),
        (7000000, 999999999, "월 700만원 이상")
    ]

    for f in filters:
        raw_field = f.get("field")
        operator = f.get("operator")
        value = f.get("value")

        if not raw_field or not operator:
            continue

        field = FIELD_ALIAS_MAP.get(raw_field, raw_field)

        # 1. not_null 처리
        if operator == "not_null":
            base_condition = f"(structured_data->>'{field}' IS NOT NULL AND structured_data->>'{field}' != 'NaN')"
            exclude_pattern = ""
            
            if field == "children_count":
                conditions.append(f"({base_condition} AND structured_data->>'{field}' NOT IN ('0', '0명') AND structured_data->>'{field}' !~ '없음')")
                continue
            
            if field == "drinking_experience":
                exclude_pattern = "마시지|않음|없음|비음주|금주|안\\s*마심|전혀"
            elif field == "smoking_experience":
                exclude_pattern = "피우지|않음|없음|비흡연|금연|안\\s*피움"
            elif field == "ott_count":
                exclude_pattern = "0개|안\\s*함|없음|이용\\s*안|보지\\s*않음"
            elif field == "fast_delivery_usage":
                exclude_pattern = "안\\s*함|이용\\s*안|없음|직접\\s*구매"
            else:
                exclude_pattern = "없음|비흡연|해당사항|피우지|금연"

            refined_condition = f"({base_condition} AND structured_data->>'{field}' !~ '{exclude_pattern}')"
            conditions.append(refined_condition)
            continue 

        # 2. 나이 계산
        if field == "birth_year" or raw_field == "age":
            if operator == "between" and isinstance(value, list) and len(value) == 2:
                age_start, age_end = value
                birth_year_end = CURRENT_YEAR - age_start
                birth_year_start = CURRENT_YEAR - age_end
                conditions.append(f"(structured_data->>'birth_year')::int BETWEEN %s AND %s")
                params.extend([birth_year_start, birth_year_end])
            continue
        
        # 3. 값 확장 (매핑) - 영어/한글 변환 및 카테고리 확장
        final_value = value
        if field in VALUE_TRANSLATION_MAP:
            mapping = VALUE_TRANSLATION_MAP[field]
            if isinstance(value, list):
                converted_list = []
                for v in value:
                    mapped_v = mapping.get(v, v)
                    if isinstance(mapped_v, list):
                        converted_list.extend(mapped_v)
                    else:
                        converted_list.append(mapped_v)
                final_value = converted_list
            else:
                mapped_v = mapping.get(value, value)
                final_value = mapped_v

        if isinstance(final_value, list):
            expanded_list = []
            for v in final_value:
                if str(v) in CATEGORY_MAPPING:
                    expanded_list.extend(CATEGORY_MAPPING[str(v)])
                else:
                    expanded_list.append(v)
            final_value = expanded_list
        elif str(final_value) in CATEGORY_MAPPING:
            final_value = CATEGORY_MAPPING[str(final_value)]

        # 4. FUZZY_MATCH (ILIKE)
        if field in FUZZY_MATCH_FIELDS or field in ARRAY_FIELDS:
            if not isinstance(final_value, list):
                final_value = [final_value]
            
            or_conditions = []
            for v in final_value:
                or_conditions.append(f"structured_data->>'{field}' ILIKE %s")
                params.append(f"%{v}%")
            
            if or_conditions:
                exclude_sql = ""
                if field == "drinking_experience":
                    exclude_patterns = "마시지|않음|없음|비음주|금주|안\\s*마심|전혀"
                    exclude_sql = f" AND structured_data->>'{field}' !~ '{exclude_patterns}'"
                elif field == "smoking_experience":
                    exclude_patterns = "피우지|않음|없음|비흡연|금연|안\\s*피움"
                    exclude_sql = f" AND structured_data->>'{field}' !~ '{exclude_patterns}'"
                elif field == "ott_count":
                    exclude_pattern = "0개|안\\s*함|없음|이용\\s*안|보지\\s*않음"
                    conditions.append(f"({base_condition} AND structured_data->>'{field}' !~ '{exclude_pattern}')")

                conditions.append(f"({' OR '.join(or_conditions)}){exclude_sql}")

        # 5. 소득 필드 스마트 처리 (숫자 범위 -> 문자열 카테고리 변환)
        elif field in ["income_household_monthly", "income_personal_monthly"] and operator in ["gte", "lte", "between"]:
            target_categories = []
            min_val = 0
            max_val = 999999999
            
            if operator == "gte": min_val = int(final_value)
            elif operator == "lte": max_val = int(final_value)
            elif operator == "between":
                min_val = int(final_value[0])
                max_val = int(final_value[1])
            
            for r_min, r_max, label in INCOME_RANGES:
                if max_val >= r_min and min_val <= r_max:
                    target_categories.append(label)
            
            if target_categories:
                str_values = [str(v) for v in target_categories]
                placeholders = ','.join(['%s'] * len(str_values))
                conditions.append(f"structured_data->>'{field}' IN ({placeholders})")
                params.extend(str_values)
            else:
                conditions.append("1=0")

        # 6. 숫자형 필드 (children_count 등)
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

        # 7. 일반 문자열 필드
        else:
            field_sql = f"structured_data->>'{field}'"

            if field == "family_size":
                if isinstance(final_value, list):
                    or_conditions = []
                    for v in final_value:
                        or_conditions.append(f"{field_sql} ~ %s")
                        params.append(f"^{v}([^0-9]|$)") 
                    conditions.append(f"({' OR '.join(or_conditions)})")
                else:
                    conditions.append(f"{field_sql} ~ %s")
                    params.append(f"^{final_value}([^0-9]|$)")

            elif operator == "eq":
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