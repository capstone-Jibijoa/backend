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

# LLM 필드명 -> 실제 DB 필드명 매핑
FIELD_ALIAS_MAP = {
    "household_size": "family_size",  
    "age": "birth_year",              
    "job": "job_title_raw",           
    "region": "region_major"          
}

# 실제 데이터(명사형)를 포함하도록 매핑 키워드 확장
VALUE_TRANSLATION_MAP = {
    'gender': {
        '남성': 'M', '여성': 'F', '남자': 'M', '여자': 'F', 
        'male': 'M', 'female': 'F' # 영어 추가
    },
    'marital_status': {
        '미혼': '미혼', '싱글': '미혼', '기혼': '기혼', '결혼': '기혼', '이혼': '이혼', '돌싱': '이혼',
        'single': '미혼', 'married': '기혼' # 영어 추가
    },
    'education_level': {
        '고졸': ['고등학교 졸업 이하', '고등학교 졸업', '고졸'],
        '고등학교 졸업': ['고등학교 졸업 이하', '고등학교 졸업'],

        '중졸 이하': ['고등학교 졸업 이하', '중학교 졸업 이하', '중학교 졸업', '초등학교 졸업 이하', '무학'],
        '중학교 졸업': ['고등학교 졸업 이하', '중학교 졸업 이하', '중학교 졸업'],

        '대졸': ['대학교 졸업', '대학원 재학 이상', '학사', '석사', '박사'],
        '대학교 졸업': ['대학교 졸업', '대학원 재학 이상'],
        '대학원': ['대학원 재학 이상'],

        '저학력': ['고등학교 졸업 이하', '중학교 졸업 이하', '고등학교 졸업', '중학교 졸업', '초등학교 졸업 이하'],
        '고학력': ['대학교 졸업', '대학원 재학 이상']
    },
    'smoking_experience': {
        'have_smoked': ['일반', '전자', '기타', '피우고', '피웠', '흡연', '연초', '궐련'],
        'smoker': ['일반', '전자', '기타', '피우고', '피웠', '흡연', '연초', '궐련'],
        
        '있음': ['일반', '전자', '기타', '피우고', '피웠', '흡연', '연초', '궐련'], 
        '흡연': ['일반', '전자', '기타', '피우고', '피웠', '흡연', '연초', '궐련'],
        
        'no': ['피워본 적이', '비흡연'],
        'none': ['피워본 적이', '비흡연'],
        'non_smoker': ['피워본 적이', '비흡연'],
        '없음': ['피워본 적이', '비흡연'],
        '비흡연': ['피워본 적이', '비흡연'],
    },
    'drinking_experience': {
        'have_drink': ['소주', '맥주', '와인', '막걸리', '위스키', '양주', '사케', '과일칵테일주', '저도주', '즐겨', '마신다'], # 영어 추가
        'drinker': ['소주', '맥주', '와인', '막걸리', '위스키', '양주', '사케', '과일칵테일주', '저도주', '즐겨', '마신다'],
        '있음': ['소주', '맥주', '와인', '막걸리', '위스키', '양주', '사케', '과일칵테일주', '저도주', '즐겨', '마신다'],
        
        'no': ['최근 1년 이내 술을 마시지 않음', '마시지', '비음주', '금주', '않음'], # 영어 추가
        '없음': ['최근 1년 이내 술을 마시지 않음', '마시지', '비음주', '금주', '않음'],
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

        field = FIELD_ALIAS_MAP.get(raw_field, raw_field)

        if operator == "not_null":
            # JSONB 필드 내에 해당 키가 존재하고, null이 아닌지 확인
            base_condition = f"(structured_data->>'{field}' IS NOT NULL AND structured_data->>'{field}' != 'NaN')"
            
            # 2. '없음', '해당 없음', '비흡연' 등 무의미한 텍스트도 제외 (SQL 레벨)
            # 정규식으로 '없음', '비흡연', '피우지 않음' 등이 포함된 경우 제외
            exclude_pattern = "없음|비흡연|해당사항|피우지|금연"
            refined_condition = f"({base_condition} AND structured_data->>'{field}' !~ '{exclude_pattern}')"
            
            conditions.append(refined_condition)
            continue 

        # --- 나이 계산 ---
        if field == "birth_year" or raw_field == "age":
            if operator == "between" and isinstance(value, list) and len(value) == 2:
                age_start, age_end = value
                birth_year_end = CURRENT_YEAR - age_start
                birth_year_start = CURRENT_YEAR - age_end
                conditions.append(f"(structured_data->>'birth_year')::int BETWEEN %s AND %s")
                params.extend([birth_year_start, birth_year_end])
            continue
        
        # --- 매핑된 값으로 변환 ---
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

        # --- JSON 배열(List) 필드 처리 ---
        # ILIKE를 사용하여 부분 문자열 검색 (배열 -> 문자열 변환 후 검색)
        if field in ARRAY_FIELDS:
            if not isinstance(final_value, list):
                final_value = [final_value]
            
            or_conditions = []
            for v in final_value:
                or_conditions.append(f"structured_data->>'{field}' ILIKE %s")
                params.append(f"%{v}%")
            
            if or_conditions:
                # 긍정 키워드('술')가 있더라도 부정어('않음')가 있으면 제외하는 로직
                exclude_sql = ""
                
                # 1. 음주 경험 (drinking_experience)
                if field == "drinking_experience":
                    # '마시지', '않음', '없음', '비음주', '금주', '안 마심' 등이 포함되면 제외
                    exclude_patterns = "마시지|않음|없음|비음주|금주|안\\s*마심|전혀"
                    exclude_sql = f" AND structured_data->>'{field}' !~ '{exclude_patterns}'"
                
                # 2. 흡연 경험 (smoking_experience) - 혹시 모를 비흡연자 데이터 방지
                elif field == "smoking_experience":
                    exclude_patterns = "피우지|않음|없음|비흡연|금연|안\\s*피움"
                    exclude_sql = f" AND structured_data->>'{field}' !~ '{exclude_patterns}'"

                elif field == "ott_count":
                    # '0개', '안 함', '없음', '이용 안' 등이 포함되면 제외
                    exclude_pattern = "0개|안\\s*함|없음|이용\\s*안|보지\\s*않음"
                    conditions.append(f"({base_condition} AND structured_data->>'{field}' !~ '{exclude_pattern}')")

                # 기존 OR 조건 뒤에 AND 제외 조건 붙이기
                conditions.append(f"({' OR '.join(or_conditions)}){exclude_sql}")

        # --- 숫자형 필드 처리 ---
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

        # --- 일반 문자열 필드 처리 ---
        else:
            field_sql = f"structured_data->>'{field}'"

            if field == "family_size":
                # 리스트인 경우 (예: 2인 가구 OR 3인 가구)
                if isinstance(final_value, list):
                    or_conditions = []
                    for v in final_value:
                        or_conditions.append(f"{field_sql} ~ %s")
                        params.append(f"^{v}([^0-9]|$)") 
                    conditions.append(f"({' OR '.join(or_conditions)})")
                # 단일 값인 경우
                else:
                    conditions.append(f"{field_sql} ~ %s")
                    params.append(f"^{final_value}([^0-9]|$)")

            elif operator == "eq":
                if field in ["job_title_raw", "job_duty_raw"]:
                     conditions.append(f"{field_sql} ILIKE %s")
                     params.append(f"%{final_value}%")
                else:
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

def find_negative_answer_ids(
    candidate_ids: Set[str],
    target_field: str,
    collection_name: str,
    is_welcome_collection: bool = False,
    threshold: float = 0.82 
) -> Set[str]:
    from mapping_rules import NEGATIVE_ANSWER_KEYWORDS
    
    negative_keywords = NEGATIVE_ANSWER_KEYWORDS.get(target_field)
    if not negative_keywords or not candidate_ids:
        return set()

    try:
        client = QdrantClient(url=os.getenv("QDRANT_HOST"))
        embeddings = initialize_embeddings()
        
        negative_vectors = embeddings.embed_documents(negative_keywords)
        
        ids_to_exclude = set()
        id_key_path = "metadata.panel_id" if is_welcome_collection else "panel_id"
        
        search_filter = Filter(
            must=[
                FieldCondition(key=id_key_path, match=MatchAny(any=list(candidate_ids)))
            ]
        )
        
        for neg_vec in negative_vectors:
            hits = client.search(
                collection_name=collection_name,
                query_vector=neg_vec,
                query_filter=search_filter,
                limit=len(candidate_ids), 
                score_threshold=threshold, 
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