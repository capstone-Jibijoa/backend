import os
import re
import logging
import threading
from typing import List, Set, Optional, Dict, Tuple
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, SearchParams
from langchain_huggingface import HuggingFaceEmbeddings

from db import get_db_connection_context
from mapping_rules import CATEGORY_MAPPING, get_field_mapping

load_dotenv()

EMBEDDINGS = None
embedding_lock = threading.Lock()
CURRENT_YEAR = datetime.now().year

def initialize_embeddings():
    """임베딩 모델 초기화 (싱글톤 패턴)"""
    global EMBEDDINGS
    if EMBEDDINGS is None:
        with embedding_lock:
            if EMBEDDINGS is None:
                logging.info("⏳ (최초 1회) 임베딩 모델 초기화 중...")
                EMBEDDINGS = HuggingFaceEmbeddings(
                    model_name="nlpai-lab/KURE-v1",
                    model_kwargs={'device': 'cpu'}
                )
    return EMBEDDINGS

def build_sql_from_structured_filters(filters: List[Dict]) -> Tuple[str, List]:
    """
    LLM이 생성한 structured_filters를 기반으로 SQL WHERE 절과 파라미터를 동적으로 생성합니다.
    'age' 필드를 'birth_year' 기반의 실시간 나이 계산으로 변환하는 로직을 포함합니다.
    """
    if not filters:
        return "", []

    conditions = []
    params = []
    CURRENT_YEAR = datetime.now().year

    for f in filters:
        field = f.get("field")
        operator = f.get("operator")
        value = f.get("value")

        if not field or not operator:
            continue

        # 'age' 필드를 'birth_year' 계산으로 특별 처리
        if field == "age":
            # 나이 필드는 연산자가 'between'일 것으로 가정
            if operator == "between" and isinstance(value, list) and len(value) == 2:
                age_start, age_end = value
                # 나이를 출생년도 범위로 변환
                birth_year_end = CURRENT_YEAR - age_start
                birth_year_start = CURRENT_YEAR - age_end
                
                conditions.append(f"(structured_data->>'birth_year')::int BETWEEN %s AND %s")
                params.extend([birth_year_start, birth_year_end])
            continue

        # 다른 일반 필드 처리
        field_sql = f"(structured_data->>'{field}')"
        # 숫자형일 수 있는 다른 필드들 처리
        if field in ['income_personal_monthly', 'family_size', 'children_count']:
             field_sql = f"({field_sql}::numeric)"

        if operator == "eq":
            conditions.append(f"{field_sql} = %s")
            params.append(value)
        elif operator == "in" and isinstance(value, list) and value:
            placeholders = ','.join(['%s'] * len(value))
            conditions.append(f"{field_sql} IN ({placeholders})")
            params.extend(value)
        elif operator == "between" and isinstance(value, list) and len(value) == 2:
            conditions.append(f"{field_sql} BETWEEN %s AND %s")
            params.extend(value)
        elif operator == "like":
            conditions.append(f"{field_sql} ILIKE %s")
            params.append(f"%{value}%")
        elif operator == "gte":
            conditions.append(f"{field_sql} >= %s")
            params.append(value)
        elif operator == "lte":
            conditions.append(f"{field_sql} <= %s")
            params.append(value)

    if not conditions:
        return "", []

    where_clause = " WHERE " + " AND ".join(conditions)
    return where_clause, params


def search_welcome_objective(
    filters: List[Dict],
    attempt_name: str = "구조화"
) -> Tuple[Set[str], Set[str]]:
    """
    Stage 1: LLM이 생성한 구조화된 필터(structured_filters)를 사용하여 PostgreSQL에서 필터링합니다.
    """
    if not filters:
        logging.info(f"   Welcome {attempt_name}: 필터 없음")
        return set(), set()

    try:
        with get_db_connection_context() as conn:
            if not conn:
                logging.error(f"   Welcome {attempt_name}: DB 연결 실패")
                return set(), set()

            cur = conn.cursor()
            where_clause, params = build_sql_from_structured_filters(filters)

            if not where_clause:
                logging.info(f"   Welcome {attempt_name}: 유효한 SQL 조건 없음")
                cur.close()
                return set(), set()

            query = f"SELECT panel_id FROM welcome_meta2 {where_clause}"
            
            cur.execute(query, tuple(params))
            results = {str(row[0]) for row in cur.fetchall()}
            cur.close()

        return results, set() # unhandled_keywords는 더 이상 사용하지 않으므로 빈 set 반환

    except Exception as e:
        logging.error(f"   Welcome {attempt_name} 검색 실패: {e}", exc_info=True)
        return set(), set()


def search_must_have_conditions(
    must_have_keywords: List[str],
    query_vectors: List[List[float]],
    qdrant_client: QdrantClient,
    collection_name: str,
    pre_filtered_panel_ids: Optional[Set[str]] = None,
    threshold: float = 0.55,
    hnsw_ef: int = 128
) -> Set[str]:
    """
    Must-have 조건들을 AND 연산으로 엄격하게 검증합니다.
    
    전략:
    1. 각 must-have 키워드마다 개별 벡터 검색 수행 (높은 threshold)
    2. Pre-filtered panel_ids가 있으면 Qdrant filter로 범위 제한 (속도 향상)
    3. 모든 검색 결과의 교집합 반환 (AND 로직)
    """
    if not must_have_keywords or not query_vectors:
        logging.info("   Must-have: 조건 없음")
        return pre_filtered_panel_ids or set()
    
    if len(must_have_keywords) != len(query_vectors):
        logging.warning(f"   Must-have: 키워드({len(must_have_keywords)})와 벡터({len(query_vectors)}) 개수 불일치")
        return set()
    
    try:
        qdrant_filter = None
        if pre_filtered_panel_ids is not None:
            panel_ids_list = list(pre_filtered_panel_ids)
            if panel_ids_list:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(key="panel_id", match=MatchAny(any=panel_ids_list))
                    ]
                )
                logging.info(f"   ⚡ Must-have: {len(panel_ids_list):,}명 범위 내에서 검색 (속도 향상)")
            else:
                logging.info("   Must-have: 사전 필터링된 후보가 0명이므로 검색을 중단합니다.")
                return set()
        
        search_params = SearchParams(hnsw_ef=hnsw_ef)
        
        result_sets = []
        for i, (keyword, vector) in enumerate(zip(must_have_keywords, query_vectors)):
            logging.info(f"   🔍 Must-have [{i+1}/{len(must_have_keywords)}]: '{keyword}' 검색 (threshold={threshold})")
            
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=vector,
                query_filter=qdrant_filter,
                limit=3000,
                with_payload=True,
                score_threshold=threshold,
                search_params=search_params
            )
            
            panel_ids = set()
            for result in search_results:
                pid = result.payload.get('panel_id')
                if not pid and 'metadata' in result.payload:
                    pid = result.payload['metadata'].get('panel_id')
                if pid:
                    panel_ids.add(str(pid))
            
            logging.info(f"      → {len(panel_ids):,}명 검색됨 (유사도 {threshold}+ 조건 만족)")
            result_sets.append(panel_ids)
        
        if result_sets:
            final_result = result_sets[0]
            for result_set in result_sets[1:]:
                final_result &= result_set
            
            logging.info(f"   ✅ Must-have 교집합 결과: {len(final_result):,}명 (모든 조건 만족)")
            return final_result
        
        return set()
    
    except Exception as e:
        logging.error(f"   ❌ Must-have 검색 실패: {e}", exc_info=True)
        return set()


def search_preference_conditions(
    preference_keywords: List[str],
    query_vectors: List[List[float]],
    qdrant_client: QdrantClient,
    collection_name: str,
    candidate_panel_ids: Set[str],
    threshold: float = 0.45,
    top_k_per_keyword: int = 500
) -> Tuple[List[tuple], List[str]]:
    """
    Preference 조건으로 후보를 스코어링하여 재순위화합니다.
    
    전략:
    1. Candidate panel_ids 중에서만 검색 (이미 objective + must-have 통과)
    2. 각 preference 키워드별 유사도 점수 집계
    """
    if not preference_keywords or not query_vectors or not candidate_panel_ids:
        logging.info("   Preference: 조건 없음 또는 후보 없음")
        return ([(pid, 0.0) for pid in candidate_panel_ids], [])
    
    try:
        candidate_list = list(candidate_panel_ids)
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="panel_id",
                    match=MatchAny(any=candidate_list)
                )
            ]
        )
        
        panel_scores: Dict[str, float] = {pid: 0.0 for pid in candidate_panel_ids}
        found_categories: List[str] = []
        
        for i, (keyword, vector) in enumerate(zip(preference_keywords, query_vectors)):
            logging.info(f"   📊 Preference [{i+1}/{len(preference_keywords)}]: '{keyword}' 스코어링 (threshold={threshold})")
            
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=vector,
                query_filter=qdrant_filter,
                limit=top_k_per_keyword,
                with_payload=True,
                score_threshold=threshold
            )
            
            for result in search_results:
                pid = result.payload.get('panel_id')
                category = result.payload.get('category', None)
                if not pid and 'metadata' in result.payload:
                    pid = result.payload['metadata'].get('panel_id')
                    if not category:
                        category = result.payload['metadata'].get('category', None)

                if pid and str(pid) in panel_scores:
                    panel_scores[str(pid)] = max(panel_scores[str(pid)], result.score)
                    if category:
                        found_categories.append(category)
            
            logging.info(f"      → {len([s for s in search_results if s.score >= threshold])}명에게 점수 부여")
        
        sorted_results = sorted(panel_scores.items(), key=lambda x: x[1], reverse=True)
        
        logging.info(f"   ✅ Preference 스코어링 완료: {len(sorted_results):,}명")
        return sorted_results, list(set(found_categories))
    
    except Exception as e:
        logging.error(f"   ❌ Preference 스코어링 실패: {e}", exc_info=True)
        return ([(pid, 0.0) for pid in candidate_panel_ids], [])


def filter_negative_conditions(
    panel_ids: Set[str],
    negative_keywords: List[str],
    query_vectors: List[List[float]],
    qdrant_client: QdrantClient,
    collection_name: str,
    threshold: float = 0.50
) -> Set[str]:
    """
    Negative 조건을 만족하는 panel_id를 제거합니다.
    
    전략:
    1. Negative 키워드에 유사도가 높은 panel_id 찾기
    2. 해당 panel_id를 결과에서 제거
    """
    if not negative_keywords or not query_vectors or not panel_ids:
        return panel_ids
    
    try:
        panel_ids_to_exclude = set()
        
        for i, (keyword, vector) in enumerate(zip(negative_keywords, query_vectors)):
            logging.info(f"   🚫 Negative [{i+1}/{len(negative_keywords)}]: '{keyword}' 제외 대상 검색 (threshold={threshold})")
            
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=5000,
                with_payload=True,
                score_threshold=threshold
            )
            
            for result in search_results:
                pid = result.payload.get('panel_id')
                if not pid and 'metadata' in result.payload:
                    pid = result.payload['metadata'].get('panel_id')
                if pid:
                    panel_ids_to_exclude.add(str(pid))
            
            logging.info(f"      → {len(panel_ids_to_exclude):,}명 제외 대상 추가")
        
        result = panel_ids - panel_ids_to_exclude
        logging.info(f"   ✅ Negative 필터링 완료: {len(panel_ids_to_exclude):,}명 제외, {len(result):,}명 남음")
        
        return result
    
    except Exception as e:
        logging.error(f"   ❌ Negative 필터링 실패: {e}", exc_info=True)
        return panel_ids


def embed_keywords(keywords: List[str], embeddings_model) -> List[List[float]]:
    """
    키워드 리스트를 임베딩 벡터 리스트로 변환
    """
    if not keywords:
        return []
    
    try:
        vectors = []
        for keyword in keywords:
            vector = embeddings_model.embed_query(keyword)
            vectors.append(vector)
        return vectors
    except Exception as e:
        logging.error(f"❌ 임베딩 실패: {e}", exc_info=True)
        return []