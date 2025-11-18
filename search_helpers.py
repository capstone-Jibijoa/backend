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

# ============================================================
# 임베딩 모델 초기화 (싱글톤)
# ============================================================
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

# ============================================================
# Stage 1: PostgreSQL Objective 필터링 (기존 로직)
# ============================================================

class ConditionBuilder:
    """SQL 조건 빌더 (기존 로직 유지)"""
    def __init__(self):
        self.conditions = []
        self.params = []
        self.grouped_conditions = {}

    def add_condition(self, keyword: str, field: str):
        if field not in self.grouped_conditions:
            self.grouped_conditions[field] = []

        if field == 'gender':
            if keyword in ['남', '남자', '남성']: 
                self.grouped_conditions[field].append('M')
            elif keyword in ['여', '여자', '여성']: 
                self.grouped_conditions[field].append('F')
        
        elif field == 'birth_year':
            birth_start, birth_end = None, None
            if '~' in keyword:
                parts = keyword.replace('대', '').split('~')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    age_start, age_end = int(parts[0]), int(parts[1])
                    birth_start, birth_end = CURRENT_YEAR - age_end - 9, CURRENT_YEAR - age_start
            elif '이상' in keyword:
                match = re.search(r'(\d+)대\s*이상', keyword)
                if match:
                    age_start = int(match.group(1))
                    birth_start, birth_end = 0, CURRENT_YEAR - age_start
            elif keyword.endswith('대') and keyword[:-1].isdigit():
                age_prefix = int(keyword[:-1])
                birth_start, birth_end = CURRENT_YEAR - age_prefix - 9, CURRENT_YEAR - age_prefix
            
            if birth_start is not None:
                self.grouped_conditions[field].append((birth_start, birth_end))
        
        elif field in ['job_duty_raw', 'job_title_raw', 'car_model_raw']:
            self.grouped_conditions[field].append(f'%{keyword}%')
        
        else:
            self.grouped_conditions[field].append(keyword)

    def finalize(self) -> Tuple[str, List]:
        final_conditions = []
        final_params = []

        for field, values in self.grouped_conditions.items():
            if not values: 
                continue

            if field == 'birth_year':
                conds = []
                for start, end in values:
                    conds.append(f"(structured_data->>'birth_year' ~ '^[0-9]+$' AND (structured_data->>'birth_year')::int BETWEEN %s AND %s)")
                    final_params.extend([start, end])
                if conds: 
                    final_conditions.append(f"({' OR '.join(conds)})")
            
            elif field in ['job_duty_raw', 'job_title_raw', 'car_model_raw']:
                conds = [f"(structured_data->>'{field}' ILIKE %s)" for _ in values]
                final_params.extend(values)
                if conds: 
                    final_conditions.append(f"({' AND '.join(conds)})")

            else:
                placeholders = ','.join(['%s'] * len(values))
                final_conditions.append(f"(structured_data->>'{field}' IN ({placeholders}))")
                final_params.extend(values)

        if not final_conditions: 
            return "", []
        
        where_clause = " WHERE " + " AND ".join(final_conditions)
        return where_clause, final_params


def _map_keywords_to_fields(keywords: List[str]) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """키워드를 확장하고 필드에 매핑"""
    expanded_keywords_map = defaultdict(set)
    used_original_keywords = set()

    for original_kw in keywords:
        expanded_kws = CATEGORY_MAPPING.get(original_kw, [original_kw])
        
        for expanded_kw in expanded_kws:
            mapping = get_field_mapping(expanded_kw)
            
            if mapping and mapping.get('type') == 'filter' and mapping.get('field') != 'unknown':
                field = mapping['field']
                expanded_keywords_map[field].add(expanded_kw)
                used_original_keywords.add(original_kw)

    return expanded_keywords_map, used_original_keywords


def build_welcome_query_conditions(keywords: List[str]) -> Tuple[str, List, Set[str]]:
    """키워드 리스트를 분석하여 SQL WHERE 절 생성"""
    builder = ConditionBuilder()
    
    expanded_keywords_map, used_original_keywords = _map_keywords_to_fields(keywords)

    for field, kws in expanded_keywords_map.items():
        for kw in kws:
            builder.add_condition(kw, field)

    where_clause, params = builder.finalize()
    unhandled_keywords = set(keywords) - used_original_keywords
    return where_clause, params, unhandled_keywords


def search_welcome_objective(
    keywords: List[str],
    attempt_name: str = "객관식"
) -> Tuple[Set[str], Set[str]]:
    """
    Stage 1: PostgreSQL로 Objective (demographic) 필터링
    """
    if not keywords:
        logging.info(f"   Welcome {attempt_name}: 키워드 없음")
        return set(), set()
    
    try:
        with get_db_connection_context() as conn:
            if not conn:
                logging.error(f"   Welcome {attempt_name}: DB 연결 실패")
                return set(), set()
            
            cur = conn.cursor()
            where_clause, params, unhandled = build_welcome_query_conditions(keywords)
            
            if not where_clause:
                logging.info(f"   Welcome {attempt_name}: 조건 없음")
                cur.close()
                return set(), unhandled
            
            query = f"SELECT panel_id FROM welcome_meta2 {where_clause}"
            logging.info(f"   [SQL] {cur.mogrify(query, tuple(params)).decode('utf-8')}")
            
            cur.execute(query, tuple(params))
            results = {str(row[0]) for row in cur.fetchall()}
            cur.close()
        
        return results, unhandled
    
    except Exception as e:
        logging.error(f"   Welcome {attempt_name} 검색 실패: {e}", exc_info=True)
        return set(), set(keywords)


# ============================================================
# Stage 2: Must-have 엄격 검증
# ============================================================

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
    [v2 핵심 로직] Must-have 조건들을 AND 연산으로 엄격하게 검증
    
    전략:
    1. 각 must-have 키워드마다 개별 벡터 검색 수행 (높은 threshold)
    2. Pre-filtered panel_ids가 있으면 Qdrant filter로 범위 제한 (속도 향상)
    3. 모든 검색 결과의 교집합 반환 (AND 로직)
    
    Parameters:
    - must_have_keywords: 필수 조건 키워드 리스트
    - query_vectors: 각 키워드에 대응하는 임베딩 벡터 리스트
    - pre_filtered_panel_ids: PostgreSQL로 사전 필터링된 panel_id 집합
    - threshold: 유사도 임계값 (기본 0.55, 높을수록 정확)
    - hnsw_ef: 검색 정확도 파라미터 (기본 128)
    
    Returns:
    - 모든 must-have 조건을 만족하는 panel_id 집합
    """
    if not must_have_keywords or not query_vectors:
        logging.info("   Must-have: 조건 없음")
        return pre_filtered_panel_ids or set()
    
    if len(must_have_keywords) != len(query_vectors):
        logging.warning(f"   Must-have: 키워드({len(must_have_keywords)})와 벡터({len(query_vectors)}) 개수 불일치")
        return set()
    
    try:
        # Qdrant filter 생성 (pre-filtered panel_ids로 검색 범위 제한)
        qdrant_filter = None
        if pre_filtered_panel_ids is not None: # None이 아닌 빈 set()일 수도 있으므로 is not None으로 체크
            panel_ids_list = list(pre_filtered_panel_ids)
            if panel_ids_list:
                qdrant_filter = Filter(
                    must=[
                        # 'metadata.panel_id' 또는 'panel_id'에 따라 컬렉션 구조에 맞게 수정 필요
                        # 여기서는 두 경우 모두를 가정하지 않고, 일반적인 필드명으로 사용
                        FieldCondition(key="panel_id", match=MatchAny(any=panel_ids_list))
                    ]
                )
                logging.info(f"   ⚡ Must-have: {len(panel_ids_list):,}명 범위 내에서 검색 (속도 향상)")
            else:
                # 사전 필터링 결과가 0명이면, 더 이상 검색할 필요가 없음
                logging.info("   Must-have: 사전 필터링된 후보가 0명이므로 검색을 중단합니다.")
                return set()
        
        search_params = SearchParams(hnsw_ef=hnsw_ef)
        
        # 각 키워드별로 검색하여 교집합 계산
        result_sets = []
        for i, (keyword, vector) in enumerate(zip(must_have_keywords, query_vectors)):
            logging.info(f"   🔍 Must-have [{i+1}/{len(must_have_keywords)}]: '{keyword}' 검색 (threshold={threshold})")
            
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=vector,
                query_filter=qdrant_filter,
                limit=3000,  # Must-have는 제한적으로 검색
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
        
        # 모든 결과의 교집합 (AND 로직)
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
    threshold: float = 0.38,
    top_k_per_keyword: int = 500
) -> Tuple[List[tuple], List[str]]:
    """
    [v2 선호 조건] Preference 조건으로 후보를 스코어링하여 재순위화
    
    전략:
    1. Candidate panel_ids 중에서만 검색 (이미 objective + must-have 통과)
    2. 각 preference 키워드별 유사도 점수 집계
    3. 점수 높은 순으로 정렬하여 반환
    
    Returns:
    - [(panel_id, total_score), ...] 형태의 리스트 (점수 높은 순)
    """
    if not preference_keywords or not query_vectors or not candidate_panel_ids:
        logging.info("   Preference: 조건 없음 또는 후보 없음")
        return ([(pid, 0.0) for pid in candidate_panel_ids], [])
    
    try:
        # Candidate panel_ids로 filter 생성
        candidate_list = list(candidate_panel_ids)
        if len(candidate_list) > 5000:
            logging.warning(f"   ⚠️  Preference: 후보가 너무 많아({len(candidate_list):,}명) 상위 5000명만 검색")
            candidate_list = candidate_list[:5000]
        
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="panel_id",
                    match=MatchAny(any=candidate_list)
                )
            ]
        )
        
        # 각 panel_id별 점수 집계
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
                category = result.payload.get('category')
                if not pid and 'metadata' in result.payload:
                    pid = result.payload['metadata'].get('panel_id')
                    category = result.payload['metadata'].get('category')
                if pid and str(pid) in panel_scores:
                    panel_scores[str(pid)] += result.score
                    if category:
                        found_categories.append(category)
            
            logging.info(f"      → {len([s for s in search_results if s.score >= threshold])}명에게 점수 부여")
        
        # 점수 높은 순으로 정렬
        sorted_results = sorted(panel_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 점수가 0보다 큰 것만 반환 (preference 조건에 일부라도 부합하는 사람)
        filtered_results = [(pid, score) for pid, score in sorted_results if score > 0]
        
        logging.info(f"   ✅ Preference 스코어링 완료: {len(filtered_results):,}명 (0점 제외)")
        return filtered_results, found_categories
    
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
    [v2 부정 조건] Negative 조건을 만족하는 panel_id 제거
    
    전략:
    1. Negative 키워드에 유사도가 높은 panel_id 찾기
    2. 해당 panel_id를 결과에서 제거
    
    Returns:
    - Negative 조건을 제외한 panel_id 집합
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
        
        # Negative 조건 제거
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