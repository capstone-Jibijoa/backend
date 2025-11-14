import os
import re
import logging
from typing import Optional, Tuple, List, Set
from datetime import datetime
import threading
from dotenv import load_dotenv
from db import get_db_connection_context, get_qdrant_client, get_db_connection
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EMBEDDINGS = None
embedding_lock = threading.Lock()
CURRENT_YEAR = datetime.now().year

CATEGORY_MAPPING = {
    '직장인': ['사무직', '전문직', '경영관리직', '생산노무직', '서비스직', '판매직', '기술직'],
    '고소득': ['월 500~599만원', '월 600~699만원', '월 700만원 이상'],
    '저소득': ['월 100~199만원', '월 200~299만원', '월 100만원 미만'],
    '중산층': ['월 300~399만원', '월 400~499만원'],
    '고학력': ['대학교 졸업', '대학원 재학 이상'],
    '저학력': ['고등학교 졸업 이하', '중학교 졸업 이하'],
    '젊은층': ['20대', '30대'],
    '청년': ['20대', '30대'],
    'MZ세대': ['20대', '30대'],
    '중장년층': ['40대', '50대'],
    'X세대': ['40대', '50대'],
    '장년층': ['50대', '60대'],
    '베이비부머': ['50대', '60대 이상'],
    '노년층': ['60대 이상'],
    '청소년': ['10대'],
}

VALID_REGIONS = [
    '서울', '경기', '인천', '부산', '대구', '대전', '광주', '울산', '세종',
    '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'
]


def expand_keywords(keywords: List[str]) -> List[str]:
    """추상 키워드를 구체 값으로 확장"""
    expanded = []
    for keyword in keywords:
        if keyword in CATEGORY_MAPPING:
            expanded.extend(CATEGORY_MAPPING[keyword])
        else:
            expanded.append(keyword)
    return expanded


def initialize_embeddings():
    """KURE 임베딩 모델 초기화 (스레드 안전)"""
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


def extract_panel_id_from_payload(payload: dict) -> Optional[str]:
    """Qdrant 페이로드에서 panel_id 추출 (Welcome용)"""
    try:
        if 'metadata' in payload and isinstance(payload['metadata'], dict):
            panel_id = payload['metadata'].get('panel_id')
            if panel_id:
                return str(panel_id)
        
        panel_id = payload.get('panel_id')
        if panel_id:
            return str(panel_id)
        
        return None
    except Exception:
        return None


class ConditionBuilder:
    """SQL 조건 빌더"""
    
    def __init__(self):
        self.conditions = []
        self.params = []
        self.regions = []
        self.jobs = []
        self.incomes = []
        self.educations = []
        self.age_ranges = []
    
    def add_gender(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['남자', '남성', '남']:
            self.conditions.append("(structured_data->>'gender' = %s)")
            self.params.append('M')
        elif kw in ['여자', '여성', '여']:
            self.conditions.append("(structured_data->>'gender' = %s)")
            self.params.append('F')
    
    def add_region(self, keyword: str):
        if keyword in VALID_REGIONS:
            self.regions.append(keyword)

    def add_age_range(self, keyword: str):
        if '대' not in keyword:
            return
        
        birth_start = None
        birth_end = None

        if '~' in keyword:
            age_range = keyword.replace('대', '').split('~')
            if len(age_range) == 2 and age_range[0].isdigit() and age_range[1].isdigit():
                age_start = int(age_range[0])
                age_end = int(age_range[1])
                birth_start = CURRENT_YEAR - age_end - 9
                birth_end = CURRENT_YEAR - age_start
        
        elif keyword[:-1].isdigit():
            age_prefix = int(keyword[:-1])
            birth_start = CURRENT_YEAR - age_prefix - 9
            birth_end = CURRENT_YEAR - age_prefix
        
        if birth_start is not None and birth_end is not None:
            self.age_ranges.append((birth_start, birth_end))
    
    def add_job(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['사무직', '전문직', '경영관리직', '생산노무직', '서비스직', '판매직', '기술직']:
            self.jobs.append(keyword)
    
    def add_income(self, keyword: str):
        if '월' in keyword and '만원' in keyword:
            self.incomes.append(keyword)
    
    def add_education(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['대학교 졸업', '대학원 재학 이상', '고등학교 졸업 이하', '중학교 졸업 이하']:
            self.educations.append(keyword)
    
    def add_marital_status(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['미혼', '싱글']:
            self.conditions.append("(structured_data->>'marital_status' = %s)")
            self.params.append('미혼')
        elif kw in ['기혼', '결혼']:
            self.conditions.append("(structured_data->>'marital_status' = %s)")
            self.params.append('기혼')
        elif kw in ['이혼', '돌싱', '사별']:
            self.conditions.append("(structured_data->>'marital_status' LIKE %s)")
            self.params.append('%기타%')
    
    def add_drinking(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['술먹는', '음주', '술', '맥주', '소주', '와인']:
            self.conditions.append(
                "(jsonb_array_length(COALESCE(structured_data->'drinking_experience', '[]'::jsonb)) > 0)"
            )
        elif kw in ['술안먹는', '금주']:
            self.conditions.append(
                "(jsonb_array_length(COALESCE(structured_data->'drinking_experience', '[]'::jsonb)) = 0)"
            )
    
    def add_smoking(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['흡연', '담배']:
            self.conditions.append(
                "(jsonb_array_length(COALESCE(structured_data->'smoking_experience', '[]'::jsonb)) > 0)"
            )
        elif kw in ['비흡연', '금연']:
            self.conditions.append(
                "(jsonb_array_length(COALESCE(structured_data->'smoking_experience', '[]'::jsonb)) = 0)"
            )
    
    def add_car_ownership(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['차있음', '자가용', '차량보유']:
            self.conditions.append("(structured_data->>'car_ownership' = %s)")
            self.params.append('있다')
        elif kw in ['차없음']:
            self.conditions.append("(structured_data->>'car_ownership' = %s)")
            self.params.append('없다')
    
    def add_family_size(self, keyword: str):
        if '가족' not in keyword or not any(char.isdigit() for char in keyword):
            return
        
        num_match = re.search(r'(\d+)', keyword)
        if not num_match:
            return
        
        num = int(num_match.group(1))
        
        if '이상' in keyword:
            self.conditions.append(
                "(structured_data->>'family_size' ~ '[0-9]' "
                "AND CAST(substring(structured_data->>'family_size' from '[0-9]+') AS int) >= %s)"
            )
            self.params.append(num)
        elif '이하' in keyword:
            self.conditions.append(
                "(structured_data->>'family_size' ~ '[0-9]' "
                "AND CAST(substring(structured_data->>'family_size' from '[0-9]+') AS int) <= %s)"
            )
            self.params.append(num)
        else:
            self.conditions.append(
                "(structured_data->>'family_size' ~ '[0-9]' "
                "AND CAST(substring(structured_data->>'family_size' from '[0-9]+') AS int) = %s)"
            )
            self.params.append(num)
    
    def finalize(self) -> Tuple[str, List]:
        """최종 WHERE 절 생성"""
        if self.jobs:
            job_conditions = ["(structured_data->>'job_title_raw' ILIKE %s)" for _ in self.jobs]
            self.conditions.append(f"({' OR '.join(job_conditions)})")
            self.params.extend([f'%{job}%' for job in self.jobs])
        
        if self.incomes:
            income_conditions = ["(structured_data->>'income_personal_monthly' = %s)" for _ in self.incomes]
            self.conditions.append(f"({' OR '.join(income_conditions)})")
            self.params.extend(self.incomes)
        
        if self.educations:
            edu_conditions = ["(structured_data->>'education_level' = %s)" for _ in self.educations]
            self.conditions.append(f"({' OR '.join(edu_conditions)})")
            self.params.extend(self.educations)
        
        if len(self.regions) == 1:
            self.conditions.append("(structured_data->>'region_major' = %s)")
            self.params.append(self.regions[0])
        elif len(self.regions) > 1:
            placeholders = ','.join(['%s'] * len(self.regions))
            self.conditions.append(f"(structured_data->>'region_major' IN ({placeholders}))")
            self.params.extend(self.regions)
        
        if self.age_ranges:
            age_conditions = []
            for start, end in self.age_ranges:
                age_conditions.append(
                    "(structured_data->>'birth_year' ~ '^[0-9]+$' "
                    "AND (structured_data->>'birth_year')::int BETWEEN %s AND %s)"
                )
                self.params.extend([start, end])
            self.conditions.append(f"({' OR '.join(age_conditions)})")

        if not self.conditions:
            return "", []
        
        where_clause = " WHERE " + " AND ".join(self.conditions)
        return where_clause, self.params


def build_welcome_query_conditions(keywords: List[str]) -> Tuple[str, List]:
    """Welcome 쿼리 조건 빌더"""
    keywords = expand_keywords(keywords)
    builder = ConditionBuilder()
    
    for keyword in keywords:
        builder.add_gender(keyword)
        builder.add_region(keyword)
        builder.add_age_range(keyword)
        builder.add_job(keyword)
        builder.add_income(keyword)
        builder.add_education(keyword)
        builder.add_marital_status(keyword)
        builder.add_drinking(keyword)
        builder.add_smoking(keyword)
        builder.add_car_ownership(keyword)
        builder.add_family_size(keyword)
    
    return builder.finalize()


def search_welcome_objective(keywords: List[str]) -> Set[str]:
    """Welcome 객관식 PostgreSQL 검색 (Connection Pool 사용)"""
    if not keywords:
        logging.info("   Welcome 객관식: 키워드 없음")
        return set()
    
    try:
        # Connection Pool 사용
        with get_db_connection_context() as conn:
            if not conn:
                logging.error("   Welcome 객관식: DB 연결 실패")
                return set()
            
            cur = conn.cursor()
            where_clause, params = build_welcome_query_conditions(keywords)
            
            if not where_clause:
                logging.info("   Welcome 객관식: 조건 없음")
                cur.close()
                return set()
            
            query = f"SELECT panel_id FROM welcome_meta2 {where_clause}"
            cur.execute(query, tuple(params))
            results = {str(row[0]) for row in cur.fetchall()}
            cur.close()
        
        logging.info(f"   ✅ Welcome 객관식: {len(results):,}명")
        return results
    except Exception as e:
        logging.error(f"   Welcome 객관식 검색 실패: {e}", exc_info=True)
        return set()


def _perform_vector_search(
    search_type_name: str,
    collection_name: str,
    query_vector: List[float],
    panel_id_key: str,
    score_threshold: float,
    pre_filter_panel_ids: Optional[Set[str]] = None,
    flatten_keywords_flag: bool = False,
) -> Set[str]:
    """
    [리팩토링] Qdrant 벡터 검색을 수행하는 공통 헬퍼 함수. (search_batch 적용)
    Welcome 주관식, QPoll 검색에서 모두 사용됩니다.
    """
    if not query_vector:
        logging.info(f"   {search_type_name}: 키워드 없음")
        return set()

    try:
        qdrant_client = get_qdrant_client()
        if not qdrant_client:
            logging.error(f"   {search_type_name}: Qdrant 연결 실패")
            return set()

        should_filter = pre_filter_panel_ids is not None and len(pre_filter_panel_ids) > 0
        if should_filter and len(pre_filter_panel_ids) > 50000:
            logging.warning(f"   {search_type_name}: 필터 ID가 {len(pre_filter_panel_ids):,}개로 너무 많아 필터링을 건너뜁니다.")
            should_filter = False

        qdrant_filter = None
        if should_filter:
            qdrant_filter = Filter(must=[FieldCondition(key=panel_id_key, match=MatchAny(any=list(pre_filter_panel_ids)))])

        search_results = qdrant_client.search(
            collection_name=collection_name, 
            query_vector=query_vector,
            query_filter=qdrant_filter, # None 또는 전체 필터
            limit=len(pre_filter_panel_ids) if should_filter else 1000, # 필터가 있으면 ID 수만큼, 없으면 1000개
            with_payload=True, 
            score_threshold=score_threshold
        )

        panel_ids = set()
        for result in search_results:
            pid = result.payload.get('panel_id') if 'panel_id' in result.payload else extract_panel_id_from_payload(result.payload)
            if pid:
                panel_ids.add(str(pid))

        logging.info(f"   ✅ {search_type_name}: {len(panel_ids):,}명")
        return panel_ids

    except Exception as e:
        logging.error(f"   {search_type_name} 검색 실패: {e}", exc_info=True)
        return set()

def search_welcome_subjective(query_vector: List[float], pre_filter_panel_ids: Optional[Set[str]] = None, keywords: List[str] = None) -> Set[str]:
    """Welcome 주관식 Qdrant 검색"""
    if not query_vector:
        logging.info("   Welcome 주관식: 벡터 없음")
        return set()

    return _perform_vector_search(
        search_type_name="Welcome 주관식",
        collection_name=os.getenv("QDRANT_COLLECTION_WELCOME_NAME", "welcome_subjective_vectors"),
        query_vector=query_vector, 
        panel_id_key="metadata.panel_id",
        score_threshold=0.3, 
        pre_filter_panel_ids=pre_filter_panel_ids
    )

def search_qpoll(query_vector: List[float], pre_filter_panel_ids: Optional[Set[str]] = None, keywords: List[str] = None) -> Set[str]:
    """QPoll Qdrant 검색"""
    if not query_vector:
        logging.info("   QPoll: 벡터 없음")
        return set()

    return _perform_vector_search(
        search_type_name="QPoll",
        collection_name=os.getenv("QDRANT_COLLECTION_QPOLL_NAME", "qpoll_vectors_v2"),
        query_vector=query_vector,
        panel_id_key="panel_id",
        score_threshold=0.3,
        pre_filter_panel_ids=pre_filter_panel_ids
    )