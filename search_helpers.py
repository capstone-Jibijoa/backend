import os
import re
import logging
from typing import Optional, Tuple, List, Set
from datetime import datetime
import threading
from dotenv import load_dotenv

from db import get_db_connection_context, get_qdrant_client, get_db_connection
from qdrant_client import QdrantClient 
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, SearchParams
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
    expanded = []
    for keyword in keywords:
        region_match = re.match(r'^(서울|경기|부산|인천|대구|광주|대전|울산|세종|강원|충북|충남|전북|전남|경북|경남|제주)(시|도)?$', keyword)
        if region_match:
            expanded.append(region_match.group(1))
        elif keyword in CATEGORY_MAPPING:
            expanded.extend(CATEGORY_MAPPING[keyword])
        else:
            expanded.append(keyword)
    return expanded
def initialize_embeddings():
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
    try:
        if 'metadata' in payload and isinstance(payload['metadata'], dict):
            panel_id = payload['metadata'].get('panel_id')
            if panel_id: return str(panel_id)
        panel_id = payload.get('panel_id')
        if panel_id: return str(panel_id)
        return None
    except Exception:
        return None
class ConditionBuilder:
    def __init__(self):
        self.conditions = []
        self.params = []
        self.regions = []
        self.jobs = []
        self.incomes = []
        self.educations = []
        self.age_ranges = []
        self.job_duties = []
        self.phone_brands = []
        self.car_manufacturers = []
        self.used_keywords = set() # [신규] 사용된 키워드 추적
    def add_gender(self, keyword: str):
        kw = keyword.strip().lower()
        if re.match(r'^(남자|남성|남)$', kw):
            self.conditions.append("(structured_data->>'gender' = %s)"); self.params.append('M')
        elif re.match(r'^(여자|여성|여)$', kw):
            self.conditions.append("(structured_data->>'gender' = %s)"); self.params.append('F')
        else: return
        self.used_keywords.add(keyword)
    def add_region(self, keyword: str):
        if keyword in VALID_REGIONS:
            self.regions.append(keyword)
            self.used_keywords.add(keyword)
    def add_age_range(self, keyword: str):
        if '대' not in keyword: return
        birth_start = None; birth_end = None
        if '~' in keyword:
            age_range = keyword.replace('대', '').split('~')
            if len(age_range) == 2 and age_range[0].isdigit() and age_range[1].isdigit():
                age_start = int(age_range[0]); age_end = int(age_range[1])
                birth_start = CURRENT_YEAR - age_end - 9; birth_end = CURRENT_YEAR - age_start
        elif keyword[:-1].isdigit():
            age_prefix = int(keyword[:-1])
            birth_start = CURRENT_YEAR - age_prefix - 9; birth_end = CURRENT_YEAR - age_prefix
        if birth_start is not None and birth_end is not None:
            self.age_ranges.append((birth_start, birth_end))
            self.used_keywords.add(keyword)
    def add_job(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['사무직', '전문직', '경영관리직', '생산노무직', '서비스직', '판매직', '기술직', '직장인']:
            self.jobs.append(keyword)
            self.used_keywords.add(keyword)
    def add_income(self, keyword: str):
        if '월' in keyword and '만원' in keyword:
            self.incomes.append(keyword)
            self.used_keywords.add(keyword)
    def add_education(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['대학교 졸업', '대학원 재학 이상', '고등학교 졸업 이하', '중학교 졸업 이하']:
            self.educations.append(keyword)
            self.used_keywords.add(keyword)
    def add_job_duty(self, keyword: str):
        kw = keyword.lower()
        kw_normalized = kw.replace('직무', '').replace('가', '').strip() 
        # 'it'와 같은 핵심 단어가 포함된 경우 job_duty 또는 job_title에서 검색
        if kw_normalized in ['it', '개발', '기획', '마케팅', '디자인', '영업', '연구']:
            self.job_duties.append(f'%{kw_normalized}%')
            self.used_keywords.add(keyword)
    def add_phone_brand(self, keyword: str):
        kw = keyword.lower()
        if kw in ['아이폰', '애플']: self.phone_brands.append('Apple')
        elif kw in ['삼성', '갤럭시']: self.phone_brands.append('Samsung')
        elif kw == 'lg': self.phone_brands.append('LG')
        else: return
        self.used_keywords.add(keyword)
    def add_car_manufacturer(self, keyword: str):
        kw = keyword.lower()
        if kw in ['현대', '현대차']: self.car_manufacturers.append('현대')
        elif kw == '기아': self.car_manufacturers.append('기아')
        elif kw == 'bmw': self.car_manufacturers.append('BMW')
        elif kw == '테슬라': self.car_manufacturers.append('테슬라')
        else: return
        self.used_keywords.add(keyword)
    def add_marital_status(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['미혼', '싱글']:
            self.conditions.append("(structured_data->>'marital_status' = %s)"); self.params.append('미혼')
        elif kw in ['기혼', '결혼']:
            self.conditions.append("(structured_data->>'marital_status' = %s)"); self.params.append('기혼')
        elif kw in ['이혼', '돌싱', '사별']:
            self.conditions.append("(structured_data->>'marital_status' LIKE %s)"); self.params.append('%기타%')
        else: return
        self.used_keywords.add(keyword)
    def add_drinking(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['술먹는', '음주', '술', '맥주', '소주', '와인']:
            self.conditions.append("(jsonb_array_length(COALESCE(structured_data->'drinking_experience', '[]'::jsonb)) > 0)")
        elif kw in ['술안먹는', '금주']:
            self.conditions.append("(jsonb_array_length(COALESCE(structured_data->'drinking_experience', '[]'::jsonb)) = 0)")
        else: return
        self.used_keywords.add(keyword)
    def add_smoking(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['흡연', '담배']:
            self.conditions.append("(jsonb_array_length(COALESCE(structured_data->'smoking_experience', '[]'::jsonb)) > 0)")
        elif kw in ['비흡연', '금연']:
            self.conditions.append("(jsonb_array_length(COALESCE(structured_data->'smoking_experience', '[]'::jsonb)) = 0)")
        else: return
        self.used_keywords.add(keyword)
    def add_car_ownership(self, keyword: str):
        kw = keyword.strip().lower()
        if kw in ['차있음', '자가용', '차량보유']:
            self.conditions.append("(structured_data->>'car_ownership' = %s)"); self.params.append('있다')
        elif kw in ['차없음']:
            self.conditions.append("(structured_data->>'car_ownership' = %s)"); self.params.append('없다')
        else: return
        self.used_keywords.add(keyword)
    def add_family_size(self, keyword: str):
        if '가족' not in keyword or not any(char.isdigit() for char in keyword): return
        num_match = re.search(r'(\d+)', keyword);
        if not num_match: return
        num = int(num_match.group(1))
        if '이상' in keyword:
            self.conditions.append("(structured_data->>'family_size' ~ '[0-9]' AND CAST(substring(structured_data->>'family_size' from '[0-9]+') AS int) >= %s)"); self.params.append(num)
        elif '이하' in keyword:
            self.conditions.append("(structured_data->>'family_size' ~ '[0-9]' AND CAST(substring(structured_data->>'family_size' from '[0-9]+') AS int) <= %s)"); self.params.append(num)
        else:
            self.conditions.append("(structured_data->>'family_size' ~ '[0-9]' AND CAST(substring(structured_data->>'family_size' from '[0-9]+') AS int) = %s)"); self.params.append(num)
        self.used_keywords.add(keyword)
    def finalize(self) -> Tuple[str, List]:
        if self.jobs:
            job_conditions = ["(structured_data->>'job_title_raw' ILIKE %s)" for _ in self.jobs]
            self.conditions.append(f"({' OR '.join(job_conditions)})"); self.params.extend([f'%{job}%' for job in self.jobs])
        if self.incomes:
            income_conditions = ["(structured_data->>'income_personal_monthly' = %s)" for _ in self.incomes]
            self.conditions.append(f"({' OR '.join(income_conditions)})"); self.params.extend(self.incomes)
        if self.educations:
            edu_conditions = ["(structured_data->>'education_level' = %s)" for _ in self.educations]
            self.conditions.append(f"({' OR '.join(edu_conditions)})"); self.params.extend(self.educations)
        if len(self.regions) == 1:
            self.conditions.append("(structured_data->>'region_major' = %s)"); self.params.append(self.regions[0])
        elif len(self.regions) > 1:
            placeholders = ','.join(['%s'] * len(self.regions))
            self.conditions.append(f"(structured_data->>'region_major' IN ({placeholders}))"); self.params.extend(self.regions)
        if self.age_ranges:
            age_conditions = []
            for start, end in self.age_ranges:
                age_conditions.append("(structured_data->>'birth_year' ~ '^[0-9]+$' AND (structured_data->>'birth_year')::int BETWEEN %s AND %s)"); self.params.extend([start, end])
            self.conditions.append(f"({' OR '.join(age_conditions)})")
        if self.job_duties:
            # 'IT' 같은 키워드가 'job_duty_raw' 또는 'job_title_raw'에 포함되는 경우 검색
            duty_conditions = []
            for duty in self.job_duties:
                duty_conditions.append("((structured_data->>'job_duty_raw' ILIKE %s) OR (structured_data->>'job_title_raw' ILIKE %s))")
                self.params.extend([duty, duty])
            self.conditions.append(f"({' OR '.join(duty_conditions)})")
        if self.phone_brands:
            brand_conditions = ["(structured_data->>'phone_brand_raw' = %s)" for _ in self.phone_brands]
            self.conditions.append(f"({' OR '.join(brand_conditions)})"); self.params.extend(self.phone_brands)
        if self.car_manufacturers:
            car_conditions = ["(structured_data->>'car_manufacturer_raw' = %s)" for _ in self.car_manufacturers]
            self.conditions.append(f"({' OR '.join(car_conditions)})"); self.params.extend(self.car_manufacturers)
        if not self.conditions: return "", []
        where_clause = " WHERE " + " AND ".join(self.conditions)
        return where_clause, self.params
def build_welcome_query_conditions(keywords: List[str]) -> Tuple[str, List, Set[str]]:
    keywords = expand_keywords(keywords)
    builder = ConditionBuilder()
    for keyword in keywords:
        builder.add_gender(keyword); builder.add_region(keyword); builder.add_age_range(keyword)
        builder.add_job(keyword); builder.add_income(keyword); builder.add_education(keyword)
        builder.add_marital_status(keyword); builder.add_drinking(keyword); builder.add_smoking(keyword)
        builder.add_car_ownership(keyword); builder.add_family_size(keyword)
        builder.add_job_duty(keyword); builder.add_phone_brand(keyword); builder.add_car_manufacturer(keyword)
    
    where_clause, params = builder.finalize()
    unhandled_keywords = set(keywords) - builder.used_keywords
    return where_clause, params, unhandled_keywords
def search_welcome_objective(
    keywords: List[str]
) -> Tuple[Set[str], Set[str]]:
    if not keywords:
        logging.info("   Welcome 객관식: 키워드 없음")
        return set(), set()
    try:
        with get_db_connection_context() as conn:
            if not conn:
                logging.error("   Welcome 객관식: DB 연결 실패")
                return set(), set()
            cur = conn.cursor()
            where_clause, params, unhandled = build_welcome_query_conditions(keywords)
            if not where_clause:
                logging.info("   Welcome 객관식: 조건 없음"); cur.close()
                return set(), unhandled
            query = f"SELECT panel_id FROM welcome_meta2 {where_clause}"
            logging.info(f"   [SQL] {cur.mogrify(query, tuple(params)).decode('utf-8')}")
            cur.execute(query, tuple(params))
            results = {str(row[0]) for row in cur.fetchall()}
            cur.close()
        return results, unhandled
    except Exception as e:
        logging.error(f"   Welcome 객관식 검색 실패: {e}", exc_info=True)
        return set(), set(keywords)

def _perform_vector_search(
    search_type_name: str,
    collection_name: str,
    query_vector: List[float],
    qdrant_client: QdrantClient, 
    panel_id_key: str,
    score_threshold: float,
    limit: int = 5000,
    hnsw_ef: Optional[int] = None,
    qdrant_filter: Optional[Filter] = None
) -> Set[str]:
    """
    [리팩토링] Qdrant 벡터 검색 (Top-K 후 교집합 전략)
    - (수정) hnsw_ef 파라미터를 받아 검색 속도 튜닝
    """
    if not query_vector:
        logging.info(f"   {search_type_name}: 쿼리 벡터 없음")
        return set()

    try:
        if not qdrant_client:
            logging.error(f"   {search_type_name}: Qdrant 클라이언트가 없음")
            return set()
        
        # ‼️ [신규] 검색 파라미터 설정
        search_params = None
        if hnsw_ef is not None:
            logging.info(f"   ⚡ {search_type_name}: hnsw_ef={hnsw_ef}로 속도/정확도 튜닝 적용")
            search_params = SearchParams(hnsw_ef=hnsw_ef)
        
        if qdrant_filter:
            logging.info(f"   ⚡ {search_type_name}: 필터 적용하여 Top-{limit} 검색 시작")
        else:
            logging.info(f"   ⚡ {search_type_name}: 필터 없이 Top-{limit} 검색 시작")
        
        search_results = qdrant_client.search(
            collection_name=collection_name, 
            query_vector=query_vector,
            query_filter=qdrant_filter, 
            limit=limit,
            with_payload=True, 
            score_threshold=score_threshold,
            search_params=search_params 
        )

        panel_ids = set()
        for result in search_results:
            pid = result.payload.get('panel_id') if 'panel_id' in result.payload else extract_panel_id_from_payload(result.payload)
            if pid:
                panel_ids.add(str(pid))
        
        logging.info(f"   ✅ {search_type_name}: {len(panel_ids):,}명 (Top-{limit} 검색 결과)")
        return panel_ids

    except Exception as e:
        logging.error(f"   {search_type_name} 검색 실패: {e}", exc_info=True)
        return set()


def search_welcome_subjective(
    query_vector: List[float], 
    qdrant_client: QdrantClient, 
    keywords: List[str] = None,
    qdrant_filter: Optional[Filter] = None
) -> Set[str]:
    """Welcome 주관식 Qdrant 검색 (Top-K 전략)"""
    if not query_vector:
        logging.info("   Welcome 주관식: 벡터 없음")
        return set()

    return _perform_vector_search(
        search_type_name="Welcome 주관식",
        collection_name=os.getenv("QDRANT_COLLECTION_WELCOME_NAME", "welcome_subjective_vectors"),
        query_vector=query_vector,
        qdrant_client=qdrant_client, 
        panel_id_key="metadata.panel_id",
        score_threshold=0.38,
        qdrant_filter=qdrant_filter
    )


def search_qpoll(
    query_vector: List[float], 
    qdrant_client: QdrantClient, 
    keywords: List[str] = None,
    qdrant_filter: Optional[Filter] = None 
) -> Set[str]:
    """QPoll Qdrant 검색 (Top-K 전략)"""
    if not query_vector:
        logging.info("   QPoll: 벡터 없음")
        return set()

    # [수정] _perform_vector_search에 필터 전달
    return _perform_vector_search(
        search_type_name="QPoll",
        collection_name=os.getenv("QDRANT_COLLECTION_QPOLL_NAME", "qpoll_vectors_v2"),
        query_vector=query_vector,
        qdrant_client=qdrant_client, 
        panel_id_key="panel_id",
        score_threshold=0.35,
        limit=5000,
        qdrant_filter=qdrant_filter
    )