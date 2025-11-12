import os
import re
from typing import Optional, Tuple, List, Set
from datetime import datetime
import threading
from dotenv import load_dotenv
from db_logic_optimized import get_db_connection_context, get_qdrant_client, get_db_connection, return_db_connection
from qdrant_client.models import Filter, FieldCondition, MatchAny
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
    """KURE 임베딩 모델 초기화 (스레드 안전하게 수정)"""
    global EMBEDDINGS

    # ⭐️ [수정] 모델이 이미 로드된 경우, 새로 생성하지 않고 즉시 반환 (싱글톤)
    if EMBEDDINGS is None:
        # Lock을 사용하여 단 한 번만 모델이 로드되도록 보장
        with embedding_lock:
            if EMBEDDINGS is None:
                print("   ⏳ (최초 1회) 임베딩 모델 초기화 중...")
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
        self.age_ranges = [] # 1. [추가] 연령대 리스트
    
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
        
        # 2. [수정] conditions.append 대신 age_ranges.append로 수정
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
        
        # ⭐️ [수정] 누락된 연령대 조건 추가
        if self.age_ranges:
            age_conditions = []
            for start, end in self.age_ranges:
                age_conditions.append("(CAST(structured_data->>'birth_year' AS INTEGER) BETWEEN %s AND %s)")
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
    """Welcome 객관식 PostgreSQL 검색"""
    if not keywords:
        print("   ⚠️  Welcome 객관식: 키워드 없음")
        return set()
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("   ❌ Welcome 객관식: DB 연결 실패")
            return set()
        
        cur = conn.cursor()
        where_clause, params = build_welcome_query_conditions(keywords)
        
        if not where_clause:
            print("   ⚠️  Welcome 객관식: 조건 없음")
            return set()
        
        query = f"SELECT panel_id FROM welcome_meta2 {where_clause}"
        cur.execute(query, tuple(params))
        results = {str(row[0]) for row in cur.fetchall()}
        cur.close()
        
        print(f"   ✅ Welcome 객관식: {len(results):,}명")
        return results
    except Exception as e:
        print(f"   ❌ Welcome 객관식 검색 실패: {e}")
        return set()
    finally:
        if conn:
            return_db_connection(conn)


# ⭐️ [신규] 헬퍼 함수: 단일 벡터 검색
def _search_single_vector(
    query_text: str, 
    pre_filter_panel_ids: Set[str]
) -> Set[str]:
    """단일 쿼리 텍스트로 Qdrant 벡터 검색 수행"""
    try:
        # 1. 스레드별 임베딩/클라이언트 초기화
        embeddings = initialize_embeddings() # 이미 로드된 모델 가져오기
        qdrant_client = get_qdrant_client()
        
        if not qdrant_client:
            raise Exception("Qdrant 클라이언트 연결 실패")
        
        with embedding_lock: # ⭐️ Lock으로 임베딩 연산 보호
            query_vector = embeddings.embed_query(query_text)
        collection_name = os.getenv("QDRANT_COLLECTION_WELCOME_NAME", "welcome_subjective_vectors")
        
        # 2. 필터 설정
        query_filter = None
        if pre_filter_panel_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.panel_id", 
                        match=MatchAny(any=list(pre_filter_panel_ids))
                    )
                ]
            )

        # 3. 검색
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,  
            limit=1000,
            with_payload=True,
            score_threshold=0.35
        )
        
        # 4. 결과 파싱
        panel_ids = set()
        for result in search_results:
            panel_id = extract_panel_id_from_payload(result.payload)
            if panel_id:
                panel_ids.add(panel_id)
        
        return panel_ids
        
    except Exception as e:
        print(f"   ❌ 하위 쿼리 '{query_text[:10]}...' 검색 실패: {e}")
        return set()


# ⭐️ [수정] 다중 쿼리를 처리하도록 search_welcome_subjective 수정
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

def search_welcome_subjective(
    keyword_groups: List[List[str]], # ⭐️ List[str] -> List[List[str]]
    pre_filter_panel_ids: Set[str] = None 
) -> Set[str]:
    """Welcome 주관식 Qdrant 검색 (다중 쿼리 지원)"""
    if not keyword_groups:
        print("   ⚠️  Welcome 주관식: 키워드 없음")
        return set()

    all_panel_ids = Counter() # ⭐️ Counter로 변경
    
    # ⭐️ 내부 ThreadPoolExecutor로 각 키워드 그룹을 병렬 검색
    with ThreadPoolExecutor(max_workers=len(keyword_groups) or 1) as executor:
        future_to_group = {
            executor.submit(
                _search_single_vector, 
                " ".join(group),  # 헬퍼 함수는 List[str]이 아닌 str을 받음
                pre_filter_panel_ids
            ): group
            for group in keyword_groups
        }
        
        for future in as_completed(future_to_group):
            try:
                panel_ids = future.result() # Set[str] 반환
                for panel_id in panel_ids:
                    all_panel_ids[panel_id] += 1 # ⭐️ 매칭된 쿼리 수 카운트
            except Exception as e:
                print(f"   ❌ 하위 쿼리 그룹 실행 실패: {e}")

    # ⭐️ (전략) 여기서는 1개 이상의 쿼리와 매칭된 모든 ID를 반환
    final_ids = {pid for pid, count in all_panel_ids.items() if count >= 1}
    print(f"   ✅ Welcome 주관식 (다중 쿼리): {len(final_ids):,}명")
    return final_ids


def search_qpoll(
    survey_type: Optional[str], 
    keywords: List[str],
    pre_filter_panel_ids: Set[str] = None
) -> Set[str]:
    """QPoll Qdrant 벡터 검색"""
    if not keywords:
        print("   ⚠️  QPoll: 키워드 없음")
        return set()

    try:
        embeddings = initialize_embeddings() # 이미 로드된 모델 가져오기
        qdrant_client = get_qdrant_client()
        if not qdrant_client:
            raise Exception("Qdrant 클라이언트 연결 실패")

        # ⭐️ [수정] LLM이 생성한 복잡한 키워드 구조(dict) 처리
        processed_keywords = []
        for item in keywords:
            if isinstance(item, dict):
                # 딕셔너리 내의 모든 문자열과 리스트 내 문자열을 재귀적으로 추출
                if 'category' in item and isinstance(item['category'], str):
                    processed_keywords.append(item['category'])
                if 'brands' in item and isinstance(item['brands'], list):
                    processed_keywords.extend([kw for kw in item['brands'] if isinstance(kw, str)])
                if 'behaviors' in item and isinstance(item['behaviors'], list):
                    processed_keywords.extend([kw for kw in item['behaviors'] if isinstance(kw, str)])
            elif isinstance(item, str):
                processed_keywords.append(item)
        
        query_text = " ".join(processed_keywords)

        with embedding_lock: # ⭐️ Lock으로 임베딩 연산 보호
            query_vector = embeddings.embed_query(query_text)
        collection_name = os.getenv("QDRANT_COLLECTION_QPOLL_NAME", "qpoll_vector_v2")

        query_filter = None
        if pre_filter_panel_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="panel_id",
                        match=MatchAny(any=list(pre_filter_panel_ids))
                    )
                ]
            )

        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=1000,
            with_payload=True,
            score_threshold=0.5
        )

        panel_ids = {str(result.payload['panel_id']) for result in search_results if 'panel_id' in result.payload}
        return panel_ids
    except Exception as e:
        print(f"   ❌ QPoll 검색 실패: {e}")
        return set()