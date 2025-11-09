import os
import re
from typing import Optional, Tuple, List, Set
from datetime import datetime
from dotenv import load_dotenv
from db_logic import get_db_connection, get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchAny
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EMBEDDINGS = None
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
    """KURE 임베딩 모델 초기화"""
    global EMBEDDINGS
    if EMBEDDINGS is None:
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
        
        if '~' in keyword:
            age_range = keyword.replace('대', '').split('~')
            if len(age_range) == 2 and age_range[0].isdigit() and age_range[1].isdigit():
                age_start = int(age_range[0])
                age_end = int(age_range[1])
                birth_start = CURRENT_YEAR - age_end - 9
                birth_end = CURRENT_YEAR - age_start
                
                self.conditions.append(
                    "(structured_data->>'birth_year' ~ '^[0-9]+$' "
                    "AND (structured_data->>'birth_year')::int BETWEEN %s AND %s)"
                )
                self.params.extend([birth_start, birth_end])
        
        elif keyword[:-1].isdigit():
            age_prefix = int(keyword[:-1])
            birth_start = CURRENT_YEAR - age_prefix - 9
            birth_end = CURRENT_YEAR - age_prefix
            
            self.conditions.append(
                "(structured_data->>'birth_year' ~ '^[0-9]+$' "
                "AND (structured_data->>'birth_year')::int BETWEEN %s AND %s)"
            )
            self.params.extend([birth_start, birth_end])
    
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
            conn.close()


def search_welcome_subjective(keywords: List[str]) -> Set[str]:
    """Welcome 주관식 Qdrant 검색"""
    if not keywords:
        print("   ⚠️  Welcome 주관식: 키워드 없음")
        return set()
    
    try:
        embeddings = initialize_embeddings()
        qdrant_client = get_qdrant_client()
        
        if not qdrant_client:
            print("   ❌ Welcome 주관식: Qdrant 연결 실패")
            return set()
        
        query_text = " ".join(keywords)
        query_vector = embeddings.embed_query(query_text)
        collection_name = os.getenv("QDRANT_COLLECTION_WELCOME_NAME", "welcome_subjective_vectors")
        
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=1000,
            with_payload=True,
            score_threshold=0.5
        )
        
        panel_ids = set()
        for result in search_results:
            panel_id = extract_panel_id_from_payload(result.payload)
            if panel_id:
                panel_ids.add(panel_id)
        
        print(f"   ✅ Welcome 주관식: {len(panel_ids):,}명")
        return panel_ids
    except Exception as e:
        print(f"   ❌ Welcome 주관식 검색 실패: {e}")
        return set()


def search_welcome_two_stage(
    objective_keywords: List[str],
    subjective_keywords: List[str],
    limit: int = 1000
) -> Set[str]:
    """2단계 하이브리드 검색"""
    
    print(f"\n🔍 2단계 검색 시작")
    print(f"   1단계 키워드: {objective_keywords}")
    print(f"   2단계 키워드: {subjective_keywords}")
    
    # 1단계: PostgreSQL
    panel_ids_stage1 = search_welcome_objective(objective_keywords)
    
    if not panel_ids_stage1:
        print("   ⚠️  1단계 결과 없음 → 검색 종료")
        return set()
    
    if not subjective_keywords:
        print("   ℹ️  2단계 키워드 없음 → 1단계 결과 반환")
        return panel_ids_stage1
    
    # 2단계: Qdrant
    try:
        embeddings = initialize_embeddings()
        qdrant_client = get_qdrant_client()
        
        if not qdrant_client:
            print("   ⚠️  Qdrant 연결 실패 → 1단계 결과 반환")
            return panel_ids_stage1
        
        query_text = " ".join(subjective_keywords)
        query_vector = embeddings.embed_query(query_text)
        collection_name = os.getenv("QDRANT_COLLECTION_WELCOME_NAME", "welcome_subjective_vectors")
        
        panel_id_list = list(panel_ids_stage1)
        chunk_size = 1000
        all_results = []
        
        print(f"   🔄 청크 검색 시작 (대상: {len(panel_ids_stage1):,}명)")
        
        for i in range(0, len(panel_id_list), chunk_size):
            chunk = panel_id_list[i:i+chunk_size]
            
            try:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.panel_id",
                            match=MatchAny(any=chunk)
                        )
                    ]
                )
                
                results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=qdrant_filter,
                    limit=min(limit, len(chunk)),
                    score_threshold=0.3
                )
                
                if results:
                    all_results.extend(results)
                
            except Exception as e:
                print(f"   ⚠️  청크 {i//chunk_size + 1} 검색 실패: {e}")
                continue
        
        print(f"   📊 Qdrant 검색 결과: {len(all_results)}개")
        
        all_results.sort(key=lambda x: x.score, reverse=True)
        all_results = all_results[:limit]
        
        panel_ids_stage2 = set()
        for result in all_results:
            panel_id = extract_panel_id_from_payload(result.payload)
            if panel_id:
                panel_ids_stage2.add(panel_id)
        
        print(f"   ✅ 2단계 최종 결과: {len(panel_ids_stage2):,}명")
        return panel_ids_stage2
        
    except Exception as e:
        print(f"   ❌ 2단계 검색 실패: {e}")
        return panel_ids_stage1


def search_qpoll(survey_type: str, keywords: List[str]) -> Set[str]:
    """QPoll Qdrant 검색"""
    if not keywords:
        print("   ⚠️  QPoll: 키워드 없음")
        return set()
    
    try:
        embeddings = initialize_embeddings()
        qdrant_client = get_qdrant_client()
        
        if not qdrant_client:
            print("   ❌ QPoll: Qdrant 연결 실패")
            return set()
        
        query_text = " ".join(keywords)
        query_vector = embeddings.embed_query(query_text)
        collection_name = os.getenv("QDRANT_COLLECTION_QPOLL_NAME", "qpoll_vector_v2")
        
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=1000,
            with_payload=True,
            score_threshold=0.3
        )
        
        print(f"   📊 Qdrant 검색 결과: {len(search_results)}개")
        
        panel_ids = set()
        for result in search_results:
            panel_id = result.payload.get('panel_id')
            if panel_id:
                panel_ids.add(str(panel_id))
        
        print(f"   ✅ QPoll: {len(panel_ids):,}명")
        return panel_ids
        
    except Exception as e:
        print(f"   ❌ QPoll 검색 실패: {e}")
        return set()


def hybrid_search(classified_keywords: dict, search_mode: str = "all") -> dict:
    """하이브리드 검색"""
    welcome_obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
    welcome_subj_keywords = classified_keywords.get('welcome_keywords', {}).get('subjective', [])
    
    use_two_stage = len(welcome_obj_keywords) > 0 and len(welcome_subj_keywords) > 0
    
    print(f"\n📌 2단계: 하이브리드 검색")
    print(f"   검색 전략: {'2단계 검색' if use_two_stage else '개별 검색'}")
    
    if use_two_stage:
        panel_id1 = search_welcome_two_stage(
            objective_keywords=welcome_obj_keywords,
            subjective_keywords=welcome_subj_keywords
        )
        panel_id2 = set()
    else:
        if welcome_obj_keywords:
            print(f"\n🔍 Welcome 객관식 검색")
            panel_id1 = search_welcome_objective(welcome_obj_keywords)
        else:
            panel_id1 = set()
        
        if welcome_subj_keywords:
            print(f"\n🔍 Welcome 주관식 검색")
            panel_id2 = search_welcome_subjective(welcome_subj_keywords)
        else:
            panel_id2 = set()
    
    qpoll_data = classified_keywords.get('qpoll_keywords', {})
    survey_type = qpoll_data.get('survey_type')
    qpoll_keywords = qpoll_data.get('keywords', [])
    
    if qpoll_keywords:
        print(f"\n🔍 QPoll 검색")
        panel_id3 = search_qpoll(survey_type, qpoll_keywords)
    else:
        print(f"\n⚠️  QPoll: 키워드 없음")
        panel_id3 = set()
    
    all_sets = [s for s in [panel_id1, panel_id2, panel_id3] if s]
    
    results = {}
    
    # 교집합
    if not all_sets:
        intersection_panel_ids = []
        intersection_scores = {}
    elif len(all_sets) == 1:
        intersection_panel_ids = list(all_sets[0])
        intersection_scores = {panel_id: 1.0 for panel_id in intersection_panel_ids}
    else:
        intersection_set = set.intersection(*all_sets)
        intersection_panel_ids = list(intersection_set)
        intersection_scores = {panel_id: float(len(all_sets)) for panel_id in intersection_panel_ids}
    
    results['intersection'] = {
        'panel_ids': intersection_panel_ids,
        'count': len(intersection_panel_ids),
        'scores': intersection_scores
    }
    
    # 합집합
    if not all_sets:
        union_panel_ids = []
        union_scores = {}
    else:
        union_set = set.union(*all_sets)
        union_scores = {panel_id: sum([1 if panel_id in s else 0 for s in [panel_id1, panel_id2, panel_id3]]) for panel_id in union_set}
        union_panel_ids = sorted(union_set, key=lambda x: union_scores[x], reverse=True)
    
    results['union'] = {
        'panel_ids': union_panel_ids,
        'count': len(union_panel_ids),
        'scores': union_scores
    }
    
    # 가중치
    weights = {'panel_id1': 0.4, 'panel_id2': 0.3, 'panel_id3': 0.3}
    
    if not all_sets:
        weighted_panel_ids = []
        weighted_scores = {}
    else:
        all_panel_ids = set.union(*all_sets)
        weighted_scores = {}
        
        for panel_id in all_panel_ids:
            score = 0.0
            if panel_id in panel_id1:
                score += weights['panel_id1']
            if panel_id in panel_id2:
                score += weights['panel_id2']
            if panel_id in panel_id3:
                score += weights['panel_id3']
            weighted_scores[panel_id] = score
        
        weighted_panel_ids = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)
    
    results['weighted'] = {
        'panel_ids': weighted_panel_ids,
        'count': len(weighted_panel_ids),
        'scores': weighted_scores,
        'weights': weights
    }
    
    # 최종 요약
    print(f"\n{'='*70}")
    print(f"📊 검색 결과 요약")
    print(f"{'='*70}")
    if use_two_stage:
        print(f"Welcome 2단계: {len(panel_id1):,}명")
    else:
        print(f"Welcome 객관식: {len(panel_id1):,}명")
        print(f"Welcome 주관식: {len(panel_id2):,}명")
    print(f"QPoll: {len(panel_id3):,}명")
    print(f"")
    print(f"교집합: {results['intersection']['count']:,}명")
    print(f"합집합: {results['union']['count']:,}명")
    print(f"가중치: {results['weighted']['count']:,}명")
    print(f"{'='*70}\n")
    
    # search_mode에 따른 최종 결과 선택
    if search_mode == 'intersection':
        final_panel_ids = results['intersection']['panel_ids']
        match_scores = results['intersection']['scores']
    elif search_mode == 'union':
        final_panel_ids = results['union']['panel_ids']
        match_scores = results['union']['scores']
    elif search_mode == 'weighted':
        final_panel_ids = results['weighted']['panel_ids']
        match_scores = results['weighted']['scores']
    else:
        final_panel_ids = results['weighted']['panel_ids']
        match_scores = results['weighted']['scores']
    
    return {
        "panel_id1": panel_id1,
        "panel_id2": panel_id2,
        "panel_id3": panel_id3,
        "final_panel_ids": final_panel_ids,
        "match_scores": match_scores,
        "results": results,
        "two_stage_used": use_two_stage
    }
