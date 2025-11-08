import os
import json
import re
from typing import Optional
from dotenv import load_dotenv
from db_logic import get_db_connection, get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchAny
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EMBEDDINGS = None

def initialize_embeddings():
    """KURE 임베딩 모델 초기화"""
    global EMBEDDINGS
    if EMBEDDINGS is None:
        print("⏳ KURE 임베딩 모델 로딩 중...")
        EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="nlpai-lab/KURE-v1",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ KURE 임베딩 모델 로드 완료")
    return EMBEDDINGS


def build_welcome_query_conditions(keywords: list[str]) -> tuple[str, list]:
    """
    실제 DB 구조에 맞춘 쿼리 빌더
    ✅ CASE WHEN으로 안전한 배열 체크!
    """
    conditions = []
    params = []
    current_year = 2025
    
    regions = []
    
    for keyword in keywords:
        kw = keyword.strip().lower()
        
        # ===== 성별 =====
        if kw in ['남자', '남성', '남']:
            conditions.append(
                "(structured_data->>'gender' IS NOT NULL "
                "AND structured_data->>'gender' != '' "
                "AND NOT (structured_data->>'gender' ~ '^Q[0-9]') "
                "AND structured_data->>'gender' = %s)"
            )
            params.append('M')
            
        elif kw in ['여자', '여성', '여']:
            conditions.append(
                "(structured_data->>'gender' IS NOT NULL "
                "AND structured_data->>'gender' != '' "
                "AND NOT (structured_data->>'gender' ~ '^Q[0-9]') "
                "AND structured_data->>'gender' = %s)"
            )
            params.append('F')
        
        # ===== 지역 =====
        elif keyword in ['서울', '경기', '인천', '부산', '대구', '대전', '광주', '울산', '세종',
                        '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']:
            regions.append(keyword)
        
        # ===== 나이대 (범위 지원) =====
        elif '대' in keyword:
            if '~' in keyword:
                age_range = keyword.replace('대', '').split('~')
                if len(age_range) == 2 and age_range[0].isdigit() and age_range[1].isdigit():
                    age_start = int(age_range[0])
                    age_end = int(age_range[1])
                    birth_start = current_year - age_end - 9
                    birth_end = current_year - age_start
                    
                    conditions.append(
                        "(structured_data->>'birth_year' IS NOT NULL "
                        "AND NOT (structured_data->>'birth_year' ~ '^Q[0-9]') "
                        "AND structured_data->>'birth_year' ~ '^-?[0-9]+$' "
                        "AND (structured_data->>'birth_year')::int BETWEEN %s AND %s)"
                    )
                    params.extend([birth_start, birth_end])
            elif keyword[:-1].isdigit():
                age_prefix = int(keyword[:-1])
                birth_start = current_year - age_prefix - 9
                birth_end = current_year - age_prefix
                
                conditions.append(
                    "(structured_data->>'birth_year' IS NOT NULL "
                    "AND NOT (structured_data->>'birth_year' ~ '^Q[0-9]') "
                    "AND structured_data->>'birth_year' ~ '^-?[0-9]+$' "
                    "AND (structured_data->>'birth_year')::int BETWEEN %s AND %s)"
                )
                params.extend([birth_start, birth_end])
        
        # ===== 결혼 상태 =====
        elif kw in ['미혼', '싱글']:
            conditions.append(
                "(structured_data->>'marital_status' IS NOT NULL "
                "AND structured_data->>'marital_status' = %s)"
            )
            params.append('미혼')
            
        elif kw in ['기혼', '결혼']:
            conditions.append(
                "(structured_data->>'marital_status' IS NOT NULL "
                "AND structured_data->>'marital_status' = %s)"
            )
            params.append('기혼')
            
        elif kw in ['이혼', '돌싱', '사별']:
            conditions.append(
                "(structured_data->>'marital_status' IS NOT NULL "
                "AND structured_data->>'marital_status' LIKE %s)"
            )
            params.append('%기타%')
        
        # ===== 음주 (✅ CASE WHEN으로 안전한 체크!) =====
        elif kw in ['술먹는', '음주', '술', '맥주', '소주', '와인']:
            conditions.append(
                "(CASE "
                "  WHEN structured_data->'drinking_experience' IS NULL THEN false "
                "  WHEN jsonb_typeof(structured_data->'drinking_experience') != 'array' THEN false "
                "  ELSE jsonb_array_length(structured_data->'drinking_experience') > 0 "
                "END)"
            )
        
        elif kw in ['술안먹는', '금주']:
            conditions.append(
                "(CASE "
                "  WHEN structured_data->'drinking_experience' IS NULL THEN true "
                "  WHEN jsonb_typeof(structured_data->'drinking_experience') != 'array' THEN true "
                "  ELSE jsonb_array_length(structured_data->'drinking_experience') = 0 "
                "END)"
            )
        
        # ===== 흡연 (✅ CASE WHEN으로 안전한 체크!) =====
        elif kw in ['흡연', '담배']:
            conditions.append(
                "(CASE "
                "  WHEN structured_data->'smoking_experience' IS NULL THEN false "
                "  WHEN jsonb_typeof(structured_data->'smoking_experience') != 'array' THEN false "
                "  ELSE jsonb_array_length(structured_data->'smoking_experience') > 0 "
                "END)"
            )
        
        elif kw in ['비흡연', '금연']:
            conditions.append(
                "(CASE "
                "  WHEN structured_data->'smoking_experience' IS NULL THEN true "
                "  WHEN jsonb_typeof(structured_data->'smoking_experience') != 'array' THEN true "
                "  ELSE jsonb_array_length(structured_data->'smoking_experience') = 0 "
                "END)"
            )
        
        # ===== 차량 보유 =====
        elif kw in ['차있음', '자가용', '차량보유']:
            conditions.append(
                "(structured_data->>'car_ownership' IS NOT NULL "
                "AND structured_data->>'car_ownership' = %s)"
            )
            params.append('있다')
            
        elif kw in ['차없음']:
            conditions.append(
                "(structured_data->>'car_ownership' IS NOT NULL "
                "AND structured_data->>'car_ownership' = %s)"
            )
            params.append('없다')
        
        # ===== 가족 구성원 수 =====
        elif '가족' in keyword and any(char.isdigit() for char in keyword):
            num_match = re.search(r'(\d+)', keyword)
            if num_match:
                num = int(num_match.group(1))
                
                if '이상' in keyword:
                    conditions.append(
                        "(structured_data->>'family_size' IS NOT NULL "
                        "AND structured_data->>'family_size' ~ '[0-9]' "
                        "AND CAST(substring(structured_data->>'family_size' from '[0-9]+') AS int) >= %s)"
                    )
                    params.append(num)
                elif '이하' in keyword:
                    conditions.append(
                        "(structured_data->>'family_size' IS NOT NULL "
                        "AND structured_data->>'family_size' ~ '[0-9]' "
                        "AND CAST(substring(structured_data->>'family_size' from '[0-9]+') AS int) <= %s)"
                    )
                    params.append(num)
                else:
                    conditions.append(
                        "(structured_data->>'family_size' IS NOT NULL "
                        "AND structured_data->>'family_size' ~ '[0-9]' "
                        "AND CAST(substring(structured_data->>'family_size' from '[0-9]+') AS int) = %s)"
                    )
                    params.append(num)
    
    # ===== 지역 조건 처리 =====
    if len(regions) == 1:
        conditions.append(
            "(structured_data->>'region_minor' IS NOT NULL "
            "AND NOT (structured_data->>'region_minor' ~ '^Q[0-9]') "
            "AND structured_data->>'region_minor' = %s)"
        )
        params.append(regions[0])
        
    elif len(regions) > 1:
        placeholders = ','.join(['%s'] * len(regions))
        conditions.append(
            "(structured_data->>'region_minor' IS NOT NULL "
            "AND NOT (structured_data->>'region_minor' ~ '^Q[0-9]') "
            "AND structured_data->>'region_minor' IN ({}))".format(placeholders)
        )
        params.extend(regions)
    
    if not conditions:
        return "", []
    
    where_clause = " WHERE " + " AND ".join(conditions)
    return where_clause, params


def search_welcome_objective(keywords: list[str]) -> set[int]:
    """Welcome 테이블의 객관식 데이터를 PostgreSQL에서 검색"""
    if not keywords:
        return set()
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return set()
        
        cur = conn.cursor()
        
        where_clause, params = build_welcome_query_conditions(keywords)
        
        if not where_clause:
            print("⚠️  Welcome 객관식: 매칭되는 조건 없음")
            return set()
        
        query = f"SELECT pid FROM welcome_meta {where_clause}"
        
        print(f"\n🔍 Welcome 객관식 쿼리:")
        print(f"   키워드: {keywords}")
        print(f"   파라미터: {params}")
        
        cur.execute(query, tuple(params))
        results = {row[0] for row in cur.fetchall()}
        
        cur.close()
        print(f"✅ Welcome 객관식 검색 결과: {len(results)}개\n")
        return results
        
    except Exception as e:
        print(f"❌ Welcome 객관식 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        return set()
    finally:
        if conn:
            conn.close()


def search_welcome_subjective(keywords: list[str]) -> set[int]:
    """Welcome 주관식 Qdrant 검색"""
    if not keywords:
        return set()
    
    try:
        embeddings = initialize_embeddings()
        qdrant_client = get_qdrant_client()
        
        if not qdrant_client:
            return set()
        
        query_text = " ".join(keywords)
        query_vector = embeddings.embed_query(query_text)
        
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "welcome_subjective_vectors")
        
        print(f"🔍 Welcome 주관식 Qdrant 검색: '{query_text}'")
        
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=1000,
            # score_threshold=0.1  
        )
        
        # ✅ LangChain 형식 payload 지원: metadata.pid 우선, 없으면 최상위 pid
        pids = set()
        for result in search_results:
            # metadata.pid 시도 (LangChain 형식)
            pid = result.payload.get('metadata', {}).get('pid')
            # 최상위 pid 시도 (직접 저장 형식)
            if pid is None:
                pid = result.payload.get('pid')
            
            if pid is not None:
                pids.add(pid)
                if len(pids) <= 3:  # 처음 3개만 디버그 출력
                    print(f"      → PID {pid} 추가 (score: {result.score:.4f})")
        
        print(f"   🔍 Qdrant 원본 검색 결과: {len(search_results)}개")
        print(f"✅ Welcome 주관식 검색 결과: {len(pids)}개\n")
        return pids
        
    except Exception as e:
        print(f"❌ Welcome 주관식 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        return set()


def search_qpoll(survey_type: str, keywords: list[str]) -> set[int]:
    """QPoll Qdrant 검색"""
    if not keywords:
        return set()
    
    try:
        embeddings = initialize_embeddings()
        qdrant_client = get_qdrant_client()
        
        if not qdrant_client:
            return set()
        
        query_text = " ".join(keywords)
        query_vector = embeddings.embed_query(query_text)
        
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "welcome_subjective_vectors")
        
        print(f"🔍 QPoll 검색: '{query_text}'")
        
        if survey_type:
            try:
                qdrant_filter = Filter(
                    must=[FieldCondition(key="survey_type", match={"value": survey_type})]
                )
                search_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=qdrant_filter,
                    limit=1000,
                    # score_threshold=0.1  
                )
            except Exception as filter_error:
                print(f"   ⚠️  필터 적용 불가: {filter_error}")
                search_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=1000,
                    # score_threshold=0.1  
                )
        else:
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=1000,
                # score_threshold=0.1 
            )
        
        # ✅ LangChain 형식 payload 지원: metadata.pid 우선, 없으면 최상위 pid
        pids = set()
        for result in search_results:
            # metadata.pid 시도 (LangChain 형식)
            pid = result.payload.get('metadata', {}).get('pid')
            # 최상위 pid 시도 (직접 저장 형식)
            if pid is None:
                pid = result.payload.get('pid')
            
            if pid is not None:
                pids.add(pid)
                if len(pids) <= 3:  # 처음 3개만 디버그 출력
                    print(f"      → PID {pid} 추가 (score: {result.score:.4f})")
        
        print(f"✅ QPoll 검색 결과: {len(pids)}개\n")
        return pids
        
    except Exception as e:
        print(f"❌ QPoll 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        return set()


def hybrid_search(classified_keywords: dict, search_mode: str = "all") -> dict:
    """하이브리드 검색 - API 응답 형식 통일"""
    print("\n" + "="*70)
    print(f"🚀 하이브리드 검색 시작 (모드: {search_mode})")
    print("="*70)
    
    welcome_obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
    pid1 = search_welcome_objective(welcome_obj_keywords)
    
    welcome_subj_keywords = classified_keywords.get('welcome_keywords', {}).get('subjective', [])
    pid2 = search_welcome_subjective(welcome_subj_keywords)
    
    qpoll_data = classified_keywords.get('qpoll_keywords', {})
    survey_type = qpoll_data.get('survey_type')
    qpoll_keywords = qpoll_data.get('keywords', [])
    pid3 = search_qpoll(survey_type, qpoll_keywords)
    
    all_sets = [s for s in [pid1, pid2, pid3] if s]
    
    results = {}
    
    # 교집합
    if not all_sets:
        intersection_pids = []
        intersection_scores = {}
    elif len(all_sets) == 1:
        intersection_pids = list(all_sets[0])
        intersection_scores = {pid: 1.0 for pid in intersection_pids}
    else:
        intersection_set = set.intersection(*all_sets)
        intersection_pids = list(intersection_set)
        intersection_scores = {pid: float(len(all_sets)) for pid in intersection_pids}
    
    results['intersection'] = {
        'pids': intersection_pids,
        'count': len(intersection_pids),
        'scores': intersection_scores
    }
    
    # 합집합
    if not all_sets:
        union_pids = []
        union_scores = {}
    else:
        union_set = set.union(*all_sets)
        union_scores = {pid: sum([1 if pid in s else 0 for s in [pid1, pid2, pid3]]) for pid in union_set}
        union_pids = sorted(union_set, key=lambda x: union_scores[x], reverse=True)
    
    results['union'] = {
        'pids': union_pids,
        'count': len(union_pids),
        'scores': union_scores
    }
    
    # 가중치
    weights = {'pid1': 0.4, 'pid2': 0.3, 'pid3': 0.3}
    
    if not all_sets:
        weighted_pids = []
        weighted_scores = {}
    else:
        all_pids = set.union(*all_sets)
        weighted_scores = {}
        
        for pid in all_pids:
            score = 0.0
            if pid in pid1:
                score += weights['pid1']
            if pid in pid2:
                score += weights['pid2']
            if pid in pid3:
                score += weights['pid3']
            weighted_scores[pid] = score
        
        weighted_pids = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)
    
    results['weighted'] = {
        'pids': weighted_pids,
        'count': len(weighted_pids),
        'scores': weighted_scores,
        'weights': weights
    }
    
    print("\n" + "="*70)
    print("📊 검색 결과 요약")
    print("="*70)
    print(f"Welcome 객관식: {len(pid1)}개")
    print(f"Welcome 주관식: {len(pid2)}개")
    print(f"QPoll: {len(pid3)}개")
    print(f"\n교집합: {results['intersection']['count']}개")
    print(f"합집합: {results['union']['count']}개")
    print(f"가중치: {results['weighted']['count']}개")
    print("="*70 + "\n")
    
    # 모드에 따라 결과 선택
    if search_mode == 'intersection':
        final_pids = results['intersection']['pids']
        match_scores = results['intersection']['scores']
    elif search_mode == 'union':
        final_pids = results['union']['pids']
        match_scores = results['union']['scores']
    elif search_mode == 'weighted':
        final_pids = results['weighted']['pids']
        match_scores = results['weighted']['scores']
    else:  # 'all'
        final_pids = results['weighted']['pids']
        match_scores = results['weighted']['scores']
    
    return {
        "pid1": pid1,
        "pid2": pid2,
        "pid3": pid3,
        "final_pids": final_pids,
        "match_scores": match_scores,
        "results": results
    }


if __name__ == "__main__":
    print("\n🧪 테스트: ['부산', '40대'. 가전제품 보유']")
    
    test = {
        "welcome_keywords": {
            "objective": ["부산"],
            "subjective": ["가전제품 보유"]
        },
        "qpoll_keywords": {
            "survey_type": None,
            "keywords": []
        }
    }
    
    result = hybrid_search(test, search_mode="all")
    print(f"\n✅ 최종 결과: {len(result['final_pids'])}개")
    if result['final_pids']:
        print(f"   상위 10개 PID: {result['final_pids'][:10]}")