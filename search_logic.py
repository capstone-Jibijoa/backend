import os
import json
from dotenv import load_dotenv
from db_logic import get_db_connection, get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchAny
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# 임베딩 모델 초기화 (모듈 로드 시 1회)
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
    키워드 리스트를 받아 JSONB WHERE 절과 파라미터를 생성합니다.
    여러 지역 키워드는 OR 조건(IN 절)으로 처리합니다.
    
    Args:
        keywords: ["경기", "30대", "남자"] 같은 키워드
        
    Returns:
        (where_clause, params): SQL WHERE 절과 파라미터 튜플
    """
    import re
    
    conditions = []
    params = []
    current_year = 2025
    
    # 지역 키워드들을 따로 모음
    regions = []
    
    for keyword in keywords:
        kw = keyword.strip().lower()
        
        # 성별
        if kw in ['남자', '남성', '남']:
            conditions.append("structured_data->>'gender' = %s")
            params.append('M')
        elif kw in ['여자', '여성', '여']:
            conditions.append("structured_data->>'gender' = %s")
            params.append('F')
        
        # 지역 (일단 모아두기만)
        elif keyword in ['서울', '경기', '인천', '부산', '대구', '대전', '광주', '울산', '세종']:
            regions.append(keyword)
        
        # 나이대 (예: 20대, 30대)
        elif '대' in keyword and keyword[:-1].isdigit():
            age_prefix = int(keyword[:-1])
            birth_start = current_year - age_prefix - 9
            birth_end = current_year - age_prefix
            conditions.append("(structured_data->>'birth_year')::int BETWEEN %s AND %s")
            params.extend([birth_start, birth_end])
        
        # 결혼 상태
        elif kw in ['미혼', '싱글']:
            conditions.append("structured_data->>'marital_status' = %s")
            params.append('미혼')
        elif kw in ['기혼', '결혼']:
            conditions.append("structured_data->>'marital_status' = %s")
            params.append('기혼')
        elif kw in ['이혼', '돌싱']:
            conditions.append("structured_data->>'marital_status' = %s")
            params.append('이혼')
        
        # 음주
        elif kw in ['술먹는', '음주']:
            conditions.append("structured_data->>'drinking_experience' = %s")
            params.append('경험 있음')
        elif kw in ['술안먹는', '금주']:
            conditions.append("structured_data->>'drinking_experience' = %s")
            params.append('경험 없음')
        
        # 흡연
        elif kw in ['흡연', '담배']:
            conditions.append("structured_data->>'smoking_experience' = %s")
            params.append('경험 있음')
        elif kw in ['비흡연', '금연']:
            conditions.append("structured_data->>'smoking_experience' = %s")
            params.append('경험 없음')
        
        # 차량 보유
        elif kw in ['차있음', '자가용', '차량보유']:
            conditions.append("structured_data->>'car_ownership' = %s")
            params.append('보유')
        elif kw in ['차없음']:
            conditions.append("structured_data->>'car_ownership' = %s")
            params.append('미보유')
        
        # 가족 구성원 수
        elif '가족' in keyword and '구성원' in keyword:
            num_match = re.search(r'(\d+)', keyword)
            if num_match:
                num = int(num_match.group(1))
                if '이상' in keyword:
                    conditions.append("(structured_data->>'family_size')::int >= %s")
                    params.append(num)
                elif '이하' in keyword:
                    conditions.append("(structured_data->>'family_size')::int <= %s")
                    params.append(num)
                else:
                    conditions.append("(structured_data->>'family_size')::int = %s")
                    params.append(num)
        
        # 가족수 (간단 버전)
        elif '가족' in keyword and any(char.isdigit() for char in keyword):
            num_match = re.search(r'(\d+)', keyword)
            if num_match:
                conditions.append("(structured_data->>'family_size')::int = %s")
                params.append(int(num_match.group(1)))
    
    # 지역 조건 처리 (여러 개면 IN 절로)
    if len(regions) == 1:
        conditions.append("structured_data->>'region' = %s")
        params.append(regions[0])
    elif len(regions) > 1:
        # IN 절: region IN ('서울', '경기', '인천')
        placeholders = ', '.join(['%s'] * len(regions))
        conditions.append(f"structured_data->>'region' IN ({placeholders})")
        params.extend(regions)
    
    if not conditions:
        return "", []
    
    where_clause = " WHERE " + " AND ".join(conditions)
    return where_clause, params

def search_welcome_objective(keywords: list[str]) -> set[int]:
    """
    Welcome 테이블의 객관식 데이터를 PostgreSQL에서 검색
    
    Args:
        keywords: ["경기", "30대", "남자"] 같은 키워드 리스트
        
    Returns:
        pid(패널 ID) 집합
    """
    if not keywords:
        return set()
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return set()
        
        cur = conn.cursor()
        
        # 개선된 조건 빌더 사용
        where_clause, params = build_welcome_query_conditions(keywords)
        
        if not where_clause:
            print("⚠️  Welcome 객관식: 매칭되는 조건 없음")
            return set()
        
        query = f"SELECT pid FROM welcome_meta {where_clause}"
        
        print(f"🔍 Welcome 객관식 쿼리: {query}")
        print(f"   파라미터: {params}")
        
        cur.execute(query, tuple(params))
        results = {row[0] for row in cur.fetchall()}
        
        cur.close()
        print(f"✅ Welcome 객관식 검색 결과: {len(results)}개")
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
    """
    Welcome 주관식 + QPoll을 Qdrant에서 임베딩 검색
    (주관식 별도 테이블 없이 welcome_subjective_vectors 컬렉션 사용)
    
    Args:
        keywords: ["럭셔리", "소비"] 같은 추상적 키워드
        
    Returns:
        pid 집합
    """
    if not keywords:
        return set()
    
    try:
        embeddings = initialize_embeddings()
        qdrant_client = get_qdrant_client()
        
        if not qdrant_client:
            return set()
        
        # 키워드를 문장으로 결합
        query_text = " ".join(keywords)
        query_vector = embeddings.embed_query(query_text)
        
        # 환경변수에서 컬렉션 이름 가져오기
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "welcome_subjective_vectors")
        
        print(f"🔍 Welcome 주관식 Qdrant 검색: '{query_text}'")
        print(f"   컬렉션: {collection_name}")
        
        # Qdrant 검색
        search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=1000,
            score_threshold=0.5
        )
        
        pids = {result.payload.get('pid') for result in search_results if result.payload.get('pid')}
        
        print(f"✅ Welcome 주관식 검색 결과: {len(pids)}개")
        return pids
        
    except Exception as e:
        print(f"❌ Welcome 주관식 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        return set()

def search_qpoll(survey_type: str, keywords: list[str]) -> set[int]:
    """
    QPoll을 Qdrant에서 임베딩 검색
    (QPoll도 welcome_subjective_vectors 컬렉션에 함께 저장됨)
    
    Args:
        survey_type: "lifestyle", "consumption" 등
        keywords: 검색할 키워드 리스트
        
    Returns:
        pid 집합
    """
    if not keywords:
        return set()
    
    try:
        embeddings = initialize_embeddings()
        qdrant_client = get_qdrant_client()
        
        if not qdrant_client:
            return set()
        
        # 키워드를 문장으로 결합
        query_text = " ".join(keywords)
        query_vector = embeddings.embed_query(query_text)
        
        # 같은 컬렉션 사용
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "welcome_subjective_vectors")
        
        print(f"🔍 QPoll 검색 - 유형: {survey_type}, 키워드: '{query_text}'")
        print(f"   컬렉션: {collection_name}")
        
        # survey_type이 있으면 필터 적용 (Qdrant payload에 survey_type 필드가 있는 경우)
        if survey_type:
            try:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="survey_type",
                            match={"value": survey_type}
                        )
                    ]
                )
                
                search_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=qdrant_filter,
                    limit=1000,
                    score_threshold=0.5
                )
            except Exception as filter_error:
                # 필터 적용 실패 시 (payload에 survey_type 없는 경우) 필터 없이 검색
                print(f"   ⚠️  필터 적용 불가, 전체 검색으로 전환: {filter_error}")
                search_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=1000,
                    score_threshold=0.5
                )
        else:
            # survey_type이 없으면 필터 없이 검색
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=1000,
                score_threshold=0.5
            )
        
        pids = {result.payload.get('pid') for result in search_results if result.payload.get('pid')}
        
        print(f"✅ QPoll 검색 결과: {len(pids)}개")
        return pids
        
    except Exception as e:
        print(f"❌ QPoll 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        return set()

def hybrid_search(classified_keywords: dict, search_mode: str = "all") -> dict:
    """
    분류된 키워드로 전체 하이브리드 검색 수행
    
    Args:
        classified_keywords: classify_query_keywords() 결과
        search_mode: 검색 모드
            - "all": 3가지 모드 모두 반환 (기본값, 추천!)
            - "intersection": 교집합만
            - "union": 합집합만
            - "weighted": 가중치 기반만
        
    Returns:
        {
            "pid1": set,
            "pid2": set,
            "pid3": set,
            "results": {
                "intersection": {...},
                "union": {...},
                "weighted": {...}
            }
        }
    """
    print("\n" + "="*60)
    print(f"🚀 하이브리드 검색 시작 (모드: {search_mode})")
    print("="*60)
    
    # 1. Welcome 정형 조건
    welcome_obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
    pid1 = search_welcome_objective(welcome_obj_keywords)
    
    # 2. Welcome 비정형 조건
    welcome_subj_keywords = classified_keywords.get('welcome_keywords', {}).get('subjective', [])
    pid2 = search_welcome_subjective(welcome_subj_keywords)
    
    # 3. QPoll 검색
    qpoll_data = classified_keywords.get('qpoll_keywords', {})
    survey_type = qpoll_data.get('survey_type')
    qpoll_keywords = qpoll_data.get('keywords', [])
    pid3 = search_qpoll(survey_type, qpoll_keywords)
    
    # 4. 비어있지 않은 집합들만 모음
    all_sets = [s for s in [pid1, pid2, pid3] if s]
    
    results = {}
    
    # 5-1. 교집합 (Intersection) 계산
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
    
    # 5-2. 합집합 (Union) 계산
    if not all_sets:
        union_pids = []
        union_scores = {}
    else:
        union_set = set.union(*all_sets)
        union_scores = {}
        
        for pid in union_set:
            score = sum([
                1 if pid in pid1 else 0,
                1 if pid in pid2 else 0,
                1 if pid in pid3 else 0
            ])
            union_scores[pid] = score
        
        # 매칭 개수 높은 순으로 정렬
        union_pids = sorted(union_set, key=lambda x: union_scores[x], reverse=True)
    
    results['union'] = {
        'pids': union_pids,
        'count': len(union_pids),
        'scores': union_scores
    }
    
    # 5-3. 가중치 (Weighted) 계산
    weights = {
        'pid1': 0.4,  # Welcome 객관식
        'pid2': 0.3,  # Welcome 주관식
        'pid3': 0.3   # QPoll
    }
    
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
        
        # 가중치 점수 높은 순으로 정렬
        weighted_pids = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)
    
    results['weighted'] = {
        'pids': weighted_pids,
        'count': len(weighted_pids),
        'scores': weighted_scores,
        'weights': weights
    }
    
    # 6. 결과 출력
    print("\n" + "="*60)
    print("📊 검색 결과 요약")
    print("="*60)
    print(f"Welcome 객관식 (pid1): {len(pid1)}개")
    print(f"Welcome 주관식 (pid2): {len(pid2)}개")
    print(f"QPoll (pid3): {len(pid3)}개")
    print()
    print(f"🔹 교집합 (Intersection): {results['intersection']['count']}개")
    print(f"🔹 합집합 (Union): {results['union']['count']}개")
    print(f"🔹 가중치 (Weighted): {results['weighted']['count']}개")
    
    # 각 모드별 상위 3개 출력
    print("\n📈 각 모드별 상위 3개 PID:")
    print("-" * 60)
    
    for mode_name, mode_data in results.items():
        if mode_data['pids']:
            print(f"\n[{mode_name.upper()}]")
            for pid in mode_data['pids'][:3]:
                score = mode_data['scores'][pid]
                sources = []
                if pid in pid1:
                    sources.append("객관식")
                if pid in pid2:
                    sources.append("주관식")
                if pid in pid3:
                    sources.append("QPoll")
                print(f"  PID {pid}: {score:.2f} ({', '.join(sources)})")
    
    print("="*60 + "\n")
    
    # 7. 단일 모드 요청 시 해당 결과만 반환
    if search_mode in ['intersection', 'union', 'weighted']:
        return {
            "pid1": pid1,
            "pid2": pid2,
            "pid3": pid3,
            "final_result": results[search_mode]['pids'],
            "match_scores": results[search_mode]['scores']
        }
    
    # "all" 모드: 모든 결과 반환
    return {
        "pid1": pid1,
        "pid2": pid2,
        "pid3": pid3,
        "results": results
    }

# 테스트 코드
if __name__ == "__main__":
    test_classification = {
        "welcome_keywords": {
            "objective": ["경기", "30대", "남자"],
            "subjective": ["음주", "소비"]
        },
        "qpoll_keywords": {
            "survey_type": "consumption",
            "keywords": ["음주", "소주", "맥주"]
        }
    }
    
    result = hybrid_search(test_classification, search_mode="all")
    
    print("\n" + "="*60)
    print("🎯 최종 결과")
    print("="*60)
    print(f"교집합 PID 수: {result['results']['intersection']['count']}개")
    print(f"합집합 PID 수: {result['results']['union']['count']}개")
    print(f"가중치 PID 수: {result['results']['weighted']['count']}개")
    
    if result['results']['intersection']['pids']:
        print(f"\n교집합 상위 10개: {result['results']['intersection']['pids'][:10]}")
    if result['results']['weighted']['pids']:
        print(f"가중치 상위 10개: {result['results']['weighted']['pids'][:10]}")