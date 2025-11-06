import os
import json
from dotenv import load_dotenv
from db_logic import get_db_connection, get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchAny
from langchain_community.embeddings import HuggingFaceEmbeddings

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
        
        # 키워드를 SQL 조건으로 변환 (예시 로직 - 실제 구조에 맞게 수정 필요)
        conditions = []
        params = []
        
        for keyword in keywords:
            # 성별 매칭
            if keyword.lower() in ['남자', '남성', '남']:
                conditions.append("gender = %s")
                params.append('M')
            elif keyword.lower() in ['여자', '여성', '여']:
                conditions.append("gender = %s")
                params.append('F')
            
            # 지역 매칭
            elif keyword in ['서울', '경기', '인천', '부산', '대구']:
                conditions.append("region = %s")
                params.append(keyword)
            
            # 나이대 매칭 (간단 버전 - 더 정교한 로직 필요)
            elif '대' in keyword:
                age = keyword.replace('대', '')
                if age.isdigit():
                    age_num = int(age)
                    current_year = 2025
                    birth_start = current_year - age_num - 9
                    birth_end = current_year - age_num
                    conditions.append("birth_year BETWEEN %s AND %s")
                    params.extend([birth_start, birth_end])
            
            # 결혼 상태
            elif keyword in ['미혼', '기혼', '이혼']:
                conditions.append("marital_status = %s")
                params.append(keyword)
        
        if not conditions:
            return set()
        
        where_clause = " AND ".join(conditions)
        query = f"SELECT pid FROM welcome WHERE {where_clause}"
        
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
    Welcome 테이블의 주관식 데이터를 Qdrant에서 임베딩 검색
    
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
        
        collection_name = os.getenv("QDRANT_WELCOME_COLLECTION", "welcome_subjective")
        
        print(f"🔍 Welcome 주관식 Qdrant 검색: '{query_text}'")
        
        # Qdrant 검색
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=1000,  # 충분히 많은 결과
            score_threshold=0.5  # 유사도 임계값
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
    QPoll 테이블에서 설문 유형별 임베딩 검색
    
    Args:
        survey_type: "lifestyle", "consumption" 등
        keywords: 검색할 키워드 리스트
        
    Returns:
        pid 집합
    """
    if not keywords or not survey_type:
        return set()
    
    try:
        embeddings = initialize_embeddings()
        qdrant_client = get_qdrant_client()
        
        if not qdrant_client:
            return set()
        
        # 키워드를 문장으로 결합
        query_text = " ".join(keywords)
        query_vector = embeddings.embed_query(query_text)
        
        # 설문 유형별 컬렉션 이름
        collection_name = os.getenv("QDRANT_QPOLL_COLLECTION", "qpoll_responses")
        
        print(f"🔍 QPoll 검색 - 유형: {survey_type}, 키워드: '{query_text}'")
        
        # 설문 유형 필터 적용
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="survey_type",
                    match={"value": survey_type}
                )
            ]
        )
        
        # Qdrant 검색
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
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

def hybrid_search(classified_keywords: dict) -> dict:
    """
    분류된 키워드로 전체 하이브리드 검색 수행
    
    Args:
        classified_keywords: classify_query_keywords() 결과
        
    Returns:
        {
            "pid1": set,  # Welcome 객관식 결과
            "pid2": set,  # Welcome 주관식 결과
            "pid3": set,  # QPoll 결과
            "intersection": set  # 교집합
        }
    """
    print("\n" + "="*60)
    print("🚀 하이브리드 검색 시작")
    print("="*60)
    
    # 1. Welcome 객관식 검색
    welcome_obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
    pid1 = search_welcome_objective(welcome_obj_keywords)
    
    # 2. Welcome 주관식 검색
    welcome_subj_keywords = classified_keywords.get('welcome_keywords', {}).get('subjective', [])
    pid2 = search_welcome_subjective(welcome_subj_keywords)
    
    # 3. QPoll 검색
    qpoll_data = classified_keywords.get('qpoll_keywords', {})
    survey_type = qpoll_data.get('survey_type')
    qpoll_keywords = qpoll_data.get('keywords', [])
    pid3 = search_qpoll(survey_type, qpoll_keywords)
    
    # 4. 교집합 계산
    all_sets = [s for s in [pid1, pid2, pid3] if s]
    
    if not all_sets:
        intersection = set()
    elif len(all_sets) == 1:
        intersection = all_sets[0]
    else:
        intersection = set.intersection(*all_sets)
    
    print("\n" + "="*60)
    print("📊 검색 결과 요약")
    print("="*60)
    print(f"Welcome 객관식 (pid1): {len(pid1)}개")
    print(f"Welcome 주관식 (pid2): {len(pid2)}개")
    print(f"QPoll (pid3): {len(pid3)}개")
    print(f"교집합: {len(intersection)}개")
    print("="*60 + "\n")
    
    return {
        "pid1": pid1,
        "pid2": pid2,
        "pid3": pid3,
        "intersection": intersection
    }

# 테스트 코드
if __name__ == "__main__":
    test_classification = {
        "welcome_keywords": {
            "objective": ["경기", "30대", "남자"],
            "subjective": ["럭셔리", "소비"]
        },
        "qpoll_keywords": {
            "survey_type": "consumption",
            "keywords": ["럭셔리", "고가", "프리미엄"]
        }
    }
    
    result = hybrid_search(test_classification)
    print("\n최종 결과:")
    print(f"교집합 PID 목록 (상위 10개): {list(result['intersection'])[:10]}")