import os
import psycopg2
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

# =======================================================
# 1. Qdrant 클라이언트
# =======================================================

def get_qdrant_client():
    """Qdrant 클라이언트를 생성하고 반환합니다."""
    try:
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        print("✅ Qdrant 클라이언트 연결 성공")
        return client
    except Exception as e:
        print(f"❌ Qdrant 클라이언트 연결 실패: {e}")
        return None

# =======================================================
# 2. PostgreSQL 연결
# =======================================================

def get_db_connection():
    """PostgreSQL 데이터베이스에 연결하고 연결 객체를 반환합니다."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return conn
    except psycopg2.Error as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return None

# =======================================================
# 3. 검색 로그 기록
# =======================================================

def log_search_query(query: str, results_count: int, user_uid: int = None):
    """
    사용자의 검색 활동을 데이터베이스에 기록합니다.
    
    Args:
        query: 검색 질의 텍스트
        results_count: 검색 결과 개수
        user_uid: 사용자 UID (선택)
        
    Returns:
        log_id: 기록된 로그의 ID
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO search_log (query, results_count, uid, created_at) 
                VALUES (%s, %s, %s, NOW()) 
                RETURNING id
                """,
                (query, results_count, user_uid)
            )
            log_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            print(f"✅ 검색 로그 기록 완료 (ID: {log_id})")
            return log_id
    except Exception as e:
        print(f"❌ 검색 로그 기록 실패: {e}")
        return None
    finally:
        if conn:
            conn.close()

# =======================================================
# 5. Welcome 객관식 조건 빌더 (개선된 버전)
# =======================================================

def build_welcome_query_conditions(keywords: list[str]) -> tuple[str, list]:
    """
    키워드 리스트를 받아 WHERE 절과 파라미터를 생성합니다.
    
    Args:
        keywords: ["경기", "30대", "남자"] 같은 키워드
        
    Returns:
        (where_clause, params): SQL WHERE 절과 파라미터 튜플
    """
    conditions = []
    params = []
    current_year = 2025
    
    for keyword in keywords:
        kw = keyword.strip().lower()
        
        # 성별
        if kw in ['남자', '남성', '남']:
            conditions.append("gender = %s")
            params.append('M')
        elif kw in ['여자', '여성', '여']:
            conditions.append("gender = %s")
            params.append('F')
        
        # 지역
        elif keyword in ['서울', '경기', '인천', '부산', '대구', '대전', '광주', '울산', '세종']:
            conditions.append("region = %s")
            params.append(keyword)
        
        # 나이대 (예: 20대, 30대)
        elif '대' in keyword and keyword[:-1].isdigit():
            age_prefix = int(keyword[:-1])
            birth_start = current_year - age_prefix - 9
            birth_end = current_year - age_prefix
            conditions.append("birth_year BETWEEN %s AND %s")
            params.extend([birth_start, birth_end])
        
        # 결혼 상태
        elif kw in ['미혼', '싱글']:
            conditions.append("marital_status = %s")
            params.append('미혼')
        elif kw in ['기혼', '결혼']:
            conditions.append("marital_status = %s")
            params.append('기혼')
        elif kw in ['이혼', '돌싱']:
            conditions.append("marital_status = %s")
            params.append('이혼')
        
        # 음주
        elif kw in ['술먹는', '음주']:
            conditions.append("drinking_experience = %s")
            params.append('경험 있음')
        elif kw in ['술안먹는', '금주']:
            conditions.append("drinking_experience = %s")
            params.append('경험 없음')
        
        # 흡연
        elif kw in ['흡연', '담배']:
            conditions.append("smoking_experience = %s")
            params.append('경험 있음')
        elif kw in ['비흡연', '금연']:
            conditions.append("smoking_experience = %s")
            params.append('경험 없음')
        
        # 차량 보유
        elif kw in ['차있음', '자가용', '차량보유']:
            conditions.append("car_ownership = %s")
            params.append('보유')
        elif kw in ['차없음']:
            conditions.append("car_ownership = %s")
            params.append('미보유')
    
    if not conditions:
        return "", []
    
    where_clause = " WHERE " + " AND ".join(conditions)
    return where_clause, params

# =======================================================
# 테스트 코드
# =======================================================

if __name__ == "__main__":
    print("데이터베이스 연결 테스트...")
    
    # PostgreSQL 연결 테스트
    conn = get_db_connection()
    if conn:
        print("✅ PostgreSQL 연결 성공")
        
        conn.close()
    else:
        print("❌ PostgreSQL 연결 실패")
    
    # Qdrant 연결 테스트
    qdrant = get_qdrant_client()
    if qdrant:
        print("✅ Qdrant 연결 성공")
    else:
        print("❌ Qdrant 연결 실패")
    
    # 조건 빌더 테스트
    print("\n조건 빌더 테스트:")
    test_keywords = ["경기", "30대", "남자", "미혼"]
    where, params = build_welcome_query_conditions(test_keywords)
    print(f"WHERE 절: {where}")
    print(f"파라미터: {params}")