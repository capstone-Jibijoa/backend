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
    
# DB에 저장된 실제 소득 문자열 목록 (순서대로)
INCOME_MAPPING = {
    "월100만원 미만": 0,
    "월100~199만원": 1000000,
    "월200~299만원": 2000000,
    "월300~399만원": 3000000,
    "월400~499만원": 4000000,
    "월500~599만원": 5000000,
    "월600만원 이상": 6000000
}

def _build_income_sql_clause(key: str, operator: str, value: int) -> (str, tuple):
    """
    숫자형 소득 쿼리(예: GT 600만)를 DB의 범주형 문자열 쿼리(예: IN '월600만원 이상')로 변환합니다.
    """
    target_categories = []
    
    if operator == "GT" or operator == "GTE": # 이상/초과
        target_categories = [cat for cat, min_val in INCOME_MAPPING.items() if min_val >= value]
        
    elif operator == "LT": # 미만 (<)
        # value(예: 200만)보다 "작은" 최소값을 가진 모든 카테고리
        target_categories = [cat for cat, min_val in INCOME_MAPPING.items() if min_val < value]
    elif operator == "LTE": # 이하 (<=)
        target_categories = [cat for cat, min_val in INCOME_MAPPING.items() if min_val <= value]

    elif operator == "EQ": # 같음 (예: 350만원 -> 월300~399만원)
        for cat, min_val in INCOME_MAPPING.items():
            if value >= min_val:
                target_categories = [cat] 
            else:
                break 

    if not target_categories:
        return " (1=0) ", () # 일치하는 카테고리 없음 (항상 False)

    # SQL "IN" 절 생성
    placeholders = ", ".join(["%s"] * len(target_categories))
    
    # ✅ [수정] 'ai_insights'가 아닌 'structured_data'를 참조합니다.
    sql_clause = f" (structured_data ->> %s IN ({placeholders})) "
    
    params = (key,) + tuple(target_categories)
    return sql_clause, params

# =======================================================
# 3. 검색 로그 기록 (권한 에러 처리 개선)
# =======================================================
def log_search_query(query: str, results_count: int, user_uid: int = None):
    """
    사용자의 검색 활동을 데이터베이스에 기록합니다.
    
    ✅ 개선: 권한 에러 시 조용히 실패 (서비스 중단 방지)
    
    Args:
        query: 검색 질의 텍스트
        results_count: 검색 결과 개수
        user_uid: 사용자 UID (선택)
        
    Returns:
        log_id: 기록된 로그의 ID (실패 시 None)
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cur = conn.cursor()
        
        # ✅ search_log 테이블이 있는지 먼저 확인
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'search_log'
            )
        """)
        
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("⚠️  search_log 테이블이 존재하지 않습니다. 로그를 건너뜁니다.")
            cur.close()
            return None
        
        # 로그 기록 시도
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
        
        # 성공 시에만 출력 (조용히)
        # print(f"✅ 검색 로그 기록 완료 (ID: {log_id})")
        return log_id
        
    except psycopg2.errors.InsufficientPrivilege as e:
        # ✅ 권한 에러는 무시하고 계속 진행
        print(f"⚠️  검색 로그 기록 권한 없음 (무시하고 계속)")
        if conn:
            conn.rollback()
        return None
        
    except psycopg2.Error as e:
        # ✅ 다른 DB 에러도 조용히 처리
        print(f"⚠️  검색 로그 기록 실패: {e} (무시하고 계속)")
        if conn:
            conn.rollback()
        return None
        
    except Exception as e:
        # ✅ 예상치 못한 에러도 조용히 처리
        print(f"⚠️  검색 로그 기록 중 예외: {e} (무시하고 계속)")
        if conn:
            conn.rollback()
        return None
        
    finally:
        if conn:
            conn.close()

# =======================================================
# 4. search_log 테이블 생성 (옵션)
# =======================================================

def create_search_log_table():
    """
    search_log 테이블이 없으면 생성합니다.
    권한이 있을 때만 사용하세요.
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ DB 연결 실패")
            return False
        
        cur = conn.cursor()
        
        # 테이블 생성
        cur.execute("""
            CREATE TABLE IF NOT EXISTS search_log (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                results_count INTEGER,
                uid INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # 인덱스 생성
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_search_log_created_at 
            ON search_log(created_at)
        """)
        
        conn.commit()
        cur.close()
        
        print("✅ search_log 테이블 생성 완료")
        return True
        
    except psycopg2.errors.InsufficientPrivilege:
        print("❌ 테이블 생성 권한이 없습니다")
        if conn:
            conn.rollback()
        return False
        
    except Exception as e:
        print(f"❌ 테이블 생성 실패: {e}")
        if conn:
            conn.rollback()
        return False
        
    finally:
        if conn:
            conn.close()