# 데이터베이스 전문가로, 테이블을 관리하고 쿼리를 실행하여 데이터를 가져오는 일을 전담하는 파일
import os
import psycopg2
from dotenv import load_dotenv
# spl_queries.py 파일에서 새로운 쿼리들을 가져옵니다.
from spl_queries import (
    CREATE_PANELS_MASTER_TABLE, 
    CREATE_PANEL_VECTORS_TABLE, 
    CREATE_SEARCH_LOG_TABLE
)

# .env 파일에서 환경 변수를 불러옵니다.
load_dotenv()

def get_db_connection():
    """데이터베이스에 연결하고 연결 객체를 반환합니다."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return conn
    except psycopg2.Error as e:
        print(f"데이터베이스 연결 실패: {e}")
        return None
    
def create_tables():
    """데이터베이스에 필요한 모든 테이블을 생성합니다."""
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            # 기존 테이블 생성 쿼리를 새로운 ERD에 맞춰 수정합니다.
            cur.execute(CREATE_PANELS_MASTER_TABLE)
            cur.execute(CREATE_PANEL_VECTORS_TABLE)
            cur.execute(CREATE_SEARCH_LOG_TABLE)
            conn.commit()
            print("테이블이 성공적으로 생성되었습니다.")
            cur.close()
    except Exception as e:
        print(f"테이블 생성 실패: {e}")
    finally:
        if conn:
            conn.close()
            
def query_database_with_vector(embedding_vector: list[float], top_k: int = 5):
    """임베딩 벡터를 사용하여 데이터베이스에서 유사도 높은 패널을 검색합니다."""
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            # 'panel_vectors' 테이블에서 유사도 높은 uid를 검색
            # pgvector의 `<->` 연산자는 벡터 간의 거리를 계산합니다. 거리가 작을수록 유사도가 높습니다.
            # 1 - (거리)를 계산하여 유사도 점수를 만듭니다.
            # 벡터를 문자열 형태로 변환하여 쿼리에 삽입
            vector_str = str(embedding_vector)
            cur.execute(
                """SELECT uid, 1 - (embedding <=> %s) AS similarity 
                   FROM panel_vectors 
                   ORDER BY similarity DESC 
                   LIMIT %s""",
                (vector_str, top_k)
            )
            results = cur.fetchall()
            cur.close()
            # 결과를 딕셔너리 리스트로 변환
            return [{"uid": row[0], "similarity": row[1]} for row in results]
    except Exception as e:
        print(f"데이터베이스 쿼리 실패: {e}")
        return None
    finally:
        if conn:
            conn.close()

### 새로운 함수: 정형 조건과 임베딩 벡터를 모두 사용하여 하이브리드 검색을 수행
def query_database_with_hybrid_search(structured_condition: str, embedding_vector: list[float], top_k: int = 10):
    """
    정형 조건과 임베딩 벡터를 모두 사용하여 하이브리드 검색을 수행합니다.
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()

            # SQL 쿼리 기본 구조
            base_query = """
            SELECT 
                master.uid, 
                master.ai_insights,
                1 - (vectors.embedding <=> %s) AS similarity
            FROM 
                panels_master AS master
            JOIN 
                panel_vectors AS vectors ON master.uid = vectors.uid
            """
            
            # 정형 조건이 존재하면 WHERE 절을 추가합니다.
            where_clause = ""
            if structured_condition:
                where_clause = f" WHERE {structured_condition}"

            # 최종 쿼리 조합
            final_query = f"{base_query}{where_clause} ORDER BY similarity DESC LIMIT %s;"
            
            # psycopg2의 안전한 쿼리 실행을 위해 변수를 튜플로 전달합니다.
            cur.execute(final_query, (str(embedding_vector), top_k))
            results = cur.fetchall()
            cur.close()

            # 결과를 딕셔너리 리스트로 변환하여 반환
            return [{"uid": row[0], "ai_insights": row[1], "similarity": row[2]} for row in results]

    except Exception as e:
        print(f"하이브리드 검색 쿼리 실패: {e}")
        return None
    finally:
        if conn:
            conn.close()

def log_search_query(query: str, results_count: int, user_uid: int = None):
    """사용자 검색 쿼리와 결과 수를 로그 테이블에 저장합니다."""
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            # uid 필드 추가를 반영하여 쿼리를 수정합니다.
            cur.execute(
                "INSERT INTO search_log (query, results_count, uid) VALUES (%s, %s, %s) RETURNING id",
                (query, results_count, user_uid)
            )
            log_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            return log_id
    except Exception as e:
        print(f"검색 로그 기록 실패: {e}")
        return None
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    if get_db_connection():
        create_tables()
    else:
        print("데이터베이스 연결 실패로 인해 테이블을 생성할 수 없습니다.")