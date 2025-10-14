# 데이터베이스 전문가로, 테이블을 관리하고 쿼리를 실행하여 데이터를 가져오는 일을 전담하는 파일
import os
import psycopg2
from dotenv import load_dotenv
from spl_queries import CREATE_CLEAN_DATA_TABLE, CREATE_EMBEDDINGS_TABLE, CREATE_SEARCH_LOG_TABLE

# .env 파일에서 환경 변수를 불러옵니다.
load_dotenv()

def get_db_connection():
    """데이터베이스에 연결하고 연결 객체를 반환."""
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
    """데이터베이스에 필요한 모든 테이블을 생성."""
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute(CREATE_CLEAN_DATA_TABLE)
            cur.execute(CREATE_EMBEDDINGS_TABLE)
            cur.execute(CREATE_SEARCH_LOG_TABLE) # 검색 로그 테이블 생성 추가
            conn.commit()
            print("테이블이 성공적으로 생성되었습니다.")
            cur.close()
    except Exception as e:
        print(f"테이블 생성 실패: {e}")
    finally:
        if conn:
            conn.close()

def query_database_with_vector(embedding_vector: list[float], top_k: int = 5):
    """임베딩 벡터를 사용하여 데이터베이스에서 유사도 높은 상품을 검색."""
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            # pgvector의 코사인 유사도(<=|>) 연산자를 사용하여 검색
            # 벡터를 문자열 형태로 변환하여 쿼리에 삽입
            vector_str = str(embedding_vector)
            cur.execute(
                """SELECT product_id, 1 - (embedding <=> %s) AS similarity 
                   FROM embeddings 
                   ORDER BY similarity DESC 
                   LIMIT %s""",
                (vector_str, top_k)
            )
            results = cur.fetchall()
            cur.close()
            # 결과를 딕셔너리 리스트로 변환
            return [{"product_id": row[0], "similarity": row[1]} for row in results]
    except Exception as e:
        print(f"데이터베이스 쿼리 실패: {e}")
        return None
    finally:
        if conn:
            conn.close()

def log_search_query(query: str, results_count: int):
    """사용자 검색 쿼리와 결과 수를 로그 테이블에 저장."""
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO search_log (query, results_count) VALUES (%s, %s) RETURNING id",
                (query, results_count)
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
