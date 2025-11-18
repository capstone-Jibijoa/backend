import os
import psycopg2
import psycopg2.pool
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from threading import Lock
from contextlib import contextmanager

load_dotenv()

_connection_pool = None
_pool_lock = Lock()

def get_connection_pool():
    """
    싱글톤 패턴으로 PostgreSQL Connection Pool을 생성하고 관리합니다.
    """
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                try:
                    _connection_pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=5,
                        maxconn=20,
                        host=os.getenv("DB_HOST", "localhost"),
                        database=os.getenv("DB_NAME", "default_db"),
                        user=os.getenv("DB_USER", "postgres"),
                        password=os.getenv("DB_PASSWORD"),
                        connect_timeout=5,
                        options="-c statement_timeout=30000"
                    )
                    logging.info("PostgreSQL Connection Pool 생성 완료 (min: 5, max: 20)")
                except Exception as e:
                    logging.critical(f"Connection Pool 생성 실패: {e}")
                    _connection_pool = None
    
    return _connection_pool


def get_db_connection():
    """Connection Pool에서 연결을 가져옵니다."""
    try:
        pool = get_connection_pool()
        if pool is None:
            logging.error("Connection Pool이 초기화되지 않았습니다.")
            return None
        
        conn = pool.getconn()
        return conn
    except psycopg2.pool.PoolError as e:
        logging.error(f"Connection Pool의 모든 연결이 사용 중입니다: {e}")
        return None
    except Exception as e:
        logging.error(f"Connection Pool에서 연결을 가져오는 데 실패했습니다: {e}")
        return None


def return_db_connection(conn):
    """사용 완료된 연결을 Pool에 반환"""
    if conn:
        try:
            pool = get_connection_pool()
            if pool:
                pool.putconn(conn)
        except Exception as e:
            logging.warning(f"DB 연결을 Pool에 반환하는 데 실패했습니다: {e}")


@contextmanager
def get_db_connection_context():
    """
    Context Manager를 사용하여 DB 연결을 가져오고 자동으로 반환합니다.
    `with` 구문과 함께 사용하는 것을 권장합니다.
    """
    conn = get_db_connection()
    try:
        yield conn
    finally:
        return_db_connection(conn)


def close_connection_pool():
    """애플리케이션 종료 시 모든 유휴 연결을 닫습니다."""
    global _connection_pool
    
    if _connection_pool:
        with _pool_lock:
            if _connection_pool:
                _connection_pool.closeall()
                _connection_pool = None
                logging.info("PostgreSQL Connection Pool이 종료되었습니다.")


def get_qdrant_client():
    """Qdrant 클라이언트를 생성하고 반환합니다."""
    try:
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
            timeout=20.0
        )
        return client
    except Exception as e:
        logging.error(f"Qdrant 클라이언트 연결 실패: {e}")
        return None


def log_search_query(query: str, results_count: int, user_uid: int = None):
    """
    사용자 검색 쿼리와 결과 수를 데이터베이스에 기록합니다.
    """
    with get_db_connection_context() as conn:
        if not conn:
            return None
        
        try:
            cur = conn.cursor()
            
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'search_log'
                )
            """)
            
            table_exists = cur.fetchone()[0]
            
            if not table_exists:
                logging.warning("search_log 테이블이 없어 로그 기록을 건너뜁니다.")
                cur.close()
                return None
            
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
            
            return log_id
            
        except psycopg2.errors.InsufficientPrivilege:
            logging.warning("search_log 테이블에 대한 권한이 없어 로그 기록을 건너뜁니다.")
            if conn:
                conn.rollback()
            return None
            
        except Exception as e:
            logging.warning(f"검색 로그 기록 중 예외 발생: {e}")
            if conn:
                conn.rollback()
            return None

def init_db():
    """애플리케이션 시작 시 Connection Pool을 초기화합니다."""
    logging.info("DB 초기화 시작...")
    pool = get_connection_pool()
    if pool:
        logging.info("DB 초기화 완료.")
        return True
    else:
        logging.error("DB 초기화 실패.")
        return False

def cleanup_db():
    """애플리케이션 종료 시 Connection Pool을 정리합니다."""
    logging.info("DB 리소스 정리 중...")
    close_connection_pool()