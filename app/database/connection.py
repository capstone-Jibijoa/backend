import logging
import psycopg2
import psycopg2.pool
from contextlib import contextmanager
from threading import Lock
from qdrant_client import QdrantClient

# 설정 파일 경로에 따라 수정 필요 (현재는 루트의 settings.py 사용 가정)
try:
    from app.core.config import settings
except ImportError:
    from settings import settings

_connection_pool = None
_pool_lock = Lock()

def get_connection_pool():
    """싱글톤 패턴으로 PostgreSQL Connection Pool 생성 및 반환"""
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                try:
                    _connection_pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=5,
                        maxconn=20,
                        host=settings.DB_HOST,
                        database=settings.DB_NAME,
                        user=settings.DB_USER,
                        password=settings.DB_PASSWORD,
                        connect_timeout=5,
                        options="-c statement_timeout=30000"
                    )
                    logging.info("✅ PostgreSQL Connection Pool 생성 완료")
                except Exception as e:
                    logging.critical(f"❌ Connection Pool 생성 실패: {e}")
                    _connection_pool = None
    return _connection_pool

def get_db_connection():
    """Pool에서 연결 객체 하나 가져오기"""
    try:
        pool = get_connection_pool()
        if pool:
            return pool.getconn()
        logging.error("Connection Pool 미초기화")
        return None
    except Exception as e:
        logging.error(f"DB 연결 가져오기 실패: {e}")
        return None

def return_db_connection(conn):
    """사용 완료된 연결 반환"""
    if conn:
        try:
            pool = get_connection_pool()
            if pool:
                pool.putconn(conn)
        except Exception as e:
            logging.warning(f"DB 연결 반환 실패: {e}")

@contextmanager
def get_db_connection_context():
    """Context Manager: with 구문용"""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        return_db_connection(conn)

def get_qdrant_client() -> QdrantClient:
    """Qdrant 클라이언트 생성"""
    try:
        return QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=20.0
        )
    except Exception as e:
        logging.error(f"❌ Qdrant 연결 실패: {e}")
        return None

def init_db():
    """앱 시작 시 초기화"""
    if get_connection_pool():
        logging.info("🚀 DB 초기화 완료")

def cleanup_db():
    """앱 종료 시 리소스 정리"""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        logging.info("🧹 DB 연결 종료 완료")