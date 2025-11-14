"""
Connection Pool을 사용한 db_logic.py 최적화 버전
- 매번 새 연결 생성 대신 연결 풀 재사용
- 동시 요청 처리 성능 향상
- 연결 생성/해제 오버헤드 제거
"""
import os
import psycopg2
import psycopg2.pool
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from threading import Lock
from contextlib import contextmanager

load_dotenv()

# =======================================================
# Connection Pool 관리
# =======================================================

_connection_pool = None
_pool_lock = Lock()


def get_connection_pool():
    """
    싱글톤 패턴으로 Connection Pool 관리
    
    개선점:
    - 애플리케이션 시작 시 한 번만 Pool 생성
    - 요청마다 연결 재사용 (빠른 응답)
    - 동시 요청 처리 가능 (max 20개)
    """
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_lock:
            # Double-checked locking
            if _connection_pool is None:
                try:
                    _connection_pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=5,   # 최소 연결 수 (항상 유지)
                        maxconn=20,  # 최대 연결 수 (피크 시간 대비)
                        host=os.getenv("DB_HOST"),
                        database=os.getenv("DB_NAME"),
                        user=os.getenv("DB_USER"),
                        password=os.getenv("DB_PASSWORD"),
                        # 추가 옵션
                        connect_timeout=5,              # 연결 타임아웃 5초
                        options="-c statement_timeout=30000"  # 쿼리 타임아웃 30초
                    )
                    print("✅ PostgreSQL Connection Pool 생성 완료 (5~20개 연결)")
                except Exception as e:
                    print(f"❌ Connection Pool 생성 실패: {e}")
                    _connection_pool = None
    
    return _connection_pool


def get_db_connection():
    """
    Connection Pool에서 연결 가져오기
    
    사용 예시:
        conn = get_db_connection()
        try:
            # ... 작업 수행
        finally:
            return_db_connection(conn)
    
    또는 context manager 사용:
        with get_db_connection_context() as conn:
            # ... 작업 수행
    """
    try:
        pool = get_connection_pool()
        if pool is None:
            print("❌ Connection Pool이 없습니다")
            return None
        
        conn = pool.getconn()
        return conn
    except psycopg2.pool.PoolError as e:
        print(f"❌ Connection Pool 고갈: {e}")
        return None
    except Exception as e:
        print(f"❌ 연결 획득 실패: {e}")
        return None


def return_db_connection(conn):
    """
    사용 완료된 연결을 Pool에 반환
    
    중요: 반드시 연결을 반환해야 Pool이 고갈되지 않음!
    """
    if conn:
        try:
            pool = get_connection_pool()
            if pool:
                pool.putconn(conn)
        except Exception as e:
            print(f"❌ 연결 반환 실패: {e}")


@contextmanager
def get_db_connection_context():
    """
    Context Manager로 자동 연결 반환
    
    추천 사용 방법:
        with get_db_connection_context() as conn:
            if conn:
                cur = conn.cursor()
                cur.execute("SELECT ...")
                # ... 작업 수행
                # conn.close() 불필요 - 자동 반환됨!
    """
    conn = get_db_connection()
    try:
        yield conn
    finally:
        return_db_connection(conn)


def close_connection_pool():
    """
    애플리케이션 종료 시 Connection Pool 닫기
    
    사용: 
        import atexit
        atexit.register(close_connection_pool)
    """
    global _connection_pool
    
    if _connection_pool:
        with _pool_lock:
            if _connection_pool:
                _connection_pool.closeall()
                _connection_pool = None
                print("✅ Connection Pool 종료 완료")


def get_pool_stats():
    """Connection Pool 통계 조회 (디버깅용)"""
    pool = get_connection_pool()
    if not pool:
        return {"status": "not_initialized"}
    
    try:
        # ThreadedConnectionPool은 _used, _pool 속성이 있음
        return {
            "status": "active",
            "minconn": pool.minconn,
            "maxconn": pool.maxconn,
            "current_used": len(pool._used),
            "available": len(pool._pool)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =======================================================
# Qdrant 클라이언트 (기존 유지)
# =======================================================

def get_qdrant_client():
    """Qdrant 클라이언트를 생성하고 반환합니다."""
    try:
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
            timeout=20.0      # 🔥 Timeout 설정 (기본 5초 → 20초로 증가)
        )
        print("✅ Qdrant 클라이언트 연결 성공")
        return client
    except Exception as e:
        print(f"❌ Qdrant 클라이언트 연결 실패: {e}")
        return None


# =======================================================
# 검색 로그 기록 (Connection Pool 사용)
# =======================================================

def log_search_query(query: str, results_count: int, user_uid: int = None):
    """
    검색 로그 기록 (Connection Pool 사용)
    
    개선점:
    - Connection Pool로 빠른 연결
    - with 문으로 자동 반환
    """
    with get_db_connection_context() as conn:
        if not conn:
            return None
        
        try:
            cur = conn.cursor()
            
            # 테이블 존재 확인
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
            
            # 로그 기록
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
            print(f"⚠️  검색 로그 기록 권한 없음 (무시하고 계속)")
            if conn:
                conn.rollback()
            return None
            
        except Exception as e:
            print(f"⚠️  검색 로그 기록 중 예외: {e} (무시하고 계속)")
            if conn:
                conn.rollback()
            return None


# =======================================================
# 예시: 최적화된 데이터 조회 함수
# =======================================================

def get_panels_data_optimized(panel_id_list: list, fields: list = None) -> list:
    """
    패널 데이터 조회 (Connection Pool + 필드 선택)
    
    Args:
        panel_id_list: 조회할 panel_id 리스트
        fields: 조회할 필드 리스트 (None이면 전체)
    
    개선점:
    - Connection Pool 사용
    - 필요한 필드만 조회
    - with 문으로 자동 연결 반환
    """
    if not panel_id_list:
        return []
    
    with get_db_connection_context() as conn:
        if not conn:
            return []
        
        try:
            cur = conn.cursor()
            
            if fields:
                # 특정 필드만 조회
                field_selects = ", ".join([
                    f"structured_data->>'{f}' as {f}" 
                    for f in fields
                ])
                query = f"""
                    SELECT panel_id, {field_selects}
                    FROM welcome_meta2 
                    WHERE panel_id = ANY(%s)
                """
            else:
                # 전체 조회
                query = """
                    SELECT panel_id, structured_data 
                    FROM welcome_meta2 
                    WHERE panel_id = ANY(%s)
                """
            
            cur.execute(query, (panel_id_list,))
            rows = cur.fetchall()
            
            panels_data = []
            for row in rows:
                if fields:
                    panel = {'panel_id': row[0]}
                    for i, field in enumerate(fields):
                        panel[field] = row[i + 1]
                else:
                    panel = {'panel_id': row[0]}
                    if isinstance(row[1], dict):
                        panel.update(row[1])
                panels_data.append(panel)
            
            cur.close()
            return panels_data
            
        except Exception as e:
            print(f"❌ 패널 데이터 조회 실패: {e}")
            return []


# =======================================================
# 애플리케이션 시작/종료 핸들러
# =======================================================

def init_db():
    """
    애플리케이션 시작 시 호출
    Connection Pool 초기화
    """
    print("🚀 DB 초기화 중...")
    pool = get_connection_pool()
    if pool:
        print("✅ DB 초기화 완료")
        return True
    else:
        print("❌ DB 초기화 실패")
        return False


def cleanup_db():
    """
    애플리케이션 종료 시 호출
    Connection Pool 정리
    """
    print("🧹 DB 정리 중...")
    close_connection_pool()


# =======================================================
# FastAPI 연동 예시
# =======================================================

# main.py에 추가:
# 
# from db_logic_optimized import init_db, cleanup_db
# 
# @app.on_event("startup")
# async def startup_event():
#     init_db()
# 
# @app.on_event("shutdown")
# async def shutdown_event():
#     cleanup_db()