import logging
import psycopg2
from typing import Optional
from app.database.connection import get_db_connection_context

class LogRepository:
    @staticmethod
    def log_search(query: str, results_count: int, user_uid: int = None) -> Optional[int]:
        """검색 로그를 DB에 기록합니다."""
        try:
            with get_db_connection_context() as conn:
                if not conn: return None
                
                with conn.cursor() as cur:
                    # 테이블 존재 여부 체크 (필요하다면 유지, 아니면 제거 가능)
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'search_log'
                        )
                    """)
                    if not cur.fetchone()[0]:
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
                    # commit은 context manager 밖에서 수행되거나 자동 수행되어야 함
                    # 여기서는 명시적 커밋을 수행 (autocommit 설정에 따라 다름)
                    conn.commit() 
                    return log_id
                    
        except psycopg2.errors.InsufficientPrivilege:
            logging.warning("search_log 권한 부족")
            return None
        except Exception as e:
            logging.warning(f"로그 기록 실패: {e}")
            return None