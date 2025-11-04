# db_logic.py (최종 버전: JSONB 쿼리 및 안전한 파라미터 전달)

import os
import psycopg2
from dotenv import load_dotenv
import json
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance, NamedVector
from qdrant_client.models import Filter, FieldCondition, MatchAny, ScoredPoint

load_dotenv()

# =======================================================
# 0. Qdrant 클라이언트 초기화
# =======================================================
def get_qdrant_client():
    """Qdrant 클라이언트를 생성하고 반환합니다."""
    try:
        # 환경 변수에서 Qdrant 호스트와 포트를 가져옵니다.
        client = QdrantClient(host=os.getenv("QDRANT_HOST"), port=os.getenv("QDRANT_PORT"))
        print("Qdrant 클라이언트 연결 성공!")
        return client
    except Exception as e:
        print(f"Qdrant 클라이언트 연결 실패: {e}")
        return None
# =======================================================
# 1. DB 연결 및 테이블 생성 함수 (로직 유지)
# =======================================================
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

# =======================================================
# 2. 유틸리티 함수 (PostgreSQL WHERE 절 변환)
# =======================================================
def _build_jsonb_where_clause(structured_condition_json_str: str) -> tuple[str, list]:
    """
    Claude로부터 받은 JSON 문자열 필터를 PostgreSQL JSONB WHERE 절과 
    psycopg2 파라미터 리스트로 변환하여 SQL 인젝션을 방지합니다.
    """
    try:
        filters = json.loads(structured_condition_json_str)
    except json.JSONDecodeError:
        return "", []

    conditions = []
    params = []
    
    for f in filters:
        key = f.get("key")
        operator = f.get("operator")
        value = f.get("value")

        if not key or not operator or value is None:
            continue

        # 모든 필터는 panels_master의 ai_insights JSONB 컬럼을 참조합니다.
        jsonb_access = f"structured_data->>'{key}'"

        if operator == "EQ":
            conditions.append(f"{jsonb_access} = %s")
            params.append(value)
        
        elif operator == "BETWEEN" and isinstance(value, list) and len(value) == 2:
            # 숫자로 명시적 캐스팅이 필요하며, BETWEEN은 두 개의 %s 파라미터가 필요합니다.
            conditions.append(f"({jsonb_access})::int BETWEEN %s AND %s")
            params.extend(value) # 리스트의 두 요소를 파라미터에 추가
            
        elif operator == "GT":
             conditions.append(f"({jsonb_access})::int > %s")
             params.append(value)
             
        elif operator == "LT":
             conditions.append(f"({jsonb_access})::int < %s")
             params.append(value)
            
        # NOTE: JSONB의 값은 텍스트(->>)로 추출되므로, 숫자 비교 시 ::int로 캐스팅합니다.

    if not conditions:
        return "", []

    # 모든 조건을 AND로 연결하고 ' WHERE ' 구문 추가
    return " WHERE " + " AND ".join(conditions), params

# =======================================================
# 4. 검색 로그 기록 함수
# =======================================================
def log_search_query(query: str, results_count: int, user_uid: int = None):
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
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