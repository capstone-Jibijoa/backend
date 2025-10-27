# db_logic.py (최종 버전: JSONB 쿼리 및 안전한 파라미터 전달)

import os
import psycopg2
from dotenv import load_dotenv
import json

from spl_queries import (
    CREATE_PANELS_MASTER_TABLE, 
    CREATE_PANEL_VECTORS_TABLE, 
    CREATE_SEARCH_LOG_TABLE
)

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

# -----------------------------------------------------------
# 헬퍼 함수: JSON 문자열 필터를 안전한 SQL WHERE 절로 변환합니다.
# -----------------------------------------------------------
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
        jsonb_access = f"master.ai_insights->>'{key}'"

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


def query_database_with_hybrid_search(structured_condition_json_str: str, embedding_vector: list[float], top_k: int = 10):
    """
    정형 조건(JSON 문자열)과 임베딩 벡터를 모두 사용하여 하이브리드 검색을 수행합니다.
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()

            # 1. JSON 필터를 안전한 SQL WHERE 절과 파라미터로 변환
            where_clause, where_params = _build_jsonb_where_clause(structured_condition_json_str)
            
            # 2. 하이브리드 쿼리 템플릿 완성
            vector_str = str(embedding_vector)
            
            # 최종 쿼리 문자열 구성 (WHERE 절 포함)
            final_query = f"""
                SELECT 
                    master.uid, 
                    master.ai_insights,
                    1 - (vectors.embedding <=> %s) AS similarity
                FROM 
                    panels_master AS master
                JOIN 
                    panel_vectors AS vectors ON master.uid = vectors.uid
                {where_clause}
                ORDER BY similarity DESC 
                LIMIT %s;
            """
            
            # 3. 모든 파라미터를 하나의 튜플로 조합
            # 순서: [벡터 문자열] + [WHERE 절 파라미터 (동적)] + [LIMIT 파라미터]
            all_params = [vector_str] + where_params + [top_k]
            
            # 4. 쿼리 실행
            cur.execute(final_query, tuple(all_params))
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