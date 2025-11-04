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
# 2. 유틸리티 함수 (PostgreSQL WHERE 절 변환)
# =======================================================
def _build_jsonb_where_clause(structured_condition_json_str: str) -> tuple[str, list]:
    """
    Claude로부터 받은 JSON 문자열 필터를 PostgreSQL JSONB WHERE 절과 
    psycopg2 파라미터 리스트로 변환하여 SQL 인젝션을 방지합니다.
    (소득 필터 특별 처리 및 structured_data 참조)
    """
    try:
        filters = json.loads(structured_condition_json_str)
    except json.JSONDecodeError:
        print(f"경고: 잘못된 JSON 필터 수신: {structured_condition_json_str}")
        return "", []

    conditions = []
    params = []
    
    for f in filters:
        key = f.get("key")
        operator = f.get("operator")
        value = f.get("value")

        if not key or not operator or value is None:
            continue

        # ✅ [수정] 'ai_insights'가 아닌 'structured_data'를 참조합니다.
        jsonb_access = f"structured_data->>'{key}'"

        # ⭐️ [핵심] 소득(Income) 필터 인터셉터
        if key == "income_monthly" and operator in ["GT", "GTE", "LT", "LTE", "EQ"]:
            try:
                numeric_val = int(value) 
                sql_clause, params_tuple = _build_income_sql_clause(key, operator, numeric_val)
                # sql_clause에는 이미 (structured_data ->> %s ...)가 포함되어 있음
                conditions.append(sql_clause)
                params.extend(params_tuple)
            except (ValueError, TypeError):
                # 값이 숫자가 아닌 경우 (예: "월300~399만원")
                conditions.append(f" ({jsonb_access} = %s) ")
                params.append(value)
        
        # ⭐️ 그 외 모든 일반 필터
        elif operator == "EQ":
            conditions.append(f" {jsonb_access} = %s ")
            params.append(value)
        
        elif operator == "BETWEEN" and isinstance(value, list) and len(value) == 2:
            # 'int' 대신 'numeric'이 더 안전 (예: 출생연도)
            conditions.append(f" ({jsonb_access})::numeric BETWEEN %s AND %s ") 
            params.extend(value) 
            
        elif operator == "GT":
            conditions.append(f" ({jsonb_access})::numeric > %s ")
            params.append(value)
            
        elif operator == "LT":
            conditions.append(f" ({jsonb_access})::numeric < %s ")
            params.append(value)

        elif operator == "IN" and isinstance(value, list):
            # (이 로직은 프롬프트가 IN을 반환할 경우를 대비해 유지)
            if not value: continue
            placeholders = ", ".join(["%s"] * len(value))
            conditions.append(f" {jsonb_access} IN ({placeholders}) ")
            params.extend(value)

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