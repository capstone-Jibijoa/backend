"""
공통 유틸리티 함수 모음
"""
from collections import Counter
from typing import List, Dict, Any, Tuple
import re
from db_logic import get_db_connection
from typing import Dict, List, Any # 상단에 추가
import datetime

def get_db_distribution(field_name: str) -> Dict[str, float]:
    """
    DB에서 직접 필드 분포를 집계합니다.
    """
    conn = None
    total_count = 0
    distribution = {}
    
    try:
        conn = get_db_connection()
        if not conn:
            return {}
        cur = conn.cursor()

        query = ""
        total_count_query = ""
        
        # 1. 필드별로 SQL 쿼리 분기
        if field_name == "birth_year":
            # 1-1. 'birth_year' (나이) 처리
            current_year = datetime.datetime.now().year # ✅ 동적 연도 계산
            
            total_count_query = "SELECT COUNT(*) FROM welcome_meta2 WHERE structured_data->>'birth_year' ~ '^[0-9]+$'"
            
            query = f"""
                SELECT 
                    CASE
                        WHEN ({current_year} - (structured_data->>'birth_year')::int) < 20 THEN '10대'
                        WHEN ({current_year} - (structured_data->>'birth_year')::int) < 30 THEN '20대'
                        WHEN ({current_year} - (structured_data->>'birth_year')::int) < 40 THEN '30대'
                        WHEN ({current_year} - (structured_data->>'birth_year')::int) < 50 THEN '40대'
                        WHEN ({current_year} - (structured_data->>'birth_year')::int) < 60 THEN '50대'
                        ELSE '60대 이상'
                    END as item,
                    COUNT(*) as count
                FROM welcome_meta2
                WHERE structured_data->>'birth_year' ~ '^[0-9]+$'
                GROUP BY item
                ORDER BY item
            """
        
        else:
            # 1-2. 'gender', 'region_major' 등 그 외 필드 처리
            total_count_query = f"SELECT COUNT(*) FROM welcome_meta2 WHERE structured_data->>'{field_name}' IS NOT NULL"
            
            query = f"""
                SELECT 
                    structured_data->>'{field_name}' as item, 
                    COUNT(*) as count
                FROM welcome_meta2
                WHERE structured_data->>'{field_name}' IS NOT NULL
                GROUP BY item
                ORDER BY count DESC
            """
        
        # 2. 쿼리 실행 (공통 로직)
        
        # 2-1. 전체 카운트
        cur.execute(total_count_query)
        total_count = cur.fetchone()[0]
        
        if total_count == 0:
            cur.close()
            return {}

        # 2-2. 분포 쿼리
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        
        # 3. 백분율로 변환
        distribution = {str(item): round((count / total_count) * 100, 1) for item, count in rows}
        return distribution
        
    except Exception as e:
        # field_name을 포함하여 에러 로그 출력
        print(f"❌ DB 집계 실패 ({field_name}): {e}")
        return {}
    finally:
        if conn:
            conn.close()

def build_sql_conditions_from_keywords(keywords: List[str], current_year: int = 2025) -> Tuple[str, List[Any]]:
    """키워드 리스트를 SQL WHERE 조건으로 변환"""
    conditions = []
    params = []
    regions_major = []
    regions_minor = []
    
    for keyword in keywords:
        kw = keyword.strip().lower()
        
        if kw in ['남자', '남성', '남']:
            conditions.append(
                "(structured_data->>'gender' IS NOT NULL "
                "AND structured_data->>'gender' = %s)"
            )
            params.append('M')
        elif kw in ['여자', '여성', '여']:
            conditions.append(
                "(structured_data->>'gender' IS NOT NULL "
                "AND structured_data->>'gender' = %s)"
            )
            params.append('F')
        elif keyword in ['서울', '경기', '인천', '부산', '대구', '대전', '광주', '울산', '세종',
                        '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']:
            regions_major.append(keyword)
        elif keyword.endswith(('시', '구', '군')):
            regions_minor.append(keyword)
        
        elif '대' in keyword and keyword[:-1].isdigit():
            age_prefix = int(keyword[:-1])
            birth_start = current_year - age_prefix - 9
            birth_end = current_year - age_prefix
            conditions.append(
                "(structured_data->>'birth_year' IS NOT NULL "
                "AND (structured_data->>'birth_year')::int BETWEEN %s AND %s)"
            )
            params.extend([birth_start, birth_end])
    
    if len(regions_major) == 1:
        conditions.append("(structured_data->>'region_major' = %s)")
        params.append(regions_major[0])
    elif len(regions_major) > 1:
        placeholders = ','.join(['%s'] * len(regions_major))
        conditions.append(f"(structured_data->>'region_major' IN ({placeholders}))")
        params.extend(regions_major)
    
    if len(regions_minor) == 1:
        conditions.append("(structured_data->>'region_minor' = %s)")
        params.append(regions_minor[0])
    elif len(regions_minor) > 1:
        placeholders = ','.join(['%s'] * len(regions_minor))
        conditions.append(f"(structured_data->>'region_minor' IN ({placeholders}))")
        params.extend(regions_minor)
    
    if not conditions:
        return "", []
    
    return " WHERE " + " AND ".join(conditions), params

def calculate_age_from_birth_year(birth_year, current_year: int = 2025) -> int:
    """출생연도로부터 나이 계산"""
    try:
        return current_year - int(birth_year)
    except:
        return 0

def get_age_group(birth_year) -> str:
    """출생연도로부터 연령대 반환"""
    age = calculate_age_from_birth_year(birth_year)
    if age < 20: return "10대"
    elif age < 30: return "20대"
    elif age < 40: return "30대"
    elif age < 50: return "40대"
    elif age < 60: return "50대"
    else: return "60대 이상"

def calculate_distribution(values: List[Any]) -> Dict[str, float]:
    """값 리스트의 분포를 백분율로 계산"""
    if not values:
        return {}
    counter = Counter(values)
    total = len(values)
    return {k: round((v / total) * 100, 1) for k, v in counter.most_common()}

def find_top_category(distribution: Dict[str, float]) -> Tuple[str, float]:
    """분포에서 가장 높은 비율의 카테고리와 비율 반환"""
    if not distribution:
        return ("없음", 0.0)
    return max(distribution.items(), key=lambda x: x[1])

def extract_field_values(data: List[Dict], field_name: str) -> List[Any]:
    """데이터에서 특정 필드의 값들을 추출"""
    values = []
    
    if field_name == "birth_year":
        values = [get_age_group(item.get(field_name, 0)) for item in data if item.get(field_name)]
    elif field_name == "region_major":
        values = [item.get("region_major") for item in data if item.get("region_major")]
    elif field_name == "region_minor":
        values = [item.get("region_minor") for item in data if item.get("region_minor")]
    else:
        values = [item.get(field_name) for item in data if item.get(field_name)]
    
    return values

WELCOME_OBJECTIVE_FIELDS = [
    ("gender", "성별"),
    ("birth_year", "연령대"),
    ("region_major", "거주 지역"),
    ("region_minor", "세부 거주 지역"),
    ("marital_status", "결혼 여부"),
    ("job_title_raw", "직업"),
    ("income_personal_monthly", "개인 소득"),
]

FIELD_NAME_MAP = dict(WELCOME_OBJECTIVE_FIELDS)

# ✅ 분석에서 제외할 raw 필드 (데이터가 너무 다양하거나 품질이 낮음)
EXCLUDED_RAW_FIELDS = {
    
}

def get_all_panels_data_from_db(limit: int = None) -> List[Dict]:
    """전체 패널 데이터 조회 (panel_id는 문자열)"""
    from db_logic import get_db_connection
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return []
        cur = conn.cursor()
        
        query = f"SELECT panel_id, structured_data FROM welcome_meta2"
        if limit:
            query += f" LIMIT {limit}"
        
        cur.execute(query)
        rows = cur.fetchall()
        
        panels_data = []
        for panel_id, structured_data in rows:
            if isinstance(structured_data, dict):
                panel = {'panel_id': panel_id}
                panel.update(structured_data)
                panels_data.append(panel)
        
        cur.close()
        return panels_data
    except Exception as e:
        print(f"❌ 전체 패널 데이터 조회 실패: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_panels_data_from_db(panel_id_list: List[str]) -> List[Dict]:
    """panel_id 리스트로부터 패널 데이터 조회 (panel_id는 문자열)"""
    if not panel_id_list:
        return []
    
    from db_logic import get_db_connection
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return []
        cur = conn.cursor()
        
        placeholders = ','.join(['%s'] * len(panel_id_list))
        query = f"SELECT panel_id, structured_data FROM welcome_meta2 WHERE panel_id IN ({placeholders})"
        
        cur.execute(query, tuple(panel_id_list))
        rows = cur.fetchall()
        
        panels_data = []
        for panel_id, structured_data in rows:
            if isinstance(structured_data, dict):
                panel = {'panel_id': panel_id}
                panel.update(structured_data)
                panels_data.append(panel)
        
        cur.close()
        return panels_data
    except Exception as e:
        print(f"❌ 패널 데이터 조회 실패: {e}")
        return []
    finally:
        if conn:
            conn.close()