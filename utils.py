"""
공통 유틸리티 함수 모음
"""
from collections import Counter
from typing import List, Dict, Any, Tuple
import re

def build_sql_conditions_from_keywords(keywords: List[str], current_year: int = 2025) -> Tuple[str, List[Any]]:
    """키워드 리스트를 SQL WHERE 조건으로 변환"""
    conditions = []
    params = []
    regions = []
    
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
            regions.append(keyword)
        elif '대' in keyword and keyword[:-1].isdigit():
            age_prefix = int(keyword[:-1])
            birth_start = current_year - age_prefix - 9
            birth_end = current_year - age_prefix
            conditions.append(
                "(structured_data->>'birth_year' IS NOT NULL "
                "AND (structured_data->>'birth_year')::int BETWEEN %s AND %s)"
            )
            params.extend([birth_start, birth_end])
    
    if len(regions) == 1:
        conditions.append("(structured_data->>'region_minor' = %s)")
        params.append(regions[0])
    elif len(regions) > 1:
        placeholders = ','.join(['%s'] * len(regions))
        conditions.append(f"(structured_data->>'region_minor' IN ({placeholders}))")
        params.extend(regions)
    
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
    elif field_name == "region":
        values = [item.get("region_minor") for item in data if item.get("region_minor")]
    else:
        values = [item.get(field_name) for item in data if item.get(field_name)]
    
    return values

WELCOME_OBJECTIVE_FIELDS = [
    ("gender", "성별"),
    ("birth_year", "연령대"),
    ("region_minor", "거주 지역"),
    ("marital_status", "결혼 여부"),
    ("job_title_raw", "직업"),
    ("income_personal_monthly", "개인 소득"),
]

FIELD_NAME_MAP = dict(WELCOME_OBJECTIVE_FIELDS)

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