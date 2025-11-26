import logging
from collections import Counter
from typing import List, Dict, Any, Tuple
from db import get_db_connection_context 
import datetime

from mapping_rules import WELCOME_OBJECTIVE_FIELDS, QPOLL_FIELDS, FIELD_NAME_MAP

def calculate_age_from_birth_year(birth_year, current_year: int = None) -> int:
    """출생연도로부터 나이 계산 (현재 연도 동적 적용)"""
    if current_year is None:
        current_year = datetime.datetime.now().year
        
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
    """데이터에서 특정 필드의 값들을 추출 (analysis.py에서 사용)"""
    values = []

    if field_name == "birth_year":
        values = [get_age_group(item.get(field_name, 0)) for item in data if item.get(field_name)]
    else:
        raw_values = [item.get(field_name) for item in data if item.get(field_name)]
        for val in raw_values:
            if isinstance(val, list):
                values.extend(val)
            elif val is not None:
                values.append(val)

    return [v for v in values if v is not None]


def get_panels_data_from_db(panel_id_list: List[str]) -> List[Dict]:
    """
    panel_id 리스트로부터 패널 데이터 조회 (Connection Pool 사용)
    """
    if not panel_id_list:
        return []
    
    panels_data = []
    try:
        with get_db_connection_context() as conn:
            if not conn:
                return []
            cur = conn.cursor()

            cur.execute(
                """
                WITH id_order (panel_id, ordering) AS (
                    SELECT * FROM unnest(%s::text[], %s::int[])
                )
                SELECT t.panel_id, t.structured_data
                FROM welcome_meta2 t
                JOIN id_order o ON t.panel_id = o.panel_id
                ORDER BY o.ordering;
                """,
                (panel_id_list, list(range(len(panel_id_list))))
            )
            rows = cur.fetchall()
            
            for panel_id, structured_data in rows:
                if isinstance(structured_data, dict):
                    panel = {'panel_id': panel_id}
                    panel.update(structured_data)
                    panels_data.append(panel)
            
            cur.close()
        return panels_data
    except Exception as e:
        logging.error(f"패널 데이터 조회 실패: {e}", exc_info=True)
        return []