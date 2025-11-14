import logging
import datetime
from collections import Counter
from typing import List, Dict, Any, Tuple
from db import get_db_connection_context # 수정된 import
import datetime

# 이 맵은 analysis.py와 utils.py 모두에서 사용됩니다.
WELCOME_OBJECTIVE_FIELDS = [
    ("gender", "성별"),
    ("birth_year", "연령대"),
    ("region_major", "거주 지역"),
    ("region_minor", "세부 거주 지역"),
    ("marital_status", "결혼 여부"),
    ("children_count","자녀수"),
    ("family_size", "가족 수"),
    ("education_level", "최종학력"),
    ("job_title_raw", "직업"),
    ("job_duty_raw", "직무"),
    ("income_personal_monthly", "월소득(개인)"),
    ("income_household_monthly", "월소득(가구)"),
    ("phone_brand_raw", "휴대폰 브랜드"),
    ("phone_model_raw", "휴대폰 모델"),
    ("car_ownership", "차량 보유 여부"),
    ("car_manufacturer_raw", "차량 제조사"),
    ("car_model_raw", "차량 모델명"),
    ("smoking_experience", "흡연 여부"),
    ("smoking_brand", "담배 종류"),
    ("smoking_brand_etc_raw", "기타 담배 종류"),
    ("e_cigarette_experience", "전자 담배 이용 경험"),
    ("smoking_brand_other_details_raw", "기타 흡연 세부 사항"),
    ("drinking_experience", "음주 경험"),
    ("drinking_experience_other_details_raw", "음주 세부 사항")
]
FIELD_NAME_MAP = dict(WELCOME_OBJECTIVE_FIELDS)


def get_db_distribution(field_name: str) -> Dict[str, float]:
    """
    [삭제 가능성 검토]
    DB에서 직접 필드 분포를 집계합니다. (analysis.py의 함수와 중복됨)
    만약 analysis.py 외 다른 곳에서 이 함수를 쓰지 않는다면 삭제해도 됩니다.
    """
    total_count = 0
    distribution = {}
    
    try:
        # Connection Pool 사용
        with get_db_connection_context() as conn:
            if not conn:
                return {}
            cur = conn.cursor()

            query = ""
            total_count_query = ""
            
            if field_name == "birth_year":
                current_year = datetime.datetime.now().year
                total_count_query = "SELECT COUNT(*) FROM welcome_meta2 WHERE structured_data->>'birth_year' ~ '^[0-9]+$'"
                query = f"""
                    SELECT CASE
                        WHEN ({current_year} - (structured_data->>'birth_year')::int) < 20 THEN '10대'
                        WHEN ({current_year} - (structured_data->>'birth_year')::int) < 30 THEN '20대'
                        WHEN ({current_year} - (structured_data->>'birth_year')::int) < 40 THEN '30대'
                        WHEN ({current_year} - (structured_data->>'birth_year')::int) < 50 THEN '40대'
                        WHEN ({current_year} - (structured_data->>'birth_year')::int) < 60 THEN '50대'
                        ELSE '60대 이상'
                    END as item, COUNT(*) as count
                    FROM welcome_meta2
                    WHERE structured_data->>'birth_year' ~ '^[0-9]+$'
                    GROUP BY item ORDER BY item
                """
            else:
                total_count_query = f"SELECT COUNT(*) FROM welcome_meta2 WHERE structured_data->>'{field_name}' IS NOT NULL"
                query = f"""
                    SELECT structured_data->>'{field_name}' as item, COUNT(*) as count
                    FROM welcome_meta2
                    WHERE structured_data->>'{field_name}' IS NOT NULL
                    GROUP BY item ORDER BY count DESC
                """
            
            cur.execute(total_count_query)
            total_count = cur.fetchone()[0]
            
            if total_count == 0:
                cur.close()
                return {}

            cur.execute(query)
            rows = cur.fetchall()
            cur.close()
        
        distribution = {str(item): round((count / total_count) * 100, 1) for item, count in rows}
        return distribution
        
    except Exception as e:
        logging.error(f"DB 집계 실패 ({field_name}): {e}", exc_info=True)
        return {}
    finally:
        pass # Context manager가 연결을 닫음


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
    """데이터에서 특정 필드의 값들을 추출 (analysis.py에서 사용)"""
    values = []

    if field_name == "birth_year":
        values = [get_age_group(item.get(field_name, 0)) for item in data if item.get(field_name)]
    else:
        raw_values = [item.get(field_name) for item in data if item.get(field_name)]
        for val in raw_values:
            if isinstance(val, list):
                # 값이 리스트이면, 각 원소를 개별 값으로 추가
                values.extend(val)
            elif val is not None:
                # 리스트가 아니면 값을 그대로 추가
                values.append(val)

    return [v for v in values if v is not None]


def get_panels_data_from_db(panel_id_list: List[str]) -> List[Dict]:
    """
    [analysis.py에서 사용]
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
            
            # 정렬된 결과를 순서대로 추가
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