import logging
from typing import List, Dict, Set, Any, Tuple
from app.database.connection import get_db_connection_context
from datetime import datetime
from app.utils.common import extract_birth_year_from_raw

from app.constants.mapping import (
    CATEGORY_MAPPING,
    FIELD_ALIAS_MAP,
    VALUE_TRANSLATION_MAP,
    FUZZY_MATCH_FIELDS,
    ARRAY_FIELDS
)

class PanelRepository:
    def get_panel_detail(self, panel_id: str) -> Dict[str, Any]:
        """패널 1명의 상세 정보 조회"""
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    query = "SELECT panel_id, structured_data FROM welcome_meta2 WHERE panel_id = %s"
                    cur.execute(query, (panel_id,))
                    row = cur.fetchone()
                    if row:
                        pid, data = row
                        result = {"panel_id": pid}
                        if data:
                            result.update(data)
                        return result
        except Exception as e:
            logging.error(f"패널 상세 조회 실패 ({panel_id}): {e}")
        return {}

    def get_panels_by_ids(self, panel_ids: List[str]) -> List[Dict]:
        """패널 ID 리스트로 데이터 조회 (단순 조회용)"""
        if not panel_ids: return []
        results = []
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT t.panel_id, t.structured_data
                        FROM welcome_meta2 t
                        JOIN unnest(%s::text[]) WITH ORDINALITY AS o(pid, ord) ON t.panel_id = o.pid
                        ORDER BY o.ord
                    """
                    cur.execute(query, (panel_ids,))
                    for pid, data in cur.fetchall():
                        if data:
                            row = {"panel_id": pid}
                            row.update(data)
                            results.append(row)
        except Exception as e:
            logging.error(f"패널 조회 실패: {e}")
        return results

    # 분석 서비스를 위한 데이터 통합 조회 메서드
    def get_panels_data_from_db(self, panel_id_list: List[str]) -> List[Dict]:
        """
        panel_id 리스트를 받아 welcome_meta2와 qpoll_meta 데이터를 병합하여 반환합니다.
        """
        if not panel_id_list:
            return []
        
        panels_data = []
        try:
            with get_db_connection_context() as conn:
                if not conn: return []
                cur = conn.cursor()

                query = """
                    WITH id_order (panel_id, ordering) AS (
                        SELECT * FROM unnest(%s::text[], %s::int[])
                    )
                    SELECT 
                        t.panel_id, 
                        t.structured_data,
                        q.category,
                        q.gender,
                        q.age_raw,
                        q.region
                    FROM welcome_meta2 t
                    JOIN id_order o ON t.panel_id = o.panel_id
                    LEFT JOIN qpoll_meta q ON t.panel_id = q.panel_id
                    ORDER BY o.ordering;
                """
                
                cur.execute(query, (panel_id_list, list(range(len(panel_id_list)))))
                rows = cur.fetchall()
                
                for row in rows:
                    panel = {'panel_id': row[0]}
                    
                    # 1. Welcome 데이터 병합
                    if isinstance(row[1], dict):
                        panel.update(row[1])
                    
                    # 2. Qpoll 데이터 병합 (우선순위 데이터)
                    if row[2]: panel['category'] = row[2]
                    if row[3]: panel['gender'] = row[3] 
                    if row[5]: panel['region'] = row[5]

                    # 3. 나이 데이터 처리
                    if row[4]: 
                        panel['birth_year'] = extract_birth_year_from_raw(row[4])
                        panel['age_raw_text'] = row[4]

                    panels_data.append(panel)
                
                cur.close()
            return panels_data
            
        except Exception as e:
            logging.error(f"패널 데이터 조회 실패: {e}", exc_info=True)
            return []

    def search_by_structure_filters(self, filters: List[Dict]) -> Set[str]:
        """
        welcome_meta2와 qpoll_meta를 LEFT JOIN하여 검색 범위를 확장
        - 예: '강남구' 검색 시 qpoll_meta.region에서 히트되도록 함
        """
        if not filters:
            return set()
            
        panel_ids = set()
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    where_clause, params = self._build_sql_conditions(filters)
                    
                    if not where_clause:
                        return set()

                    # LEFT JOIN 추가 및 Alias t 사용
                    query = f"""
                        SELECT t.panel_id 
                        FROM welcome_meta2 t 
                        LEFT JOIN qpoll_meta q ON t.panel_id = q.panel_id 
                        {where_clause}
                    """
                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()
                    panel_ids = {row[0] for row in rows}
                    
                    logging.info(f"   [DB] 필터링 SQL 실행 완료: {len(panel_ids)}건 발견")
        except Exception as e:
            logging.error(f"구조적 필터 검색 실패: {e}")
            
        return panel_ids

    def _build_sql_conditions(self, filters: List[Dict]) -> Tuple[str, List]:
        """
        테이블 Alias(t, q)를 적용하고 Region 검색 로직을 강화한 SQL 생성기
        """
        conditions = []
        params = []
        CURRENT_YEAR = datetime.now().year

        # 소득 범위 정의
        INCOME_RANGES = [
            (0, 999999, "월 100만원 미만"),
            (1000000, 1999999, "월 100~199만원"),
            (2000000, 2999999, "월 200~299만원"),
            (3000000, 3999999, "월 300~399만원"),
            (4000000, 4999999, "월 400~499만원"),
            (5000000, 5999999, "월 500~599만원"),
            (6000000, 6999999, "월 600~699만원"),
            (7000000, 999999999, "월 700만원 이상")
        ]

        for f in filters:
            raw_field = f.get("field")
            operator = f.get("operator")
            value = f.get("value")

            if not raw_field or not operator: continue

            field = FIELD_ALIAS_MAP.get(raw_field, raw_field)

            # 1. 지역(region) 필터 처리: welcome(t) OR qpoll(q)
            if field in ['region', 'region_major']:
                if isinstance(value, list):
                    or_conditions = []
                    for v in value:
                        # q.region(강남구 등 상세) 또는 t.structured_data(서울 등 광역) 검색
                        or_conditions.append(f"(q.region ILIKE %s OR t.structured_data->>'region_major' ILIKE %s)")
                        params.extend([f"%{v}%", f"%{v}%"])
                    if or_conditions:
                        conditions.append(f"({' OR '.join(or_conditions)})")
                else:
                    conditions.append(f"(q.region ILIKE %s OR t.structured_data->>'region_major' ILIKE %s)")
                    params.extend([f"%{value}%", f"%{value}%"])
                continue

            # 2. not_null (부정어 제외 로직) -> t.structured_data 사용
            if operator == "not_null":
                base_condition = f"(t.structured_data->>'{field}' IS NOT NULL AND t.structured_data->>'{field}' != 'NaN')"
                exclude_pattern = ""
                
                if field == "children_count":
                    conditions.append(f"({base_condition} AND t.structured_data->>'{field}' NOT IN ('0', '0명') AND t.structured_data->>'{field}' !~ '없음')")
                    continue
                
                if field == "drinking_experience": exclude_pattern = "마시지|않음|없음|비음주|금주|안\\s*마심|전혀"
                elif field == "smoking_experience": exclude_pattern = "피우지|않음|없음|비흡연|금연|안\\s*피움"
                elif field == "ott_count": exclude_pattern = "0개|안\\s*함|없음|이용\\s*안|보지\\s*않음"
                elif field == "fast_delivery_usage": exclude_pattern = "안\\s*함|이용\\s*안|없음|직접\\s*구매"
                else: exclude_pattern = "없음|비흡연|해당사항|피우지|금연"

                conditions.append(f"({base_condition} AND t.structured_data->>'{field}' !~ '{exclude_pattern}')")
                continue

            # 3. 나이 계산 -> t.structured_data 사용
            if field == "birth_year" or raw_field == "age":
                if operator == "between" and isinstance(value, list) and len(value) == 2:
                    age_start, age_end = value
                    birth_year_end = CURRENT_YEAR - age_start
                    birth_year_start = CURRENT_YEAR - age_end
                    conditions.append(f"(t.structured_data->>'birth_year')::int BETWEEN %s AND %s")
                    params.extend([birth_year_start, birth_year_end])
                continue

            # 4. 값 매핑 (Mapping Rule)
            final_value = value
            if field in VALUE_TRANSLATION_MAP:
                mapping = VALUE_TRANSLATION_MAP[field]
                if isinstance(value, list):
                    converted_list = []
                    for v in value:
                        mapped_v = mapping.get(v, v)
                        if isinstance(mapped_v, list): converted_list.extend(mapped_v)
                        else: converted_list.append(mapped_v)
                    final_value = converted_list
                else:
                    mapped_v = mapping.get(value, value)
                    final_value = mapped_v

            if isinstance(final_value, list):
                expanded_list = []
                for v in final_value:
                    if str(v) in CATEGORY_MAPPING: expanded_list.extend(CATEGORY_MAPPING[str(v)])
                    else: expanded_list.append(v)
                final_value = expanded_list
            elif str(final_value) in CATEGORY_MAPPING:
                final_value = CATEGORY_MAPPING[str(final_value)]

            # 5. Fuzzy Match -> t.structured_data 사용
            if field in FUZZY_MATCH_FIELDS or field in ARRAY_FIELDS:
                if not isinstance(final_value, list): final_value = [final_value]
                
                or_conditions = []
                for v in final_value:
                    or_conditions.append(f"t.structured_data->>'{field}' ILIKE %s")
                    params.append(f"%{v}%")
                
                if or_conditions:
                    conditions.append(f"({' OR '.join(or_conditions)})")

            # 6. 소득 필드 처리 -> t.structured_data 사용
            elif field in ["income_household_monthly", "income_personal_monthly"] and operator in ["gte", "lte", "between"]:
                target_categories = []
                min_val, max_val = 0, 999999999
                
                if operator == "gte": min_val = int(final_value)
                elif operator == "lte": max_val = int(final_value)
                elif operator == "between": min_val, max_val = int(final_value[0]), int(final_value[1])
                
                for r_min, r_max, label in INCOME_RANGES:
                    if max_val >= r_min and min_val <= r_max:
                        target_categories.append(label)
                
                if target_categories:
                    placeholders = ','.join(['%s'] * len(target_categories))
                    conditions.append(f"t.structured_data->>'{field}' IN ({placeholders})")
                    params.extend(target_categories)
                else:
                    conditions.append("1=0")

            # 7. 일반 처리 -> t.structured_data 사용
            else:
                field_sql = f"t.structured_data->>'{field}'"
                if operator == "eq":
                    conditions.append(f"{field_sql} = %s")
                    params.append(str(final_value))
                elif operator == "in" and isinstance(final_value, list) and final_value:
                    str_values = [str(v) for v in final_value]
                    placeholders = ','.join(['%s'] * len(str_values))
                    conditions.append(f"{field_sql} IN ({placeholders})")
                    params.extend(str_values)

        if not conditions: return "", []
        return " WHERE " + " AND ".join(conditions), params
    
    def get_field_distribution(self, field_name: str, limit: int = 50) -> Dict[str, float]:
        """필드 분포 집계"""
        distribution = {}
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    if field_name == "birth_year":
                        query = f"""
                            WITH age_groups AS (
                                SELECT 
                                    CASE 
                                        WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 20 THEN '10대'
                                        WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 30 THEN '20대'
                                        WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 40 THEN '30대'
                                        WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 50 THEN '40대'
                                        WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 60 THEN '50대'
                                        ELSE '60대 이상'
                                    END as age_group
                                FROM welcome_meta2
                                WHERE structured_data->>'birth_year' IS NOT NULL
                                  AND structured_data->>'birth_year' ~ '^\d{{4}}$'
                            )
                            SELECT age_group, COUNT(*), ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)
                            FROM age_groups GROUP BY age_group ORDER BY 3 DESC LIMIT {limit}
                        """
                    elif field_name == "children_count":
                        query = f"""
                            SELECT 
                                CONCAT((structured_data->>'{field_name}')::numeric::int, '명') as val, 
                                COUNT(*), ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)
                            FROM welcome_meta2
                            WHERE structured_data->>'{field_name}' IS NOT NULL
                            GROUP BY val ORDER BY 3 DESC LIMIT {limit}
                        """
                    else:
                        query = f"""
                            SELECT structured_data->>'{field_name}', COUNT(*), ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)
                            FROM welcome_meta2 WHERE structured_data->>'{field_name}' IS NOT NULL
                            GROUP BY 1 ORDER BY 3 DESC LIMIT {limit}
                        """
                    
                    cur.execute(query)
                    distribution = {row[0]: float(row[2]) for row in cur.fetchall() if row[0]}
        except Exception as e:
            logging.error(f"필드 분포 집계 실패 ({field_name}): {e}")
        return distribution