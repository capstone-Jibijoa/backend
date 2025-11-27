# app/repositories/panel_repo.py

import logging
import re
from typing import List, Dict, Set, Any, Tuple
from app.database.connection import get_db_connection_context
from datetime import datetime
from app.utils.common import extract_birth_year_from_raw
from app.constants.mapping import FIELD_ALIAS_MAP, VALUE_TRANSLATION_MAP, FUZZY_MATCH_FIELDS, ARRAY_FIELDS, CATEGORY_MAPPING

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
        """패널 ID 리스트로 데이터 조회 (welcome 우선, qpoll 보조)"""
        if not panel_ids: return []
        results = []
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT 
                            t.panel_id, 
                            t.structured_data,
                            q.category,
                            q.gender,
                            q.age_raw,
                            q.region
                        FROM welcome_meta2 t
                        JOIN unnest(%s::text[]) WITH ORDINALITY AS o(pid, ord) ON t.panel_id = o.pid
                        LEFT JOIN qpoll_meta q ON t.panel_id = q.panel_id
                        ORDER BY o.ord
                    """
                    cur.execute(query, (panel_ids,))
                
                    for row in cur.fetchall():
                        pid = row[0]
                        panel = {"panel_id": pid}
                    
                        # 1. Welcome 데이터 병합 (기본값)
                        if row[1] and isinstance(row[1], dict):
                            panel.update(row[1])

                        if 'region_minor' in panel:
                            logging.debug(f"   [Data] ID({pid[:10]}...) region_minor: {panel['region_minor']}")
                    
                        # 2. ✅ Qpoll 데이터 병합 (welcome에 없는 것만 추가)
                        if row[2]:  # category
                            panel['category'] = row[2]
                    
                        # ✅ 성별: welcome 우선 (welcome에 없으면 qpoll 변환)
                        if 'gender' not in panel or not panel['gender']:
                            if row[3]:  # qpoll gender
                                gender_map = {'남': 'M', '여': 'F'}
                                panel['gender'] = gender_map.get(row[3], row[3])
                    
                        # ✅ 지역: welcome 우선 (welcome에 없으면 qpoll에서 추출)
                        if 'region_major' not in panel or not panel['region_major']:
                            if row[5]:  # qpoll region
                                # "서울 중구" → "서울"
                                panel['region_major'] = row[5].split(' ')[0] if ' ' in row[5] else row[5]
                    
                        # ✅ 나이: welcome 우선 (welcome에 없으면 qpoll age_raw에서 추출)
                        if 'birth_year' not in panel or not panel['birth_year']:
                            if row[4]:  # qpoll age_raw
                                extracted_year = extract_birth_year_from_raw(row[4])
                                if extracted_year > 1900:
                                    panel['birth_year'] = extracted_year
                    
                        results.append(panel)
                    
        except Exception as e:
            logging.error(f"패널 조회 실패: {e}")
        return results

    def get_panels_data_from_db(self, panel_id_list: List[str]) -> List[Dict]:
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
        
                        -- ✅ welcome 데이터 우선, 없으면 qpoll 사용
                        COALESCE(t.structured_data->>'gender', 
                                CASE q.gender 
                                    WHEN '남' THEN 'M' 
                                    WHEN '여' THEN 'F' 
                                    ELSE q.gender 
                                 END) as gender,
        
                        COALESCE(
                            (t.structured_data->>'birth_year')::int,
                            NULLIF(REGEXP_REPLACE(q.age_raw, '[^0-9]', '', 'g'), '')::int
                        ) as birth_year,
        
                        COALESCE(
                            t.structured_data->>'region_major',
                            SPLIT_PART(q.region, ' ', 1)
                        ) as region_major
        
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
                    
                    # 2. Qpoll 데이터 병합
                    if row[2]: panel['category'] = row[2]
                    if row[3]: panel['gender'] = row[3] 
                    if row[5]: panel['region'] = row[5]

                    # 3. 나이 데이터 처리
                    if row[4]: 
                        extracted_year = extract_birth_year_from_raw(row[4])
                        if extracted_year > 1900:
                            panel['birth_year'] = extracted_year
                        panel['age_raw_text'] = row[4]

                    panels_data.append(panel)
                
                cur.close()
            return panels_data
            
        except Exception as e:
            logging.error(f"패널 데이터 조회 실패: {e}", exc_info=True)
            return []

    def search_by_structure_filters(self, filters: List[Dict]) -> Set[str]:
        if not filters:
            return set()
            
        panel_ids = set()
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    where_clause, params = self._build_sql_conditions(filters)
                    
                    if not where_clause:
                        return set()

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
        conditions = []
        params = []
        CURRENT_YEAR = datetime.now().year

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

        VALID_SQL_FIELDS = [
            'region', 'region_major', 'gender', 'age', 'birth_year', 
            'marital_status', 'family_size', 'children_count', 
            'job_title_raw', 'income_household_monthly', 'car_ownership',
            'income_personal_monthly', 'drinking_experience', 'smoking_experience'
        ]

        for f in filters:
            raw_field = f.get("field")
            operator = f.get("operator")
            value = f.get("value")
            
            field = FIELD_ALIAS_MAP.get(raw_field, raw_field)

            if field not in VALID_SQL_FIELDS and field not in VALUE_TRANSLATION_MAP:
                continue

            # [값 매핑 및 역방향 확장]
            normalized_set = set()
            if field in VALUE_TRANSLATION_MAP:
                mapping = VALUE_TRANSLATION_MAP[field]
                input_values = value if isinstance(value, list) else [value]
                
                for v in input_values:
                    norm = mapping.get(v, v)
                    if isinstance(norm, list): normalized_set.update(norm)
                    else: normalized_set.add(norm)
            else:
                if isinstance(value, list): normalized_set.update(value)
                else: normalized_set.add(value)

            final_value_set = set(normalized_set)
            if field in VALUE_TRANSLATION_MAP:
                mapping = VALUE_TRANSLATION_MAP[field]
                for raw_k, norm_v in mapping.items():
                    is_match = False
                    if isinstance(norm_v, list):
                        if any(nv in normalized_set for nv in norm_v): is_match = True
                    else:
                        if norm_v in normalized_set: is_match = True
                    
                    if is_match:
                        final_value_set.add(raw_k)
            
            final_value = list(final_value_set)

            if isinstance(final_value, list):
                expanded_list = []
                for v in final_value:
                    if str(v) in CATEGORY_MAPPING: expanded_list.extend(CATEGORY_MAPPING[str(v)])
                    else: expanded_list.append(v)
                final_value = expanded_list
            elif str(final_value) in CATEGORY_MAPPING:
                final_value = CATEGORY_MAPPING[str(final_value)]

            # -------------------------------------------------------------
            # [CRITICAL FIX] 리스트 형태의 단일 값 처리 (e.g., [1] -> 1)
            val_to_use = final_value
            if isinstance(final_value, list) and len(final_value) == 1:
                val_to_use = final_value[0]
            # -------------------------------------------------------------

            # 1. 지역(region) 처리 - 정확한 매칭 로직
            if field in ['region', 'region_major']:
                if isinstance(final_value, list):
                    or_conditions = []
                    for v in final_value:
                        # qpoll의 region은 "서울 중구" 형태이므로 앞부분만 추출
                        or_conditions.append(
                            f"(SPLIT_PART(q.region, ' ', 1) = %s OR t.structured_data->>'region_major' = %s)"
                        )
                        params.extend([v, v])
                    if or_conditions:
                        conditions.append(f"({' OR '.join(or_conditions)})")
                else:
                    conditions.append(
                        f"(SPLIT_PART(q.region, ' ', 1) = %s OR t.structured_data->>'region_major' = %s)"
                    )
                    params.extend([val_to_use, val_to_use])
                continue

            # 2. 성별(gender) 처리 - 값 정규화
            if field == 'gender':
                val_list = final_value if isinstance(final_value, list) else [final_value]
                gender_conditions = []
    
                # 정규화 매핑
                gender_map = {'남': 'M', '여': 'F', 'M': 'M', 'F': 'F', '남성': 'M', '여성': 'F'}
    
                for v in val_list:
                    norm_v = gender_map.get(str(v), str(v))
        
                    # qpoll은 "남", welcome은 "M" 형태
                    qpoll_val = '남' if norm_v == 'M' else ('여' if norm_v == 'F' else str(v))
        
                    gender_conditions.append(
                        f"(t.structured_data->>'gender' = %s OR q.gender = %s)"
                    )
                    params.extend([norm_v, qpoll_val])
    
                if gender_conditions:
                    conditions.append(f"({' OR '.join(gender_conditions)})")
                continue

            # 3. not_null 처리
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

            # 4. 나이(birth_year) 처리
            if field == "birth_year" or raw_field == "age":
                if operator == "between" and isinstance(value, list) and len(value) == 2:
                    age_start, age_end = value
                    birth_year_end = CURRENT_YEAR - age_start
                    birth_year_start = CURRENT_YEAR - age_end
                    
                    age_condition = f"""
                        (
                            COALESCE(
                                -- 1. welcome_meta2 JSONB의 birth_year를 정수화 (가장 신뢰도 높음)
                                NULLIF(regexp_replace(t.structured_data->>'birth_year', '[^0-9]', '', 'g'), '')::int,
                                -- 2. welcome_meta2에 없으면 qpoll_meta의 age_raw에서 연도 추출 (Fallback)
                                CASE WHEN q.age_raw ~ '^\\d{{4}}' THEN substring(q.age_raw, 1, 4)::int ELSE NULL END
                            ) BETWEEN %s AND %s
                        )
                    """
                    conditions.append(age_condition)
                    
                    # 이제 파라미터는 시작/끝 연도 두 개만 필요합니다.
                    params.extend([birth_year_start, birth_year_end])
                    continue

            # 5. Fuzzy Match
            if field in FUZZY_MATCH_FIELDS or field in ARRAY_FIELDS:
                if not isinstance(final_value, list): final_value = [final_value]
                or_conditions = []
                for v in final_value:
                    or_conditions.append(f"t.structured_data->>'{field}' ILIKE %s")
                    params.append(f"%{v}%")
                if or_conditions:
                    conditions.append(f"({' OR '.join(or_conditions)})")

            # 6. 소득 필드
            elif field in ["income_household_monthly", "income_personal_monthly"] and operator in ["gte", "lte", "between"]:
                target_categories = []
                min_val, max_val = 0, 999999999
                
                # val_to_use 사용
                if operator == "gte": min_val = int(val_to_use)
                elif operator == "lte": max_val = int(val_to_use)
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

            # 7. 숫자형 필드 처리 (family_size, children_count)
            elif field in ['family_size', 'children_count']:
                # 리스트 값이 들어올 경우 첫 번째 값 추출 (ex: [1] -> 1)
                val_to_use = final_value
                if isinstance(final_value, list) and len(final_value) == 1:
                    val_to_use = final_value[0]

                field_sql = f"t.structured_data->>'{field}'"
                
                # 문자열에서 숫자만 추출 (예: "4명" -> 4)
                numeric_extract_sql = f"NULLIF(REGEXP_REPLACE({field_sql}, '[^0-9]', '', 'g'), '')::int"
                
                if operator == "eq":
                    # val_to_use 사용
                    if str(val_to_use).isdigit():
                        conditions.append(f"{numeric_extract_sql} = %s")
                        params.append(int(val_to_use))
                    else:
                        logging.warning(f"⚠️ [SQL Build] '{field}' 필드 eq 조건 오류: {val_to_use}")

                elif operator == "in" and isinstance(final_value, list) and final_value:
                    valid_nums = [int(v) for v in final_value if str(v).isdigit()]
                    if valid_nums:
                        placeholders = ','.join(['%s'] * len(valid_nums))
                        conditions.append(f"{numeric_extract_sql} IN ({placeholders})")
                        params.extend(valid_nums)

                elif operator == "gte":
                    # val_to_use 사용
                    if str(val_to_use).isdigit():
                        conditions.append(f"{numeric_extract_sql} >= %s")
                        params.append(int(val_to_use))
                    else:
                        logging.warning(f"⚠️ [SQL Build] '{field}' gte 조건 값 오류: {val_to_use}")

                elif operator == "lte":
                    if str(val_to_use).isdigit():
                        conditions.append(f"{numeric_extract_sql} <= %s")
                        params.append(int(val_to_use))
                    else:
                        logging.warning(f"⚠️ [SQL Build] '{field}' lte 조건 값 오류: {val_to_use}")

                elif operator == "between" and isinstance(final_value, list) and len(final_value) == 2:
                    v1, v2 = final_value
                    if str(v1).isdigit() and str(v2).isdigit():
                        conditions.append(f"{numeric_extract_sql} BETWEEN %s AND %s")
                        params.extend([int(v1), int(v2)])

            # 8. 일반 처리
            else:
                field_sql = f"t.structured_data->>'{field}'"
                if operator == "eq":
                    conditions.append(f"{field_sql} = %s")
                    params.append(str(val_to_use))
                elif operator == "in" and isinstance(final_value, list) and final_value:
                    str_values = [str(v) for v in final_value]
                    placeholders = ','.join(['%s'] * len(str_values))
                    conditions.append(f"{field_sql} IN ({placeholders})")
                    params.extend(str_values)

        if not conditions: return "", []
        return " WHERE " + " AND ".join(conditions), params

    def get_field_distribution(self, field_name: str, limit: int = 50) -> Dict[str, float]:
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
                                COUNT(*) as count,
                                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
                            FROM welcome_meta2
                            WHERE structured_data->>'{field_name}' IS NOT NULL
                            GROUP BY val ORDER BY percentage DESC LIMIT {limit}
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