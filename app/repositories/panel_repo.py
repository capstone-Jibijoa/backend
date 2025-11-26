import logging
from typing import List, Dict, Set, Any
from app.database.connection import get_db_connection_context
from datetime import datetime

class PanelRepository:
    def get_panels_by_ids(self, panel_ids: List[str]) -> List[Dict]:
        """패널 ID 리스트로 데이터 조회 (순서 보장)"""
        if not panel_ids: return []
        results = []
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    # unnest를 사용하여 입력 순서대로 조회
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

    def search_by_structure_filters(self, filters: List[Dict]) -> Set[str]:
        """SQL 기반 구조적 필터링"""
        if not filters:
            return set()
            
        panel_ids = set()
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    where_clauses = []
                    params = []
                    current_year = datetime.now().year

                    for f in filters:
                        field = f['field']
                        op = f['operator']
                        val = f['value']

                        # 나이 계산 로직
                        if field == 'age':
                            db_field = f"({current_year} - COALESCE((structured_data->>'birth_year')::int, 0))"
                        else:
                            db_field = f"structured_data->>'{field}'"

                        if op == 'eq':
                            where_clauses.append(f"{db_field} = %s")
                            params.append(str(val))
                        elif op == 'in':
                            placeholders = ','.join(['%s'] * len(val))
                            where_clauses.append(f"{db_field} IN ({placeholders})")
                            params.extend([str(v) for v in val])
                        elif op == 'between':
                            where_clauses.append(f"{db_field}::numeric BETWEEN %s AND %s")
                            params.extend(val)
                        elif op == 'gte':
                            where_clauses.append(f"{db_field}::numeric >= %s")
                            params.append(val)
                        elif op == 'lte':
                            where_clauses.append(f"{db_field}::numeric <= %s")
                            params.append(val)
                        elif op == 'not_null':
                            where_clauses.append(f"{db_field} IS NOT NULL")

                    if not where_clauses:
                        return set()

                    query = f"SELECT panel_id FROM welcome_meta2 WHERE {' AND '.join(where_clauses)}"
                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()
                    panel_ids = {row[0] for row in rows}
        except Exception as e:
            logging.error(f"구조적 필터 검색 실패: {e}")
            
        return panel_ids

    def get_field_distribution(self, field_name: str, limit: int = 50) -> Dict[str, float]:
        """
        특정 필드의 전체 분포(%)를 DB에서 집계 (차트용)
        """
        distribution = {}
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    # 1. 연령대 계산 (birth_year)
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
                            )
                            SELECT age_group, COUNT(*), ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)
                            FROM age_groups GROUP BY age_group ORDER BY 3 DESC LIMIT {limit}
                        """
                    
                    # 2. 자녀 수 (숫자 + '명' 처리)
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
                    
                    # 3. 일반 필드 집계
                    else:
                        query = f"""
                            SELECT structured_data->>'{field_name}', COUNT(*), ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)
                            FROM welcome_meta2 WHERE structured_data->>'{field_name}' IS NOT NULL
                            GROUP BY 1 ORDER BY 3 DESC LIMIT {limit}
                        """

                    cur.execute(query)
                    rows = cur.fetchall()
                    
                    # {값: 비율} 형태의 딕셔너리로 변환
                    distribution = {row[0]: float(row[2]) for row in rows if row[0]}

        except Exception as e:
            logging.error(f"필드 분포 집계 실패 ({field_name}): {e}")
        
        return distribution