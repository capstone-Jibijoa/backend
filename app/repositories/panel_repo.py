import logging
from typing import List, Dict, Set, Any
from app.database.connection import get_db_connection_context
from datetime import datetime

class PanelRepository:
    def get_panels_by_ids(self, panel_ids: List[str]) -> List[Dict]:
        """패널 ID 리스트로 데이터 조회"""
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

    def search_by_structure_filters(self, filters: List[Dict]) -> Set[str]:
        """
        SQL 기반 구조적 필터링
        """
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

                        # 1. 컬럼 매핑 (나이 계산 포함)
                        if field == 'age':
                            # (현재연도 - birth_year)
                            db_field = f"({current_year} - COALESCE((structured_data->>'birth_year')::int, 0))"
                        else:
                            # JSON 필드 접근
                            db_field = f"structured_data->>'{field}'"

                        # 2. 연산자 처리
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
                    logging.debug(f"SQL 실행: {query} / Params: {params}")
                    
                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()
                    panel_ids = {row[0] for row in rows}
                    
        except Exception as e:
            logging.error(f"구조적 필터 검색 실패: {e}", exc_info=True)
            
        return panel_ids