import logging
from typing import List, Dict, Any, Optional
from app.database.connection import get_db_connection_context  # db.py가 이동했다고 가정

class PanelRepository:
    @staticmethod
    def get_panels_by_ids(panel_ids: List[str]) -> List[Dict]:
        """
        패널 ID 리스트에 해당하는 데이터를 순서대로 조회합니다.
        (기존 main.py의 _get_ordered_welcome_data 및 utils.py 로직 통합)
        """
        if not panel_ids:
            return []

        try:
            with get_db_connection_context() as conn:
                if not conn:
                    raise Exception("DB Connection Failed")
                
                with conn.cursor() as cur:
                    # UNNEST를 사용하여 입력된 ID 순서를 보장하며 조회
                    query = """
                        WITH id_order (panel_id, ordering) AS (
                            SELECT * FROM unnest(%s::text[], %s::int[])
                        )
                        SELECT t.panel_id, t.structured_data
                        FROM welcome_meta2 t
                        JOIN id_order o ON t.panel_id = o.panel_id
                        ORDER BY o.ordering;
                    """
                    cur.execute(query, (panel_ids, list(range(len(panel_ids)))))
                    results = cur.fetchall()

                    data = []
                    for row in results:
                        pid, struct_data = row
                        if struct_data:
                            # panel_id를 데이터 안에 포함
                            item = {"panel_id": pid, **struct_data}
                            data.append(item)
                    return data
        except Exception as e:
            logging.error(f"PanelRepository.get_panels_by_ids 실패: {e}")
            return []

    @staticmethod
    def get_panel_detail(panel_id: str) -> Optional[Dict]:
        """단일 패널 상세 조회"""
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT panel_id, structured_data FROM welcome_meta2 WHERE panel_id = %s", 
                        (panel_id,)
                    )
                    result = cur.fetchone()
                    if result:
                        pid, data = result
                        return {"panel_id": pid, **(data or {})}
                    return None
        except Exception as e:
            logging.error(f"PanelRepository.get_panel_detail 실패: {e}")
            return None

    @staticmethod
    def get_field_distribution(field_name: str, limit: int = 50) -> Dict[str, float]:
        """
        특정 필드의 분포 통계를 DB에서 직접 계산 (insights.py 로직 이동)
        """
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    if field_name == "birth_year":
                        # 연령대 계산 쿼리
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
                        # 일반 필드
                        query = f"""
                            SELECT structured_data->>'{field_name}', COUNT(*), ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)
                            FROM welcome_meta2 WHERE structured_data->>'{field_name}' IS NOT NULL
                            GROUP BY 1 ORDER BY 3 DESC LIMIT {limit}
                        """
                    
                    cur.execute(query)
                    rows = cur.fetchall()
                    return {row[0]: float(row[2]) for row in rows if row[0]}
        except Exception as e:
            logging.error(f"PanelRepository.get_field_distribution 실패 ({field_name}): {e}")
            return {}