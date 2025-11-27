# repository.py
import logging
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, MatchText

# 기존 db.py에서 연결 함수만 가져옵니다
from db import get_db_connection_context, get_qdrant_client
from mapping_rules import QPOLL_FIELD_TO_TEXT

# --- PostgreSQL Repository ---

class PanelRepository:
    """PostgreSQL(welcome_meta2) 데이터 접근 담당"""

    @staticmethod
    def fetch_panels_data(panel_ids: List[str]) -> List[Dict]:
        """여러 패널의 전체 데이터 조회 (JSONB)"""
        if not panel_ids: return []
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    query = "SELECT structured_data FROM welcome_meta2 WHERE panel_id = ANY(%s)"
                    cur.execute(query, (panel_ids,))
                    # panel_id가 누락되지 않도록 structured_data 안에 포함되어 있다고 가정하거나 병합
                    results = []
                    for row in cur.fetchall():
                        data = row[0]
                        if data: results.append(data)
                    return results
        except Exception as e:
            logging.error(f"DB 패널 데이터 조회 실패: {e}")
            return []

    @staticmethod
    def fetch_panel_detail(panel_id: str) -> Optional[Dict]:
        """단일 패널 상세 조회"""
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT panel_id, structured_data FROM welcome_meta2 WHERE panel_id = %s", (panel_id,))
                    row = cur.fetchone()
                    if not row: return None
                    pid, data = row
                    result = {"panel_id": pid}
                    if isinstance(data, dict): result.update(data)
                    return result
        except Exception as e:
            logging.error(f"패널 상세 조회 실패 ({panel_id}): {e}")
            raise e

    @staticmethod
    def fetch_ordered_table_data(panel_ids: List[str]) -> List[Tuple[str, Dict]]:
        """검색 결과 테이블용 데이터 조회 (순서 보장)"""
        if not panel_ids: return []
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
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
                    return cur.fetchall()
        except Exception as e:
            logging.error(f"Table Data 조회 실패: {e}")
            return []

    @staticmethod
    def search_panel_ids_by_sql(where_clause: str, params: List[Any]) -> Set[str]:
        """동적 SQL(WHERE 절)을 실행하여 panel_id 집합 반환"""
        if not where_clause: return set()
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    query = f"SELECT panel_id FROM welcome_meta2 {where_clause}"
                    cur.execute(query, tuple(params))
                    return {str(row[0]) for row in cur.fetchall()}
        except Exception as e:
            logging.error(f"SQL 필터 검색 실패: {e}")
            return set()

    @staticmethod
    def aggregate_field(query: str) -> Dict[str, float]:
        """통계용 집계 쿼리 실행"""
        try:
            with get_db_connection_context() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()
                    return {row[0]: float(row[2]) for row in rows if row[0]}
        except Exception as e:
            logging.error(f"DB 집계 실패: {e}")
            return {}


# --- Qdrant Repository ---

class VectorRepository:
    """Qdrant Vector DB 접근 담당"""

    @staticmethod
    def _scroll_all(collection_name: str, scroll_filter: Filter, with_payload: bool = True, with_vectors: bool = False, limit_per_req: int = 1000) -> List[Any]:
        """Qdrant Scroll 헬퍼 (전체 데이터 순회)"""
        client = get_qdrant_client()
        if not client: return []
        
        all_points = []
        next_offset = None
        try:
            while True:
                points, next_offset = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=scroll_filter,
                    limit=limit_per_req,
                    offset=next_offset,
                    with_payload=with_payload,
                    with_vectors=with_vectors
                )
                all_points.extend(points)
                if next_offset is None: break
            return all_points
        except Exception as e:
            logging.error(f"Qdrant Scroll 실패 ({collection_name}): {e}")
            return []

    @staticmethod
    def fetch_qpoll_responses(panel_ids: List[str], questions: List[str]) -> List[Any]:
        """특정 패널들의 특정 질문에 대한 응답 조회"""
        if not panel_ids or not questions: return []
        
        query_filter = Filter(must=[
            FieldCondition(key="panel_id", match=MatchAny(any=panel_ids)),
            FieldCondition(key="question", match=MatchAny(any=questions))
        ])
        
        # limit 계산: 패널 수 * 질문 수 (여유있게)
        limit = len(panel_ids) * len(questions)
        
        client = get_qdrant_client()
        if not client: return []
        
        try:
            # 여기서는 scroll 대신 한번에 가져오기 시도 (혹은 내부 루프)
            # 대량일 경우 _scroll_all 사용 권장
            return VectorRepository._scroll_all("qpoll_vectors_v2", query_filter)
        except Exception as e:
            logging.error(f"Q-Poll 응답 조회 실패: {e}")
            return []

    @staticmethod
    def fetch_qpoll_for_panel(panel_id: str) -> List[Any]:
        """단일 패널의 모든 Q-Poll 응답 조회"""
        query_filter = Filter(must=[FieldCondition(key="panel_id", match=MatchValue(value=panel_id))])
        return VectorRepository._scroll_all("qpoll_vectors_v2", query_filter, limit_per_req=100)

    @staticmethod
    def fetch_qpoll_by_question(question_text: str) -> List[Any]:
        """특정 질문에 대한 모든 패널의 응답 조회 (통계용)"""
        query_filter = Filter(must=[FieldCondition(key="question", match=MatchValue(value=question_text))])
        return VectorRepository._scroll_all("qpoll_vectors_v2", query_filter)

    @staticmethod
    def hybrid_search(collection_name: str, query_vector: List[float], query_filter: Optional[Filter] = None, limit: int = 100) -> List[Any]:
        """벡터 검색 수행"""
        client = get_qdrant_client()
        if not client: return []
        try:
            return client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )
        except Exception as e:
            logging.error(f"Qdrant Search 실패: {e}")
            return []