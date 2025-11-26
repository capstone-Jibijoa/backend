import logging
import os
from collections import Counter
from typing import List, Dict, Optional
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, MatchValue

from app.database.connection import get_qdrant_client
from app.utils.text_utils import extract_answer_from_template
from app.constants.mapping import QPOLL_FIELD_TO_TEXT 

class QpollRepository:
    def __init__(self):
        self.collection_name = os.getenv("QDRANT_COLLECTION_QPOLL_NAME", "qpoll_vectors_v2")

    def get_responses_for_table(self, panel_ids: List[str], qpoll_fields: List[str]) -> Dict[str, Dict[str, str]]:
        """
        테이블 표시용: 특정 패널들의 특정 질문 답변만 조회
        """
        client = get_qdrant_client()
        if not client or not panel_ids or not qpoll_fields:
            return {}

        # 필드명을 질문 텍스트로 변환
        questions_to_fetch = [QPOLL_FIELD_TO_TEXT[f] for f in qpoll_fields if f in QPOLL_FIELD_TO_TEXT]
        if not questions_to_fetch:
            return {}

        try:
            query_filter = Filter(must=[
                FieldCondition(key="panel_id", match=MatchAny(any=panel_ids)),
                FieldCondition(key="question", match=MatchAny(any=questions_to_fetch))
            ])

            # Scroll로 데이터 조회
            limit = len(panel_ids) * len(questions_to_fetch)
            results, _ = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            # 결과 매핑
            result_map = {pid: {} for pid in panel_ids}
            text_to_field_map = {v: k for k, v in QPOLL_FIELD_TO_TEXT.items()}

            for point in results:
                pid = point.payload.get("panel_id")
                question = point.payload.get("question")
                sentence = point.payload.get("sentence")
                
                if pid and question and sentence:
                    field_key = text_to_field_map.get(question)
                    if field_key:
                        # 템플릿 파싱하여 핵심 답변만 추출
                        core_value = extract_answer_from_template(field_key, sentence)
                        result_map[pid][field_key] = core_value
            
            return result_map
        except Exception as e:
            logging.error(f"QpollRepository.get_responses_for_table 실패: {e}")
            return {}

    def get_all_responses_by_panel(self, panel_id: str) -> Dict[str, str]:
        """특정 패널의 모든 설문 응답 조회 (상세 페이지용)"""
        client = get_qdrant_client()
        data = {"qpoll_응답_개수": 0}
        
        if not client: return data

        try:
            res, _ = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=[FieldCondition(key="panel_id", match=MatchValue(value=panel_id))]),
                limit=100,
                with_payload=True, 
                with_vectors=False
            )
            
            if res:
                data["qpoll_응답_개수"] = len(res)
                txt_map = {v: k for k, v in QPOLL_FIELD_TO_TEXT.items()}
                for p in res:
                    if p.payload:
                        q = p.payload.get("question")
                        s = p.payload.get("sentence")
                        if q and s:
                            k = txt_map.get(q)
                            if k: data[k] = s
            return data
        except Exception as e:
            logging.error(f"QpollRepository.get_all_responses_by_panel 실패: {e}")
            return data

    def get_distribution(self, qpoll_field: str, limit: int = 50) -> Dict[str, float]:
        """Qdrant에서 특정 질문의 답변 분포 집계"""
        client = get_qdrant_client()
        question_text = QPOLL_FIELD_TO_TEXT.get(qpoll_field)
        
        if not client or not question_text:
            return {}

        try:
            query_filter = Filter(must=[FieldCondition(key="question", match=MatchValue(value=question_text))])
            
            all_points = []
            next_offset = None
            
            # 전체 데이터 스캔 (데이터가 아주 많아지면 비효율적일 수 있으나 현재 구조 유지)
            while True:
                points, next_offset = client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=1000,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False
                )
                all_points.extend(points)
                if next_offset is None:
                    break
            
            extracted_values = []
            for p in all_points:
                if p.payload and p.payload.get("sentence"):
                    raw_sentence = p.payload.get("sentence")
                    core_val = extract_answer_from_template(qpoll_field, raw_sentence)
                    if core_val:
                        extracted_values.append(core_val)
            
            if not extracted_values:
                return {}
                
            val_counts = Counter(extracted_values)
            total = len(extracted_values)
            return {k: round((v / total) * 100, 1) for k, v in val_counts.most_common(limit)}

        except Exception as e:
            logging.error(f"QpollRepository.get_distribution 실패: {e}")
            return {}