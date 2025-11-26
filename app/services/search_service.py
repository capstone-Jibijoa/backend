import asyncio
import time
import logging
from typing import Dict, List, Tuple, Any
from schemas.search import SearchQuery

from services.query_parser import parse_query_intelligent
from app.constants.mapping import (
    FIELD_NAME_MAP, 
    QPOLL_FIELD_TO_TEXT, 
    VECTOR_CATEGORY_TO_FIELD,
    find_related_fields
)
from schemas.search import SearchQuery
from services.query_parser import parse_query_intelligent
from app.core.embeddings import initialize_embeddings
from app.repositories.panel_repo import PanelRepository
from app.repositories.qpoll_repo import QpollRepository

class SearchService:
    def __init__(self):
        self.panel_repo = PanelRepository()
        self.qpoll_repo = QpollRepository()

    async def search_panels(self, query: SearchQuery) -> Dict[str, Any]:
        """
        [Lite 모드] 검색 및 테이블 데이터 구성
        """
        start_time = time.time()
        query_text = query.query  # 객체에서 값 추출
        
        # 1. 공통 검색 수행
        lite_info, panel_ids, classification = await self._perform_common_search(query_text, mode="lite")
        
        # 2. 화면 표시 필드 결정
        display_fields = self._prepare_display_fields(classification, query_text)
        
        # 3. 데이터 페칭
        field_keys = [f['field'] for f in display_fields]
        qpoll_fields = [f for f in field_keys if f in QPOLL_FIELD_TO_TEXT]
        
        welcome_data, qpoll_data = await asyncio.gather(
            asyncio.to_thread(self.panel_repo.get_panels_by_ids, panel_ids[:500]),
            asyncio.to_thread(self.qpoll_repo.get_responses_for_table, panel_ids[:500], qpoll_fields)
        )

        # 4. 데이터 병합
        table_data = self._merge_table_data(welcome_data, qpoll_data, display_fields, classification)
        
        search_time = time.time() - start_time
        logging.info(f"⏱️ 검색 서비스 완료: {search_time:.2f}초")

        return {
            "query": query_text,
            "classification": classification,
            "total_count": lite_info['total_count'],
            "tableData": table_data,
            "display_fields": display_fields,
            "mode": "lite"
        }
    
    async def _perform_common_search(self, query_text: str, mode: str) -> Tuple[Dict, List[str], Dict]:
        """공통 검색 로직 (LLM 파싱 -> 하이브리드 검색)"""
        # LLM 파싱
        classification = parse_query_intelligent(query_text)
        user_limit = classification.get('limit', 100)

        # I/O 바운드 작업이므로 스레드로 분리 권장
        search_results = await asyncio.to_thread(hybrid_search, query=query_text, limit=user_limit)
        
        panel_id_list = search_results.get('final_panel_ids', [])
        classification['target_field'] = search_results.get('target_field')
        
        info = {
            "query": query_text,
            "classification": classification,
            "total_count": len(panel_id_list),
            "final_panel_ids": panel_id_list
        }
        return info, panel_id_list, classification

    def _prepare_display_fields(self, classification: Dict, query_text: str) -> List[Dict]:
        """화면에 표시할 컬럼 결정 로직 (main.py에서 이동)"""
        relevant_fields = {"gender", "birth_year", "region_major"}
        target_field = classification.get('target_field')

        # Q-Poll인 경우 배경 정보 추가
        if target_field and target_field in QPOLL_FIELD_TO_TEXT:
            relevant_fields.update(["job_title_raw", "education_level", "income_household_monthly"])
        
        if target_field and target_field != 'unknown':
            relevant_fields.add(target_field)
            # 카테고리 관련 필드 추가
            for _, fields in VECTOR_CATEGORY_TO_FIELD.items():
                if target_field in fields:
                    relevant_fields.update(fields)
                    break

        # 필터 조건 필드 추가
        filters = classification.get('demographic_filters', {})
        relevant_fields.update(filters.keys())

        # 동적 필드 추가
        if query_text:
            dynamic = find_related_fields(query_text)
            relevant_fields.update(dynamic)

        # 정렬 및 포맷팅
        final_list = []
        if target_field and target_field != 'unknown':
            label = QPOLL_FIELD_TO_TEXT.get(target_field, FIELD_NAME_MAP.get(target_field, target_field))
            final_list.append({'field': target_field, 'label': label})
            relevant_fields.discard(target_field)

        for field in relevant_fields:
            if field in FIELD_NAME_MAP:
                final_list.append({'field': field, 'label': FIELD_NAME_MAP[field]})
        
        return final_list[:12]

    def _merge_table_data(self, welcome_data: List[Dict], qpoll_data: Dict, display_fields: List[Dict], classification: Dict) -> List[Dict]:
        """DB 데이터와 Qdrant 데이터 병합"""
        merged = []
        target_field = classification.get('target_field')
        field_keys = [f['field'] for f in display_fields]

        for row in welcome_data:
            pid = row.get('panel_id')
            
            # QPoll 데이터 병합
            if pid and pid in qpoll_data:
                row.update(qpoll_data[pid])
            
            # 없는 필드는 "-" 처리
            for field in field_keys:
                if field not in row or str(row[field]) == 'nan':
                    row[field] = "-"
            
            merged.append(row)
        return merged