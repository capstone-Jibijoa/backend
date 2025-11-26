import asyncio
import time
import logging
import re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas.search import SearchQuery
from app.services.llm_prompt import parse_query_intelligent
from app.core.embeddings import initialize_embeddings
from app.repositories.panel_repo import PanelRepository
from app.repositories.qpoll_repo import QpollRepository
from app.utils.common import find_related_fields, get_negative_patterns
from app.database.connection import get_qdrant_client  

# 텍스트 유틸리티 추가 (연령대 변환, 답변 추출용)
from app.utils.text_utils import (
    truncate_text, 
    clean_label, 
    get_age_group, 
    extract_answer_from_template
)

# 상수 및 매핑 규칙
from app.constants.mapping import (
    FIELD_NAME_MAP, 
    QPOLL_FIELD_TO_TEXT, 
    VECTOR_CATEGORY_TO_FIELD,
    WELCOME_OBJECTIVE_FIELDS,
    VALUE_TRANSLATION_MAP,
)
from app.core.semantic_router import router
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, MatchText

class SearchService:
    def __init__(self):
        self.panel_repo = PanelRepository()
        self.qpoll_repo = QpollRepository()
        self.embeddings = initialize_embeddings()

    async def search_panels(self, query: SearchQuery) -> Dict[str, Any]:
        """[Lite 모드] 검색 및 테이블 데이터 구성"""
        start_time = time.time()
        query_text = query.query
        
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

        # 4. 데이터 병합 (데이터 가공 포함)
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
    
    async def get_table_data(self, panel_ids: List[str], display_fields: List[Dict], classification: Dict = None, limit: int = 100) -> List[Dict]:
        """[Pro 모드용] 테이블 데이터 조회"""
        if not panel_ids: return []
        target_ids = panel_ids[:limit]
        
        field_keys = [f['field'] for f in display_fields]
        qpoll_fields = [f for f in field_keys if f in QPOLL_FIELD_TO_TEXT]
        
        welcome_data, qpoll_data = await asyncio.gather(
            asyncio.to_thread(self.panel_repo.get_panels_by_ids, target_ids),
            asyncio.to_thread(self.qpoll_repo.get_responses_for_table, target_ids, qpoll_fields)
        )
        
        safe_classification = classification if classification else {'target_field': None}
        return self._merge_table_data(welcome_data, qpoll_data, display_fields, safe_classification)
    
    async def _perform_common_search(self, query_text: str, mode: str) -> Tuple[Dict, List[str], Dict]:
        """공통 검색 로직"""
        classification = parse_query_intelligent(query_text)
        user_limit = classification.get('limit', 100)

        search_results = await self._hybrid_search_logic(query_text, user_limit, classification)
        
        panel_id_list = search_results.get('final_panel_ids', [])
        classification['target_field'] = search_results.get('target_field')
        
        info = {
            "query": query_text,
            "classification": classification,
            "total_count": len(panel_id_list),
            "final_panel_ids": panel_id_list
        }
        return info, panel_id_list, classification

    async def _hybrid_search_logic(self, query: str, limit: int, classification: Dict) -> Dict:
        """Semantic Search V3 로직"""
        try:
            logging.info(f"🚀 Semantic Search V3 (Service): {query}")
            all_conditions = classification.get("semantic_conditions", [])
            positive_conditions = [c for c in all_conditions if not c.get('is_negative', False)]
            negative_conditions = [c for c in all_conditions if c.get('is_negative', False)]
            structured_filters = classification.get("demographic_filters", {})
            user_limit = limit

            intent = ""
            if positive_conditions:
                intent = positive_conditions[0].get("original_keyword", "")

            target_field, target_desc, intent = self._determine_target_field(intent, all_conditions)

            filtered_panel_ids = set()
            filters_for_sql = self._build_sql_filters(structured_filters, target_field, intent)

            if filters_for_sql:
                filtered_panel_ids = await asyncio.to_thread(
                    self.panel_repo.search_by_structure_filters, filters_for_sql
                )

            final_panel_ids = filtered_panel_ids
            vector_matched_ids = set()
            is_structured_target = target_field and target_field not in QPOLL_FIELD_TO_TEXT

            if is_structured_target and filtered_panel_ids:
                logging.info(f"🎯 정형 데이터 타겟({target_field}) 감지 -> 벡터 검색 없이 SQL 결과 사용")
                final_panel_ids = filtered_panel_ids

            elif intent and target_field:
                qdrant_client = get_qdrant_client()
                query_vector = await asyncio.to_thread(self.embeddings.embed_query, intent)
                collection_name, id_key_path, target_question_text, is_welcome = self._get_collection_config(target_field)
                negative_patterns = get_negative_patterns(target_field)

                if filtered_panel_ids:
                    logging.info(f"🚀 Reranking 모드: {len(filtered_panel_ids)}명 대상")
                    reranked_ids = await asyncio.to_thread(
                        self._rerank_candidates,
                        candidate_ids=list(filtered_panel_ids),
                        query_vector=query_vector,
                        qdrant_client=qdrant_client,
                        collection_name=collection_name,
                        id_key_path=id_key_path,
                        negative_patterns=negative_patterns,
                        target_question=target_question_text
                    )
                    vector_matched_ids = set(reranked_ids)
                else:
                    logging.info("🔍 일반 벡터 검색 모드")
                    vector_search_k = max(user_limit * 5, 500)
                    search_results = await asyncio.to_thread(
                        self._search_vectors_basic,
                        qdrant_client, collection_name, query_vector, target_question_text, vector_search_k
                    )
                    valid_hits = self._process_vector_hits(search_results, negative_patterns, is_welcome)
                    vector_matched_ids = set(valid_hits)
                    
                    if negative_conditions and vector_matched_ids:
                        neg_keywords = [q for nc in negative_conditions for q in nc.get('expanded_queries', [])]
                        if neg_keywords:
                            vector_matched_ids = await self._apply_negative_vector_filter(
                                vector_matched_ids, neg_keywords, qdrant_client, collection_name
                            )
                final_panel_ids = vector_matched_ids
            else:
                final_panel_ids = filtered_panel_ids

            return {
                "final_panel_ids": list(final_panel_ids),
                "target_field": target_field,
                "intent": intent
            }
        except Exception as e:
            logging.error(f"❌ _hybrid_search_logic 오류: {e}", exc_info=True)
            return {"final_panel_ids": [], "target_field": None}

    def _determine_target_field(self, intent: str, all_conditions: List) -> Tuple[Optional[str], Optional[str], str]:
        """라우팅 및 타겟 필드 보정 로직"""
        target_field_info = router.find_closest_field(intent)
        if not target_field_info:
            return None, None, intent

        target_field = target_field_info['field']
        target_desc = target_field_info['description']

        # 타겟이 정형 데이터인데 다른 조건이 설문형이면 교체 (보정)
        objective_fields = [f[0] for f in WELCOME_OBJECTIVE_FIELDS]
        if target_field in objective_fields:
            for cond in all_conditions:
                kw = cond.get('original_keyword', '')
                if kw == intent: continue
                
                alt_info = router.find_closest_field(kw)
                if alt_info and alt_info['field'] in QPOLL_FIELD_TO_TEXT:
                    logging.info(f"🔄 타겟 재설정: {target_field} -> {alt_info['field']}")
                    return alt_info['field'], alt_info['description'], kw

        return target_field, target_desc, intent

    def _build_sql_filters(self, structured_filters: Dict, target_field: str, intent: str) -> List[Dict]:
        """Structured Filters를 SQL용 리스트로 변환"""
        filters_for_sql = []
        
        # 기본 필터 변환
        for key, value in structured_filters.items():
            if key == "age_range":
                filters_for_sql.append({"field": "age", "operator": "between", "value": value})
            elif isinstance(value, dict) and any(k in value for k in ["min", "max", "gte", "lte"]):
                min_val = value.get("min") or value.get("gte")
                max_val = value.get("max") or value.get("lte")
                if min_val is not None and max_val is not None:
                    filters_for_sql.append({"field": key, "operator": "between", "value": [min_val, max_val]})
                elif min_val is not None:
                    filters_for_sql.append({"field": key, "operator": "gte", "value": min_val})
                elif max_val is not None:
                    filters_for_sql.append({"field": key, "operator": "lte", "value": max_val})
            else:
                filters_for_sql.append({"field": key, "operator": "in", "value": value})

        # Target Field를 SQL 필터로 변환 (정형 데이터일 경우)
        if target_field and target_field not in QPOLL_FIELD_TO_TEXT:
            is_specific = False
            if target_field in VALUE_TRANSLATION_MAP:
                for key in VALUE_TRANSLATION_MAP[target_field].keys():
                    if key == intent or (len(intent) < 10 and key in intent):
                        filters_for_sql.append({"field": target_field, "operator": "eq", "value": key})
                        is_specific = True
                        break
            
            if not is_specific:
                filters_for_sql.append({"field": target_field, "operator": "not_null", "value": "check"})

        return filters_for_sql

    def _get_collection_config(self, target_field: str) -> Tuple[str, str, Optional[str], bool]:
        """타겟 필드에 따른 Qdrant 컬렉션 설정 반환"""
        if target_field in QPOLL_FIELD_TO_TEXT:
            return "qpoll_vectors_v2", "panel_id", QPOLL_FIELD_TO_TEXT[target_field], False
        else:
            return "welcome_subjective_vectors", "metadata.panel_id", None, True

    def _rerank_candidates(self, candidate_ids: list, query_vector: list, qdrant_client, collection_name: str, id_key_path: str, negative_patterns: list, target_question: str = None) -> list:
        """[핵심 로직] In-Memory Reranking"""
        # (Safety) 대상 제한
        if len(candidate_ids) > 10000:
            candidate_ids = candidate_ids[:5000]

        use_python_filter = (len(candidate_ids) <= 2000) and (target_question is not None)
        
        # 기본 ID 필터
        must_conditions = [FieldCondition(key=id_key_path, match=MatchAny(any=[str(pid) for pid in candidate_ids]))]
        
        # DB 레벨 텍스트 필터 (대량일 때만)
        if target_question and not use_python_filter:
            must_conditions.append(FieldCondition(key="question", match=MatchText(text=target_question)))

        search_filter = Filter(must=must_conditions)
        
        # 데이터 조회 (Scroll)
        all_points = []
        offset = None
        while True:
            points, next_offset = qdrant_client.scroll(
                collection_name=collection_name, scroll_filter=search_filter, limit=2000, 
                with_vectors=True, with_payload=True, offset=offset
            )
            all_points.extend(points)
            offset = next_offset
            if offset is None: break

        if not all_points: return []

        # Python 정밀 매칭
        target_points = []
        if use_python_filter:
            norm_target = self._normalize_text(target_question)
            for p in all_points:
                p_question = p.payload.get("question", "")
                if norm_target in self._normalize_text(p_question):
                    target_points.append(p)
            if not target_points and all_points:
                target_points = all_points # Fallback
        else:
            target_points = all_points

        # 유사도 계산 및 정렬
        if not target_points: return []
        
        vectors = [p.vector for p in target_points]
        query_vec_np = np.array([query_vector])
        scores = cosine_similarity(query_vec_np, vectors)[0]

        results = []
        seen_pids = set()
        
        # 점수와 함께 부정어 필터링
        temp_results = []
        for i, point in enumerate(target_points):
            answer = point.payload.get('page_content') or point.payload.get('sentence') or ""
            # 부정어 체크
            if any(re.search(pat, answer) for pat in negative_patterns):
                continue
            
            pid = point.payload.get('panel_id') or point.payload.get('metadata', {}).get('panel_id')
            if pid:
                temp_results.append((pid, scores[i]))

        temp_results.sort(key=lambda x: x[1], reverse=True)
        
        final_ids = []
        for pid, _ in temp_results:
            if pid not in seen_pids:
                final_ids.append(pid)
                seen_pids.add(pid)
        
        return final_ids

    def _search_vectors_basic(self, client, collection, query_vector, target_question, limit):
        """일반 벡터 검색"""
        must = []
        if target_question:
            must.append(FieldCondition(key="question", match=MatchText(text=target_question)))
        
        try:
            return client.search(
                collection_name=collection, query_vector=query_vector,
                query_filter=Filter(must=must), limit=limit, with_payload=True
            )
        except: return []

    def _process_vector_hits(self, hits, negative_patterns, is_welcome):
        """검색 결과 후처리 (부정어 제거 및 ID 추출)"""
        valid_ids = []
        for hit in hits:
            if not hit.payload: continue
            answer = hit.payload.get('page_content') or hit.payload.get('sentence') or ""
            
            if any(re.search(pat, answer) for pat in negative_patterns):
                continue

            pid = hit.payload.get('metadata', {}).get('panel_id') if is_welcome else hit.payload.get('panel_id')
            if pid: valid_ids.append(pid)
        return valid_ids

    async def _apply_negative_vector_filter(self, panel_ids, neg_keywords, client, collection, threshold=0.55):
        """[심화] 벡터 유사도 기반 부정 필터링"""
        if not panel_ids: return set()
        
        # 1. 부정 키워드 임베딩
        neg_vectors = await asyncio.to_thread(self.embeddings.embed_documents, neg_keywords)
        
        # 2. 패널들의 답변 벡터 조회
        filtered_ids = set(panel_ids)
        return filtered_ids

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text: return ""
        return re.sub(r'[^a-zA-Z0-9가-힣]', '', text)

    def _prepare_display_fields(self, classification: Dict, query_text: str) -> List[Dict]:
        """
        화면에 표시할 컬럼들을 결정합니다.
        순서: 1. Target Field -> 2. 주요 인구통계(고정) -> 3. 기타 필드
        """
        relevant_fields = {"gender", "birth_year", "region_major"}
        target_field = classification.get('target_field')

        # 연관 필드 추가
        if target_field and target_field in QPOLL_FIELD_TO_TEXT:
            relevant_fields.update(["job_title_raw", "education_level", "income_household_monthly"])
        
        if target_field and target_field != 'unknown':
            relevant_fields.add(target_field)
            for _, fields in VECTOR_CATEGORY_TO_FIELD.items():
                if target_field in fields:
                    relevant_fields.update(fields)
                    break

        filters = classification.get('demographic_filters', {})
        relevant_fields.update(filters.keys())

        if query_text:
            dynamic = find_related_fields(query_text)
            relevant_fields.update(dynamic)

        # [핵심] 컬럼 순서 및 라벨 적용
        final_list = []

        # 1. Target Field (항상 맨 앞)
        if target_field and target_field != 'unknown':
            label = QPOLL_FIELD_TO_TEXT.get(target_field, FIELD_NAME_MAP.get(target_field, target_field))
            final_list.append({'field': target_field, 'label': label})
            relevant_fields.discard(target_field)

        # 2. 주요 인구통계 (고정 순서)
        priority_order = ["gender", "birth_year", "region_major", "job_title_raw", "education_level", "income_household_monthly"]
        for field in priority_order:
            if field in relevant_fields:
                final_list.append({'field': field, 'label': FIELD_NAME_MAP.get(field, field)})
                relevant_fields.discard(field)

        # 3. 나머지 필드 (정렬하여 추가)
        remaining_fields = sorted(list(relevant_fields))
        for field in remaining_fields:
            if field in FIELD_NAME_MAP:
                final_list.append({'field': field, 'label': FIELD_NAME_MAP[field]})
            # Q-Poll 필드인데 FIELD_NAME_MAP에 없는 경우 처리 (Fallback)
            elif field in QPOLL_FIELD_TO_TEXT:
                final_list.append({'field': field, 'label': QPOLL_FIELD_TO_TEXT[field]})
        
        return final_list[:12]

    async def get_table_data(self, panel_ids: List[str], display_fields: List[Dict], classification: Dict = None, limit: int = 100) -> List[Dict]:
        """
        패널 ID 리스트를 받아 테이블용 병합 데이터를 반환합니다.
        :param classification: 검색 결과 분류 정보 (target_field 포함)
        """
        if not panel_ids:
            return []
            
        target_ids = panel_ids[:limit]
        
        # 필요한 필드 키 추출
        field_keys = [f['field'] for f in display_fields]
        qpoll_fields = [f for f in field_keys if f in QPOLL_FIELD_TO_TEXT]
        
        # 병렬 데이터 조회
        welcome_data, qpoll_data = await asyncio.gather(
            asyncio.to_thread(self.panel_repo.get_panels_by_ids, target_ids),
            asyncio.to_thread(self.qpoll_repo.get_responses_for_table, target_ids, qpoll_fields)
        )
        
        # 필터링을 위한 classification 처리
        safe_classification = classification if classification else {'target_field': None}
        
        # 병합 및 필터링 수행
        return self._merge_table_data(welcome_data, qpoll_data, display_fields, safe_classification)

    # [수정] 필터링 로직 복원
    def _merge_table_data(self, welcome_data: List[Dict], qpoll_data: Dict, display_fields: List[Dict], classification: Dict) -> List[Dict]:
        """DB 데이터와 Qdrant 데이터 병합 + 타겟 필드 유효성 검사 + 데이터 정제"""
        merged = []
        target_field = classification.get('target_field')
        field_keys = [f['field'] for f in display_fields]

        for row in welcome_data:
            pid = row.get('panel_id')
            
            # 1. QPoll 데이터 병합
            if pid and pid in qpoll_data:
                row.update(qpoll_data[pid])
            
            # 2. 타겟 필드 유효성 검사 (Pro 모드 로직)
            is_valid_row = True
            if target_field and target_field != 'unknown':
                val = row.get(target_field)
                if not val or str(val).strip().lower() == 'nan':
                    is_valid_row = False
            
            if is_valid_row:
                for field in field_keys:
                    val = row.get(field)
                    
                    # [추가] 데이터 정제 (리스트 -> 문자열, 텍스트 자르기)
                    if isinstance(val, list):
                        val = ", ".join(map(str, val))
                    
                    # 값이 없으면 '-' 처리, 있으면 말줄임
                    if not val or str(val).strip().lower() == 'nan':
                        row[field] = "-"
                    else:
                        row[field] = truncate_text(str(val), 20) # 20자 제한

                merged.append(row)

        return merged