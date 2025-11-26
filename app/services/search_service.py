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
from app.database.connection import get_qdrant_client  

# 텍스트 유틸리티 추가 (연령대 변환, 답변 추출용)
from app.utils.common import (
    truncate_text, 
    clean_label, 
    get_age_group, 
    extract_answer_from_template,
    find_related_fields, 
    get_negative_patterns
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
        
        # 2. 화면 표시 필드 결정 (정렬 로직 개선됨)
        display_fields = self._prepare_display_fields(classification, query_text)
        
        # 3. 데이터 페칭
        field_keys = [f['field'] for f in display_fields]
        qpoll_fields = [f for f in field_keys if f in QPOLL_FIELD_TO_TEXT]
        
        welcome_data, qpoll_data = await asyncio.gather(
            asyncio.to_thread(self.panel_repo.get_panels_by_ids, panel_ids[:500]),
            asyncio.to_thread(self.qpoll_repo.get_responses_for_table, panel_ids[:500], qpoll_fields)
        )

        # 4. 데이터 병합 (포맷팅 적용됨)
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
        """
        Semantic Search V3 로직 (디버깅 로그 강화)
        """
        try:
            logging.info("="*50)
            logging.info(f"🚀 [Search V3] 검색 시작: '{query}'")
            
            all_conditions = classification.get("semantic_conditions", [])
            structured_filters = classification.get("demographic_filters", {})
            user_limit = limit

            intent = ""
            positive_conditions = [c for c in all_conditions if not c.get('is_negative', False)]
            negative_conditions = [c for c in all_conditions if c.get('is_negative', False)]
            
            if positive_conditions:
                intent = positive_conditions[0].get("original_keyword", "")
            
            logging.info(f"   🕵️ [Intent] 파악된 의도: '{intent}'")
            logging.info(f"   🧬 [Filter] 구조적 필터: {structured_filters}")

            # 2. 타겟 필드 결정
            target_field, target_desc, intent = self._determine_target_field(intent, all_conditions)
            logging.info(f"   🎯 [Target] 결정된 타겟 필드: {target_field} ({target_desc})")

            # 3. 1차 SQL 필터링
            filtered_panel_ids = set()
            filters_for_sql = self._build_sql_filters(structured_filters, target_field, intent)

            if filters_for_sql:
                logging.info(f"   📋 [SQL] 필터 쿼리 생성: {filters_for_sql}")
                filtered_panel_ids = await asyncio.to_thread(
                    self.panel_repo.search_by_structure_filters, filters_for_sql
                )
                logging.info(f"   ✅ [SQL] 1차 필터링 결과: {len(filtered_panel_ids)}명")
            else:
                logging.info("   📋 [SQL] 적용할 필터 없음 (전체 대상)")

            # 4. 검색 로직 분기
            final_panel_ids = filtered_panel_ids
            vector_matched_ids = set()
            is_structured_target = target_field and target_field not in QPOLL_FIELD_TO_TEXT

            # [Case A] 정형 데이터 타겟 + SQL 필터 존재
            if is_structured_target and filtered_panel_ids:
                logging.info(f"   🏃‍♂️ [Skip] 정형 데이터 타겟({target_field}) -> 벡터 검색 생략")
                final_panel_ids = filtered_panel_ids

            # [Case B] 벡터 검색 필요
            elif intent and target_field:
                qdrant_client = get_qdrant_client()
                query_vector = await asyncio.to_thread(self.embeddings.embed_query, intent)
                collection_name, id_key_path, target_question_text, is_welcome = self._get_collection_config(target_field)
                negative_patterns = get_negative_patterns(target_field)
                
                logging.info(f"   🔍 [Vector] 검색 준비: 컬렉션={collection_name}, 질문필터='{target_question_text}'")

                # [분기 1] SQL 결과 있음 -> Reranking
                if filtered_panel_ids:
                    logging.info(f"   🚀 [Rerank] 대상 인원: {len(filtered_panel_ids)}명 (전수 조사)")
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
                    logging.info(f"   ✅ [Rerank] 완료: {len(filtered_panel_ids)}명 -> {len(vector_matched_ids)}명 (유사도/부정어 필터링)")

                # [분기 2] SQL 결과 없음 -> 일반 벡터 검색
                else:
                    vector_search_k = max(user_limit * 5, 500)
                    logging.info(f"   🔍 [Vector] 일반 검색 시작 (Limit: {vector_search_k})")
                    
                    search_results = await asyncio.to_thread(
                        self._search_vectors_basic,
                        qdrant_client, collection_name, query_vector, target_question_text, vector_search_k
                    )
                    logging.info(f"   📥 [Vector] Raw 결과: {len(search_results)}건")

                    valid_hits = self._process_vector_hits(search_results, negative_patterns, is_welcome)
                    vector_matched_ids = set(valid_hits)
                    logging.info(f"   ✅ [Vector] 텍스트 후처리 완료: {len(vector_matched_ids)}명 (부정어 제외)")
                    
                    # 부정 조건 필터링 (심화)
                    if negative_conditions and vector_matched_ids:
                        neg_keywords = [q for nc in negative_conditions for q in nc.get('expanded_queries', [])]
                        if neg_keywords:
                            logging.info(f"   🚫 [Negative] 부정 조건 필터링 적용: {neg_keywords}")
                            before_cnt = len(vector_matched_ids)
                            # (간단한 필터링 로직 호출)
                            vector_matched_ids = await self._apply_negative_vector_filter(
                                vector_matched_ids, neg_keywords, qdrant_client, collection_name
                            )
                            logging.info(f"   ✂️ [Negative] 필터링 결과: {before_cnt}명 -> {len(vector_matched_ids)}명")

                final_panel_ids = vector_matched_ids

            else:
                # 의도가 없거나 타겟 필드를 못 찾은 경우 SQL 결과만 사용
                logging.info("   ⚠️ [Warning] 의도/타겟 불분명 -> SQL 필터 결과만 사용")
                final_panel_ids = filtered_panel_ids

            logging.info(f"🎉 [Result] 최종 검색 결과: {len(final_panel_ids)}명")
            logging.info("="*50)

            return {
                "final_panel_ids": list(final_panel_ids),
                "target_field": target_field,
                "intent": intent
            }
        except Exception as e:
            logging.error(f"❌ _hybrid_search_logic 오류: {e}", exc_info=True)
            return {"final_panel_ids": [], "target_field": None}

    def _determine_target_field(self, intent: str, all_conditions: List) -> Tuple[Optional[str], Optional[str], str]:
        target_field_info = router.find_closest_field(intent)
        if not target_field_info: return None, None, intent
        target_field = target_field_info['field']
        target_desc = target_field_info['description']
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
        filters_for_sql = []
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
        if target_field in QPOLL_FIELD_TO_TEXT:
            return "qpoll_vectors_v2", "panel_id", QPOLL_FIELD_TO_TEXT[target_field], False
        else:
            return "welcome_subjective_vectors", "metadata.panel_id", None, True

    def _rerank_candidates(self, candidate_ids: list, query_vector: list, qdrant_client, collection_name: str, id_key_path: str, negative_patterns: list, target_question: str = None) -> list:
        if len(candidate_ids) > 10000: candidate_ids = candidate_ids[:5000]
        use_python_filter = (len(candidate_ids) <= 2000) and (target_question is not None)
        must_conditions = [FieldCondition(key=id_key_path, match=MatchAny(any=[str(pid) for pid in candidate_ids]))]
        if target_question and not use_python_filter:
            must_conditions.append(FieldCondition(key="question", match=MatchText(text=target_question)))
        search_filter = Filter(must=must_conditions)
        all_points = []
        offset = None
        while True:
            points, next_offset = qdrant_client.scroll(collection_name=collection_name, scroll_filter=search_filter, limit=2000, with_vectors=True, with_payload=True, offset=offset)
            all_points.extend(points)
            offset = next_offset
            if offset is None: break
        if not all_points: return []
        target_points = []
        if use_python_filter:
            norm_target = self._normalize_text(target_question)
            for p in all_points:
                p_question = p.payload.get("question", "")
                if norm_target in self._normalize_text(p_question): target_points.append(p)
            if not target_points and all_points: target_points = all_points
        else: target_points = all_points
        if not target_points: return []
        vectors = [p.vector for p in target_points]
        query_vec_np = np.array([query_vector])
        scores = cosine_similarity(query_vec_np, vectors)[0]
        temp_results = []
        for i, point in enumerate(target_points):
            answer = point.payload.get('page_content') or point.payload.get('sentence') or ""
            if any(re.search(pat, answer) for pat in negative_patterns): continue
            pid = point.payload.get('panel_id') or point.payload.get('metadata', {}).get('panel_id')
            if pid: temp_results.append((pid, scores[i]))
        temp_results.sort(key=lambda x: x[1], reverse=True)
        final_ids = []
        seen_pids = set()
        for pid, _ in temp_results:
            if pid not in seen_pids:
                final_ids.append(pid)
                seen_pids.add(pid)
        return final_ids

    def _search_vectors_basic(self, client, collection, query_vector, target_question, limit):
        must = []
        if target_question: must.append(FieldCondition(key="question", match=MatchText(text=target_question)))
        try: return client.search(collection_name=collection, query_vector=query_vector, query_filter=Filter(must=must), limit=limit, with_payload=True)
        except: return []

    def _process_vector_hits(self, hits, negative_patterns, is_welcome):
        valid_ids = []
        for hit in hits:
            if not hit.payload: continue
            answer = hit.payload.get('page_content') or hit.payload.get('sentence') or ""
            if any(re.search(pat, answer) for pat in negative_patterns): continue
            pid = hit.payload.get('metadata', {}).get('panel_id') if is_welcome else hit.payload.get('panel_id')
            if pid: valid_ids.append(pid)
        return valid_ids

    async def _apply_negative_vector_filter(self, panel_ids: Set[str], neg_keywords: List[str], client, collection_name: str, threshold: float = 0.55) -> Set[str]:
        """
        부정 키워드와 유사한 벡터를 가진 패널을 검색 결과에서 제외 (캡스톤 로직 복원)
        """
        if not panel_ids or not neg_keywords:
            return panel_ids

        logging.info(f"🚫 [Negative Filter] 제외 키워드: {neg_keywords} (Threshold: {threshold})")
        
        # 부정 키워드 벡터화
        neg_vectors = await asyncio.to_thread(self.embeddings.embed_documents, neg_keywords)
        
        ids_to_exclude = set()
        
        # 각 부정 벡터에 대해 유사한 패널 검색
        for vector in neg_vectors:
            try:
                # Qdrant 검색 (점수가 threshold 이상이면 제외 대상)
                search_results = await asyncio.to_thread(
                    client.search,
                    collection_name=collection_name,
                    query_vector=vector,
                    limit=2000, # 충분히 많은 수 검색
                    with_payload=True,
                    score_threshold=threshold
                )
                
                for hit in search_results:
                    # 메타데이터 혹은 페이로드에서 panel_id 추출
                    pid = hit.payload.get('panel_id')
                    if not pid and 'metadata' in hit.payload:
                        pid = hit.payload['metadata'].get('panel_id')
                    
                    if pid:
                        ids_to_exclude.add(str(pid))
                        
            except Exception as e:
                logging.error(f"부정 필터 검색 중 오류: {e}")

        if ids_to_exclude:
            logging.info(f"   ✂️ [Negative] {len(ids_to_exclude)}명 제외됨")
        
        # 차집합 반환
        return panel_ids - ids_to_exclude

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text: return ""
        return re.sub(r'[^a-zA-Z0-9가-힣]', '', text)

    def _prepare_display_fields(self, classification: Dict, query_text: str) -> List[Dict]:
        """
        화면에 표시할 컬럼을 결정하고 '고정된 우선순위'로 정렬합니다.
        """
        relevant_fields = {"gender", "birth_year", "region_major"}
        target_field = classification.get('target_field')

        # 1. 연관 필드 수집
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

        final_list = []

        # [1] Target Field 
        if target_field and target_field != 'unknown':
            label = QPOLL_FIELD_TO_TEXT.get(target_field, FIELD_NAME_MAP.get(target_field, target_field))
            final_list.append({'field': target_field, 'label': label})
            relevant_fields.discard(target_field)

        # [2] 주요 인구통계 (고정 순서: 성별 -> 나이 -> 지역 -> 직업 -> 학력 -> 소득)
        priority_order = [
            "gender", "birth_year", "region_major", 
            "job_title_raw", "education_level", "income_household_monthly",
            "marital_status", "children_count"
        ]
        
        for field in priority_order:
            if field in relevant_fields:
                final_list.append({'field': field, 'label': FIELD_NAME_MAP.get(field, field)})
                relevant_fields.discard(field) 

        # [3] 나머지 필드 (알파벳순 또는 임의 순서)
        remaining_fields = sorted(list(relevant_fields))
        for field in remaining_fields:
            if field in FIELD_NAME_MAP:
                final_list.append({'field': field, 'label': FIELD_NAME_MAP[field]})
            elif field in QPOLL_FIELD_TO_TEXT:
                final_list.append({'field': field, 'label': QPOLL_FIELD_TO_TEXT[field]})
        
        return final_list[:12]

    def _merge_table_data(self, welcome_data: List[Dict], qpoll_data: Dict, display_fields: List[Dict], classification: Dict) -> List[Dict]:
        """
        DB 데이터 + Qdrant 데이터 병합 + 필드 값 가공 + *필요한 컬럼만 필터링*
        """
        merged = []
        target_field = classification.get('target_field')
        field_keys = [f['field'] for f in display_fields]

        for row in welcome_data:
            pid = row.get('panel_id')
            
            # 1. QPoll 데이터 병합 
            if pid and pid in qpoll_data:
                row.update(qpoll_data[pid])
            
            # 2. 타겟 필드 유효성 검사 (Pro 모드 필터링)
            is_valid_row = True
            if target_field and target_field != 'unknown':
                val = row.get(target_field)
                if not val or str(val).strip().lower() == 'nan':
                    is_valid_row = False
            
            # 3. 데이터 가공 및 '선별된 컬럼만' 담기
            if is_valid_row:
                # 원본 row를 그대로 쓰지 않고, 보여줄 데이터만 담을 새 딕셔너리 생성
                filtered_row = {'panel_id': pid} 
                
                for field in field_keys:
                    val = row.get(field)
                    
                    # (A) 생년월일 -> 연령대 변환
                    if field == 'birth_year':
                        val = get_age_group(val)
                    
                    # (B) Q-Poll 서술형 응답 -> 핵심 답변 추출
                    elif field in QPOLL_FIELD_TO_TEXT and val:
                        val = extract_answer_from_template(field, str(val))
                        
                    # (C) 리스트 -> 문자열 변환
                    elif isinstance(val, list):
                        val = ", ".join(map(str, val))
                    
                    # (D) 결측치 처리 및 말줄임
                    if not val or str(val).strip().lower() == 'nan':
                        filtered_row[field] = "-"
                    else:
                        filtered_row[field] = truncate_text(str(val), 20)

                merged.append(filtered_row)

        return merged