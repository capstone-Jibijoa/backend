"""
하이브리드 검색 v2

검색 흐름 (3-Stage Strict Verification):
1. Stage 1: PostgreSQL로 objective (demographic) 필터링
2. Stage 2: 필터링된 범위 내에서 must-have 조건 엄격 검증 (AND 연산)
3. Stage 3: Preference 조건으로 재순위화
4. Stage 4: Negative 조건 제거
"""

import logging
from typing import List, Set, Dict, Optional
from qdrant_client import QdrantClient

from llm import classify_query_keywords, classify_keyword_to_qpoll_topic
from search_helpers import (
    search_welcome_objective,
    initialize_embeddings,
    search_must_have_conditions,
    search_preference_conditions,
    filter_negative_conditions,
    embed_keywords
)
import re
from db import get_qdrant_client
from mapping_rules import get_field_mapping, QPOLL_FIELD_TO_TEXT, QPOLL_ANSWER_TEMPLATES


def hybrid_search(
    query: str,
    use_welcome: bool = True,
    use_qpoll: bool = True,
    limit: Optional[int] = None
) -> Dict:
    """3-Stage 하이브리드 검색 메인 함수"""
    logging.info("="*80)
    logging.info(f"🚀 하이브리드 검색 v2 시작")
    logging.info(f"📝 쿼리: {query}")
    logging.info("="*80)
    
    logging.info("\n[Step 1] LLM 쿼리 분류")
    try:
        classification = classify_query_keywords(query)
        logging.info(f"✅ 분류 완료:")
        logging.info(f"  - Structured Filters: {classification.get('structured_filters')}")
        logging.info(f"  - Must-have: {classification['must_have_keywords']}")
        logging.info(f"  - Preference: {classification['preference_keywords']}")
        logging.info(f"  - Negative: {classification['negative_keywords']}")
        
        final_limit = limit or classification.get('limit', 100)
        logging.info(f"  - 목표 인원: {final_limit}명")

        # --- 사용자 요청에 따른 QPoll 키워드 분류 확인 및 답변 템플릿 기반 변환 로직 ---
        transformed_must_have_keywords = []
        transformed_preference_keywords = []
        transformed_negative_keywords = []

        all_keywords_categories = {
            "must_have": classification['must_have_keywords'],
            "preference": classification['preference_keywords'],
            "negative": classification['negative_keywords']
        }

        for category, keywords_list in all_keywords_categories.items():
            transformed_list = []
            for keyword in keywords_list:
                mapping_info = get_field_mapping(keyword)
                
                final_keyword_for_embedding = keyword # 기본값은 원본 키워드
                qpoll_field = None

                if mapping_info and mapping_info.get("type") == "qpoll":
                    qpoll_field = mapping_info['field']
                    logging.info(f"💡 키워드 '{keyword}'는 QPoll 주제로 사전 분류됨: '{qpoll_field}'")
                else:
                    logging.info(f"⚠️ 키워드 '{keyword}'는 QPoll 주제로 사전 분류되지 않음. LLM 분류 시도...")
                    qpoll_field = classify_keyword_to_qpoll_topic(keyword)

                if qpoll_field and qpoll_field in QPOLL_ANSWER_TEMPLATES:
                    template = QPOLL_ANSWER_TEMPLATES[qpoll_field]
                    # (으)로, (이)다 같은 문법적 부분을 제거
                    cleaned_template = re.sub(r'\([가-힣]\)', '', template)
                    # {answer_str}을 키워드로 교체
                    final_sentence = cleaned_template.replace("{answer_str}", keyword)
                    final_keyword_for_embedding = final_sentence
                    logging.info(f"✅ 키워드 '{keyword}' -> QPoll 답변 문장으로 변환: '{final_sentence}'")
                else:
                    if qpoll_field:
                        logging.warning(f"❌ QPoll 주제 '{qpoll_field}'에 대한 답변 템플릿을 찾을 수 없습니다. 원본 키워드로 임베딩.")
                    else:
                        logging.warning(f"❌ LLM도 키워드 '{keyword}'에 대한 QPoll 주제를 찾지 못했습니다. 원본 키워드로 임베딩.")

                transformed_list.append(final_keyword_for_embedding)

            if category == "must_have":
                transformed_must_have_keywords = transformed_list
            elif category == "preference":
                transformed_preference_keywords = transformed_list
            elif category == "negative":
                transformed_negative_keywords = transformed_list

        # 원래 classification 객체를 업데이트
        classification['must_have_keywords'] = transformed_must_have_keywords
        classification['preference_keywords'] = transformed_preference_keywords
        classification['negative_keywords'] = transformed_negative_keywords
        # --- QPoll 키워드 분류 확인 및 변환 로직 끝 ---

    except Exception as e:
        logging.error(f"❌ LLM 분류 실패: {e}")
        return {
            "panel_ids": [],
            "total_count": 0,
            "error": str(e),
            "classification": {}
        }
    
    embeddings = None
    if classification['must_have_keywords'] or classification['preference_keywords'] or classification['negative_keywords']:
        logging.info("\n[Step 1.5] 벡터 검색 필요 - 임베딩 모델 초기화")
        embeddings = initialize_embeddings()
        if not embeddings:
            logging.error("❌ 임베딩 모델 초기화 실패")
            return {
                "panel_ids": [],
                "total_count": 0,
                "error": "임베딩 모델 초기화 실패",
                "classification": classification
            }
    else:
        logging.info("\n[Step 1.5] 벡터 검색 불필요 - 임베딩 모델 초기화 스킵")

    qdrant_client = None
    if embeddings:
        logging.info("\n[Step 1.6] Qdrant 클라이언트 초기화")
        qdrant_client = get_qdrant_client()
        if not qdrant_client:
            logging.error("❌ Qdrant 클라이언트 연결 실패")
            return {
                "panel_ids": [],
                "total_count": 0,
                "error": "Qdrant 연결 실패",
                "classification": classification
            }

    logging.info("\n[Stage 1] PostgreSQL Structured 필터링")
    
    structured_filters = classification.get('structured_filters', [])
    stage1_ids = set()
    
    if structured_filters:
        if use_welcome:
            welcome_ids, _ = search_welcome_objective(
                filters=structured_filters,
                attempt_name="구조화(Stage1)"
            )
            stage1_ids = welcome_ids
            logging.info(f"   Welcome 구조화 필터: {len(welcome_ids):,}명")
        
    else:
        logging.info("   구조화 필터 없음 - Stage 1 스킵")
        stage1_ids = None
    
    if stage1_ids is not None:
        logging.info(f"✅ Stage 1 완료: {len(stage1_ids):,}명 (Structured 필터링)")
    else:
        logging.info(f"⚠️  Stage 1: 구조화 필터 없음 - 전체 검색 모드")

    logging.info("\n[Stage 2] Must-have 조건 엄격 검증")
    
    must_have_keywords = classification['must_have_keywords']
    stage2_ids = stage1_ids
    
    if must_have_keywords:
        must_have_vectors = embed_keywords(must_have_keywords, embeddings)
        
        per_keyword_results = []
        for i, (keyword, vector) in enumerate(zip(must_have_keywords, must_have_vectors)):
            logging.info(f"   🔍 Must-have [{i+1}/{len(must_have_keywords)}]: '{keyword}' 검색")
            
            single_keyword_list = [keyword]
            single_vector_list = [vector]

            welcome_matches = set()
            if use_welcome:
                welcome_matches = search_must_have_conditions(
                    must_have_keywords=single_keyword_list,
                    query_vectors=single_vector_list,
                    qdrant_client=qdrant_client,
                    collection_name="welcome_subjective_vectors",
                    pre_filtered_panel_ids=stage1_ids,
                    threshold=0.53,
                    hnsw_ef=128
                )

            qpoll_matches = set()
            if use_qpoll:
                qpoll_matches = search_must_have_conditions(
                    must_have_keywords=single_keyword_list,
                    query_vectors=single_vector_list,
                    qdrant_client=qdrant_client,
                    collection_name="qpoll_vectors_v2",
                    pre_filtered_panel_ids=stage1_ids,
                    threshold=0.50,
                    hnsw_ef=128
                )
            
            keyword_total_matches = welcome_matches | qpoll_matches
            per_keyword_results.append(keyword_total_matches)
            logging.info(f"      → '{keyword}' 조건 일치: {len(keyword_total_matches):,}명 (Welcome 또는 QPoll)")

        if per_keyword_results:
            final_must_have_ids = per_keyword_results[0]
            for other_result in per_keyword_results[1:]:
                final_must_have_ids &= other_result
            
            stage2_ids = final_must_have_ids
        else:
            stage2_ids = set()

        if stage1_ids is not None:
             stage2_ids &= stage1_ids
        
        logging.info(f"✅ Stage 2 완료: {len(stage2_ids):,}명 (모든 Must-have 조건 교집합)")
        
        min_threshold = max(10, int(final_limit * 0.2))
        if must_have_keywords and len(stage2_ids) < min_threshold:
            logging.warning(f"   ⚠️  Stage 2 결과 부족 ({len(stage2_ids)}명 < {min_threshold}명)")
            logging.warning("   🔄 Fallback: Must-have 키워드를 Preference로 강등하여 재시도")
            
            preference_keywords_original = classification['preference_keywords']
            classification['preference_keywords'] = preference_keywords_original + must_have_keywords
            classification['must_have_keywords'] = []
            
            stage2_ids = stage1_ids if stage1_ids is not None else set()
            
            logging.info(f"   ✅ Fallback 완료: Preference 키워드 {len(classification['preference_keywords'])}개로 재검색")
    else:
        logging.info("   Must-have 키워드 없음 - Stage 2 스킵")
        stage2_ids = stage1_ids if stage1_ids is not None else set()
    
    logging.info("\n[Stage 3] Preference 조건 재순위화")
    
    preference_keywords = classification['preference_keywords']
    stage3_scored = []
    
    if preference_keywords and stage2_ids:
        preference_vectors = embed_keywords(preference_keywords, embeddings)
        all_found_categories = []
        
        welcome_scored = []
        if use_welcome:
            welcome_scored, welcome_categories = search_preference_conditions(
                preference_keywords=preference_keywords,
                query_vectors=preference_vectors,
                qdrant_client=qdrant_client,
                collection_name="welcome_subjective_vectors",
                candidate_panel_ids=stage2_ids,
                threshold=0.38,
                top_k_per_keyword=500
            )
            all_found_categories.extend(welcome_categories) 
            logging.info(f"   Welcome Preference: {len(welcome_scored)}명 스코어링")
        
        qpoll_scored = []
        if use_qpoll:
            qpoll_scored, qpoll_categories = search_preference_conditions(
                preference_keywords=preference_keywords,
                query_vectors=preference_vectors,
                qdrant_client=qdrant_client,
                collection_name="qpoll_vectors_v2",
                candidate_panel_ids=stage2_ids,
                threshold=0.38,
                top_k_per_keyword=500
            )
            all_found_categories.extend(qpoll_categories)
            logging.info(f"   QPoll Preference: {len(qpoll_scored)}명 스코어링")
        
        max_scores = {}
        for pid, score in welcome_scored + qpoll_scored:
            max_scores[pid] = max(max_scores.get(pid, 0.0), score)
        
        stage3_scored = sorted(max_scores.items(), key=lambda x: x[1], reverse=True)
        if all_found_categories:
            from collections import Counter
            category_counts = Counter(all_found_categories)
            top_categories = [cat for cat, count in category_counts.most_common(2)]
            classification['found_categories'] = top_categories

        if 'found_categories' not in classification:
            classification['found_categories'] = []

        logging.info(f"✅ Stage 3 완료: {len(stage3_scored):,}명 (Preference 스코어링)")
    else:
        stage3_scored = [(pid, 0.0) for pid in stage2_ids] if stage2_ids else []
        logging.info("   Preference 키워드 없음 - Stage 3 스킵")

    logging.info("\n[Stage 4] Negative 조건 제거")
    
    negative_keywords = classification['negative_keywords']
    stage4_ids = {pid for pid, _ in stage3_scored}
    
    if negative_keywords and stage4_ids:
        negative_vectors = embed_keywords(negative_keywords, embeddings)
        
        if use_welcome:
            stage4_ids = filter_negative_conditions(
                panel_ids=stage4_ids,
                negative_keywords=negative_keywords,
                query_vectors=negative_vectors,
                qdrant_client=qdrant_client,
                collection_name="welcome_subjective_vectors",
                threshold=0.50
            )

        stage3_scored = [(pid, score) for pid, score in stage3_scored if pid in stage4_ids]
        logging.info(f"✅ Stage 4 완료: {len(stage4_ids):,}명 (Negative 제거)")
    else:
        logging.info("   Negative 키워드 없음 또는 대상 없음 - Stage 4 스킵")

    logging.info("\n[최종 결과 정리]")
    
    final_panel_ids = [pid for pid, _ in stage3_scored[:final_limit]]
    
    result = {
        "final_panel_ids": final_panel_ids, 
        "total_count": len(final_panel_ids),
        "stage_details": {
            "stage1_objective": len(stage1_ids) if stage1_ids is not None else 0,
            "stage2_must_have": len(stage2_ids) if stage2_ids is not None else 0,
            "stage3_preference": len(stage3_scored),
            "stage4_negative": len(stage4_ids)
        },
        "classification": classification
    }

    logging.info(f"✅ 최종 결과: {len(final_panel_ids):,}명 (목표: {final_limit}명)")
    logging.info(f"   Stage 1 (Structured): {result['stage_details']['stage1_objective']:,}명")
    logging.info(f"   Stage 2 (Must-have): {result['stage_details']['stage2_must_have']:,}명")
    logging.info(f"   Stage 3 (Preference): {result['stage_details']['stage3_preference']:,}명")
    logging.info(f"   Stage 4 (Negative): {result['stage_details']['stage4_negative']:,}명")
    logging.info("="*80)
    
    return result
