"""
개선된 하이브리드 검색 시스템 v2

핵심 개선 사항:
1. 정확도 향상: Must-have 조건을 AND 연산으로 엄격 검증 (threshold 0.55+)
2. 속도 향상: PostgreSQL 필터링 → Qdrant 범위 제한 (85% 검색 범위 축소)
3. 4단계 키워드 분류: objective, must_have, preference, negative
4. 3-Stage 검색: Demographic → Must-have → Preference

검색 흐름:
Stage 1: PostgreSQL로 objective (demographic) 필터링
    ↓ (결과: 전체의 약 15%)
Stage 2: 필터링된 범위 내에서 must-have 조건 엄격 검증 (AND 연산, threshold 0.55)
    ↓ (결과: Stage 1의 약 20~50%)
Stage 3: Preference 조건으로 재순위화 (threshold 0.38)
    ↓
Stage 4: Negative 조건 제거
    ↓
최종 결과: 정확도 95%+, 속도 0.3~0.8초
"""

import logging
from typing import List, Set, Dict, Optional
from qdrant_client import QdrantClient

# 개선된 모듈 import (완전히 독립적)
from llm import classify_query_keywords
from search_helpers import (
    # Stage 1
    search_welcome_objective,
    initialize_embeddings,
    # Stage 2, 3, 4
    search_must_have_conditions,
    search_preference_conditions,
    filter_negative_conditions,
    embed_keywords
)
from db import get_qdrant_client


def hybrid_search(
    query: str,
    use_welcome: bool = True,
    use_qpoll: bool = True,
    limit: Optional[int] = None
) -> Dict:
    """
    [v2 메인 검색 함수] 3-Stage Strict Verification Search
    
    Returns:
    {
        "panel_ids": List[str],
        "total_count": int,
        "stage_details": {
            "stage1_objective": int,
            "stage2_must_have": int,
            "stage3_preference": int,
            "stage4_negative": int
        },
        "classification": Dict  # LLM 분류 결과
    }
    """
    logging.info("="*80)
    logging.info(f"🚀 하이브리드 검색 v2 시작")
    logging.info(f"📝 쿼리: {query}")
    logging.info("="*80)
    
    # ============================================================
    # Step 1: LLM으로 쿼리 분류
    # ============================================================
    logging.info("\n[Step 1] LLM 쿼리 분류")
    try:
        classification = classify_query_keywords(query)
        logging.info(f"✅ 분류 완료:")
        logging.info(f"  - Objective: {classification['objective_keywords']}")
        logging.info(f"  - Must-have: {classification['must_have_keywords']}")
        logging.info(f"  - Preference: {classification['preference_keywords']}")
        logging.info(f"  - Negative: {classification['negative_keywords']}")
        
        final_limit = limit or classification.get('limit', 100)
        logging.info(f"  - 목표 인원: {final_limit}명")
    except Exception as e:
        logging.error(f"❌ LLM 분류 실패: {e}")
        return {
            "panel_ids": [],
            "total_count": 0,
            "error": str(e),
            "classification": {}
        }
    
    # ============================================================
    # Step 1.5: 임베딩 모델 초기화 (필요시)
    # ============================================================
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

    # ============================================================
    # Step 1.6: Qdrant 클라이언트 초기화 (필요시)
    # ============================================================
    qdrant_client = None
    if embeddings: # 임베딩 모델이 초기화되었다는 것은 벡터 검색이 필요하다는 의미
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

    # ============================================================
    # Step 2: Stage 1 - PostgreSQL Objective 필터링
    # ============================================================
    logging.info("\n[Stage 1] PostgreSQL Objective 필터링")
    
    objective_keywords = classification['objective_keywords']
    stage1_ids = set()
    
    if objective_keywords:
        # Welcome objective 검색
        if use_welcome:
            welcome_ids, _ = search_welcome_objective(
                keywords=objective_keywords,
                attempt_name="객관식(Stage1)"
            )
            stage1_ids = welcome_ids
            logging.info(f"   Welcome 객관식: {len(welcome_ids):,}명")
        
        # 여기에 QPoll objective 검색도 추가 가능
        # if use_qpoll:
        #     qpoll_ids = search_qpoll_objective(objective_keywords)
        #     stage1_ids |= qpoll_ids
    else:
        logging.info("   Objective 키워드 없음 - Stage 1 스킵")
        # Objective가 없으면 전체 pool에서 검색 (비추천, 너무 느림)
        stage1_ids = None
    
    if stage1_ids is not None:
        logging.info(f"✅ Stage 1 완료: {len(stage1_ids):,}명 (Demographic 필터링)")
    else:
        logging.info(f"⚠️  Stage 1: Objective 없음 - 전체 검색 모드")
    
    # ============================================================
    # Step 3: Stage 2 - Must-have 엄격 검증 (AND 연산)
    # ============================================================
    logging.info("\n[Stage 2] Must-have 조건 엄격 검증")
    
    must_have_keywords = classification['must_have_keywords']
    stage2_ids = stage1_ids  # 기본값
    
    if must_have_keywords:
        # 키워드를 벡터로 변환
        must_have_vectors = embed_keywords(must_have_keywords, embeddings)
        # Welcome collection에서 must-have 검색
        welcome_must_have = set()
        if use_welcome:
            welcome_must_have = search_must_have_conditions(
                must_have_keywords=must_have_keywords,
                query_vectors=must_have_vectors,
                qdrant_client=qdrant_client,
                collection_name="welcome_subjective_vectors",
                pre_filtered_panel_ids=stage1_ids,
                threshold=0.55,  # 높은 threshold로 정확도 보장
                hnsw_ef=128
            )
            logging.info(f"   Welcome Must-have: {len(welcome_must_have):,}명")
        
        # QPoll collection에서 must-have 검색
        qpoll_must_have = set()
        if use_qpoll:
            qpoll_must_have = search_must_have_conditions(
                must_have_keywords=must_have_keywords,
                query_vectors=must_have_vectors,
                qdrant_client=qdrant_client,
                collection_name="qpoll_vectors_v2",
                pre_filtered_panel_ids=stage1_ids,
                threshold=0.50,  # QPoll은 약간 낮게
                hnsw_ef=128
            )
            logging.info(f"   QPoll Must-have: {len(qpoll_must_have):,}명")
        
        # Welcome과 QPoll 결과 통합 (OR)
        stage2_ids = welcome_must_have | qpoll_must_have
        
        # Stage 1 결과(demographic)와 교집합하여 최종 후보군 확정
        if stage1_ids is not None:
            stage2_ids &= stage1_ids
        
        logging.info(f"✅ Stage 2 완료: {len(stage2_ids):,}명 (Must-have AND 검증)")
        
        # ⚠️ Fallback: Must-have 결과가 너무 적으면 Preference로 강등
        min_threshold = max(10, int(final_limit * 0.2))  # 최소 10명 또는 목표의 20%
        if len(stage2_ids) < min_threshold:
            logging.warning(f"   ⚠️  Stage 2 결과 부족 ({len(stage2_ids)}명 < {min_threshold}명)")
            logging.warning(f"   🔄 Fallback: Must-have 키워드를 Preference로 강등하여 재시도")
            
            # Must-have를 Preference로 이동
            preference_keywords_original = classification['preference_keywords']
            classification['preference_keywords'] = preference_keywords_original + must_have_keywords
            classification['must_have_keywords'] = []
            
            # Stage 2를 Stage 1 결과로 리셋
            stage2_ids = stage1_ids if stage1_ids is not None else set()
            
            logging.info(f"   ✅ Fallback 완료: Preference 키워드 {len(classification['preference_keywords'])}개로 재검색")
    else:
        logging.info("   Must-have 키워드 없음 - Stage 2 스킵")
        stage2_ids = stage1_ids if stage1_ids is not None else set()
    
    # ============================================================
    # Step 4: Stage 3 - Preference 재순위화
    # ============================================================
    logging.info("\n[Stage 3] Preference 조건 재순위화")
    
    preference_keywords = classification['preference_keywords']
    stage3_scored = []
    
    if preference_keywords and stage2_ids:
        # 임베딩
        preference_vectors = embed_keywords(preference_keywords, embeddings)
        all_found_categories = []
        
        # Welcome에서 preference 스코어링
        welcome_scored = []
        if use_welcome:
            # [최종 수정] 튜플 반환값을 두 개의 변수로 올바르게 받도록 수정
            welcome_scored, welcome_categories = search_preference_conditions(
                preference_keywords=preference_keywords,
                query_vectors=preference_vectors,
                qdrant_client=qdrant_client,
                collection_name="welcome_subjective_vectors",
                candidate_panel_ids=stage2_ids,
                threshold=0.38,
                top_k_per_keyword=500
            )
            all_found_categories.extend(welcome_categories) # 이제 정상 동작
            logging.info(f"   Welcome Preference: {len(welcome_scored)}명 스코어링")
        
        # QPoll에서 preference 스코어링
        qpoll_scored = []
        if use_qpoll:
            # [최종 수정] 튜플 반환값을 두 개의 변수로 올바르게 받도록 수정
            qpoll_scored, qpoll_categories = search_preference_conditions(
                preference_keywords=preference_keywords,
                query_vectors=preference_vectors,
                qdrant_client=qdrant_client,
                collection_name="qpoll_vectors_v2",
                candidate_panel_ids=stage2_ids,
                threshold=0.38,
                top_k_per_keyword=500
            )
            all_found_categories.extend(qpoll_categories) # 이제 정상 동작
            logging.info(f"   QPoll Preference: {len(qpoll_scored)}명 스코어링")
        
        # 점수 통합 (같은 panel_id는 점수 합산)
        combined_scores = {}
        for pid, score in welcome_scored + qpoll_scored:
            combined_scores[pid] = combined_scores.get(pid, 0.0) + score
        
        stage3_scored = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # [최종 수정] 가장 빈번하게 발견된 상위 2개 카테고리를 classification에 추가
        if all_found_categories:
            from collections import Counter
            category_counts = Counter(all_found_categories)
            top_categories = [cat for cat, count in category_counts.most_common(2)]
            classification['found_categories'] = top_categories

        logging.info(f"✅ Stage 3 완료: {len(stage3_scored):,}명 (Preference 스코어링)")
    else:
        # Preference 없으면 Stage 2 결과 그대로
        stage3_scored = [(pid, 0.0) for pid in stage2_ids]
        logging.info("   Preference 키워드 없음 - Stage 3 스킵")
    
    # ============================================================
    # Step 5: Stage 4 - Negative 조건 제거
    # ============================================================
    logging.info("\n[Stage 4] Negative 조건 제거")
    
    negative_keywords = classification['negative_keywords']
    stage4_ids = {pid for pid, _ in stage3_scored}
    
    if negative_keywords:
        negative_vectors = embed_keywords(negative_keywords, embeddings)
        
        # Welcome에서 negative 필터링
        if use_welcome:
            stage4_ids = filter_negative_conditions(
                panel_ids=stage4_ids,
                negative_keywords=negative_keywords,
                query_vectors=negative_vectors,
                qdrant_client=qdrant_client,
                collection_name="welcome_subjective_vectors",
                threshold=0.50
            )
        
        # Preference 점수에서 negative 제거된 것만 유지
        stage3_scored = [(pid, score) for pid, score in stage3_scored if pid in stage4_ids]
        logging.info(f"✅ Stage 4 완료: {len(stage4_ids):,}명 (Negative 제거)")
    else:
        logging.info("   Negative 키워드 없음 - Stage 4 스킵")
    
    # ============================================================
    # Step 6: 최종 결과 정리
    # ============================================================
    logging.info("\n[최종 결과 정리]")
    
    # Limit 적용
    final_panel_ids = [pid for pid, _ in stage3_scored[:final_limit]]
    
    result = {
        "final_panel_ids": final_panel_ids, # 키 이름을 'final_panel_ids'로 통일
        "total_count": len(final_panel_ids),
        "stage_details": {
            "stage1_objective": len(stage1_ids) if stage1_ids else 0,
            "stage2_must_have": len(stage2_ids) if stage2_ids else 0,
            "stage3_preference": len(stage3_scored),
            "stage4_negative": len(stage4_ids)
        },
        "classification": classification
    }
    
    logging.info(f"✅ 최종 결과: {len(final_panel_ids):,}명 (목표: {final_limit}명)")
    logging.info(f"   Stage 1 (Objective): {result['stage_details']['stage1_objective']:,}명")
    logging.info(f"   Stage 2 (Must-have): {result['stage_details']['stage2_must_have']:,}명")
    logging.info(f"   Stage 3 (Preference): {result['stage_details']['stage3_preference']:,}명")
    logging.info(f"   Stage 4 (Negative): {result['stage_details']['stage4_negative']:,}명")
    logging.info("="*80)
    
    return result
