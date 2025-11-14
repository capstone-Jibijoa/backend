import os
import re
import time
import logging
from typing import Optional, Tuple, List, Set, Dict
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client.http.models import Filter, FieldCondition, MatchAny

from db import get_qdrant_client 
from search_helpers import (
    initialize_embeddings,
    build_welcome_query_conditions,
    search_welcome_objective, search_welcome_subjective, search_qpoll
)

load_dotenv()

def hybrid_search_parallel(
    classified_keywords: dict, 
    search_mode: str = "all", 
    limit: Optional[int] = None,
    is_comparison: bool = False # [신규] 비교 그룹 검색 플래그
) -> dict:
    """
    하이브리드 검색 (병렬 실행)
    - (수정) Qdrant 클라이언트를 미리 생성하여 스레드에 전달
    """
    welcome_obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
    welcome_subj_keywords = classified_keywords.get('welcome_keywords', {}).get('subjective', [])
    qpoll_data = classified_keywords.get('qpoll_keywords', {})
    
    # 쿼리 복잡도에 따른 동적 임계값 설정
    def _get_dynamic_threshold(objective_keywords: List[str]) -> int:
        num_keywords = len(objective_keywords)
        if num_keywords <= 1:
            # 광범위한 쿼리 (e.g., "20대") -> 높은 임계값
            return 1000
        elif num_keywords <= 3:
            # 일반적인 쿼리 (e.g., "서울 30대 남성") -> 기본 임계값
            return 500
        else:
            # 매우 구체적인 쿼리 (e.g., "서울 30대 남성 사무직") -> 낮은 임계값
            return 200

    TWO_STAGE_THRESHOLD = _get_dynamic_threshold(welcome_obj_keywords)
    logging.info(f"   ⚙️  동적 임계값 설정: {TWO_STAGE_THRESHOLD} (객관식 키워드 수: {len(welcome_obj_keywords)})")
    two_stage_used = False

    logging.info("📌 2단계: 하이브리드 검색 (병렬 실행)")
    start_time = time.time()
    
    # 2. Qdrant 클라이언트를 메인 스레드에서 *한 번만* 생성
    qdrant_client = None
    try:
        qdrant_client = get_qdrant_client()
        if not qdrant_client:
            logging.error("   ❌ Qdrant 클라이언트 생성 실패. 벡터 검색 중단.")
    except Exception as e:
        logging.error(f"   ❌ Qdrant 클라이언트 생성 중 오류: {e}", exc_info=True)
    
    # 3. 임베딩 미리 수행 (Lock 사용)
    subjective_vector = None
    qpoll_vector = None
    embeddings = None
    
    try:
        embeddings = initialize_embeddings()
        
        def flatten(items):
            flat = []
            for item in items:
                if isinstance(item, list): flat.extend(flatten(item))
                elif item is not None: flat.append(str(item))
            return flat
        
        # Subjective 벡터 생성 
        if welcome_subj_keywords:
            expansion_keywords = classified_keywords.get('welcome_keywords', {}).get('subjective_expansion', [])
            combined_keywords = welcome_subj_keywords + expansion_keywords
            subj_query_text = " ".join(flatten(combined_keywords))
            if subj_query_text:
                subjective_vector = embeddings.embed_query(subj_query_text)
    
        # QPoll 벡터 생성 
        qpoll_keywords = qpoll_data.get('keywords')
        if qpoll_keywords:
            qpoll_query_text = " ".join(qpoll_keywords)
            if qpoll_query_text:
                qpoll_vector = embeddings.embed_query(qpoll_query_text)
    except Exception as e:
        logging.error(f"   ❌ 임베딩 생성 중 오류: {e}", exc_info=True)

    # 4. [수정] 모든 DB/네트워크 I/O 작업을 병렬로 실행
    panel_id1 = set()
    panel_id2 = set()
    panel_id3 = set()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        
        # 작업 1: 객관식 검색
        if welcome_obj_keywords:
            logging.info("   ⚡ Welcome 객관식 검색 스레드 시작...")
            futures['objective'] = executor.submit(search_welcome_objective, welcome_obj_keywords)

        # 작업 2 & 3: 벡터 검색 (임베딩이 성공했을 경우)
        if qdrant_client:
            vector_search_filter = None # [수정] 변수를 바깥 스코프에서 미리 초기화
            # 객관식 검색이 끝날 때까지 기다렸다가 2단계 전략 결정
            if 'objective' in futures:
                try:
                    # 객관식 결과 먼저 받아오기
                    panel_id1, unhandled_obj_keywords = futures['objective'].result(timeout=60)
                    logging.info(f"   ✅ Welcome 객관식 완료: {len(panel_id1):,}명")
                    
                    if unhandled_obj_keywords:
                        logging.warning(f"   ⚠️  객관식 키워드 일부가 벡터 검색으로 전환됩니다: {unhandled_obj_keywords}")
                        welcome_subj_keywords.extend(list(unhandled_obj_keywords))
                        welcome_subj_keywords = list(dict.fromkeys(welcome_subj_keywords))
                        # 재실행 필요 시, 벡터 재생성 (캐싱되어 있다면 비용 적음)
                        if unhandled_obj_keywords and embeddings:
                             subj_query_text = " ".join(welcome_subj_keywords)
                             subjective_vector = embeddings.embed_query(subj_query_text)

                except Exception as e:
                    logging.error(f"   ❌ 객관식 검색 스레드 실패: {e}", exc_info=True)
                    panel_id1 = set() # 실패 시 빈 결과로 처리
                
                # 2단계 검색 전략 결정
                if len(panel_id1) >= TWO_STAGE_THRESHOLD:
                    logging.info(f"   ✨ 1단계 전략: 객관식 결과({len(panel_id1)}명)가 충분하여, 이 ID를 필터로 벡터 검색 실행")
                    vector_search_filter = Filter(must=[FieldCondition(key="panel_id", match=MatchAny(any=list(panel_id1)))])
                    two_stage_used = True

            if subjective_vector:
                logging.info("   ⚡ Welcome 주관식 검색 스레드 시작 (Top-K)")
                subjective_filter = None
                if vector_search_filter:
                    subjective_filter = Filter(must=[FieldCondition(key="metadata.panel_id", match=MatchAny(any=list(panel_id1)))])
                
                if any(kw in subj_query_text for kw in ['직무', '직업', '업무']):
                    logging.info("   -&gt; '직무' 관련 검색으로 '무직'/'학생' 제외 필터 활성화")
  
                    existing_must_conditions = subjective_filter.must if subjective_filter and subjective_filter.must else []
                    job_must_not_conditions = [
                        FieldCondition(key="metadata.job_title_raw", match={"any": ["무직", "학생", "대학생", "대학원생"]})
                    ]

                    subjective_filter = Filter(must=existing_must_conditions, must_not=job_must_not_conditions)
                futures['subjective'] = executor.submit(search_welcome_subjective, subjective_vector, qdrant_client, combined_keywords, subjective_filter)
            
            if qpoll_vector:
                logging.info("   ⚡ QPoll 검색 스레드 시작 (Top-K)")
                futures['qpoll'] = executor.submit(search_qpoll, qpoll_vector, qdrant_client, qpoll_data.get('keywords'), vector_search_filter)

        # 나머지 결과 취합
        if 'subjective' in futures:
            try:
                panel_id2 = futures['subjective'].result(timeout=60)
                logging.info(f"   ✅ Welcome 주관식 완료: {len(panel_id2):,}명")
            except Exception as e:
                logging.error(f"   ❌ 주관식 검색 스레드 실패: {e}", exc_info=True)
                panel_id2 = set()
        
        if 'qpoll' in futures:
            try:
                panel_id3 = futures['qpoll'].result(timeout=60)
                logging.info(f"   ✅ QPoll 완료: {len(panel_id3):,}명")
            except Exception as e:
                logging.error(f"   ❌ QPoll 검색 스레드 실패: {e}", exc_info=True)
                panel_id3 = set()
    
    elapsed = time.time() - start_time
    logging.info(f"⚡ 병렬 검색 완료: {elapsed:.2f}초")

    # [신규] 2단계 검색 후처리: 객관식 결과가 부족했을 때, 벡터 검색 결과에 객관식 필터링 적용
    if welcome_obj_keywords and not two_stage_used and panel_id1:
        logging.info(f"   ✨ 2단계 전략: 객관식 결과({len(panel_id1)}명)가 부족하여, 벡터 검색 결과에 객관식 필터 적용")
        panel_id2 = panel_id2.intersection(panel_id1) if panel_id1 else panel_id2
        panel_id3 = panel_id3.intersection(panel_id1) if panel_id1 else panel_id3
        two_stage_used = True
        logging.info(f"   -> 필터 후: Welcome(Subj)={len(panel_id2):,}, QPoll={len(panel_id3):,}")

    # 5. 결과 통합 
    all_sets = [s for s in [panel_id1, panel_id2, panel_id3] if s]
    results = {}
    
    if not all_sets: intersection_panel_ids = []; intersection_scores = {}
    elif len(all_sets) == 1:
        intersection_panel_ids = list(all_sets[0])
        intersection_scores = {panel_id: 1.0 for panel_id in intersection_panel_ids}
    else:
        intersection_set = set.intersection(*all_sets)
        intersection_panel_ids = list(intersection_set)
        intersection_scores = {panel_id: float(len(all_sets)) for panel_id in intersection_panel_ids}
    
    results['intersection'] = { 'panel_ids': intersection_panel_ids, 'count': len(intersection_panel_ids), 'scores': intersection_scores }
    
    union_set = set.union(*all_sets) if all_sets else set()
    union_scores = { panel_id: sum([1 if panel_id in s else 0 for s in [panel_id1, panel_id2, panel_id3]]) for panel_id in union_set }
    union_panel_ids = sorted(union_set, key=lambda x: union_scores[x], reverse=True)
    results['union'] = { 'panel_ids': union_panel_ids, 'count': len(union_panel_ids), 'scores': union_scores }
    
    def _get_dynamic_weights(classification: Dict) -> Dict[str, float]:
        """쿼리 특성에 따라 동적으로 가중치를 계산합니다."""
        obj_kws = classification.get('welcome_keywords', {}).get('objective', [])
        subj_kws = classification.get('welcome_keywords', {}).get('subjective', [])
        qpoll_kws = classification.get('qpoll_keywords', {}).get('keywords', [])

        # 각 검색 소스의 기본 중요도 점수
        scores = {
            'panel_id1': 1.5 if obj_kws else 0.0,      # 객관식은 중요하므로 높은 기본 점수
            'panel_id2': 1.0 if subj_kws else 0.0,      # 주관식은 일반 점수
            'panel_id3': 1.2 if qpoll_kws else 0.0       # QPoll은 특정 행동/의견이므로 약간 더 중요
        }
        
        total_score = sum(scores.values())
        
        if total_score == 0:
            return {'panel_id1': 0.33, 'panel_id2': 0.33, 'panel_id3': 0.34} # 모든 키워드가 없는 경우

        # 점수를 정규화하여 총합이 1이 되도록 가중치 계산
        weights = {k: round(v / total_score, 2) for k, v in scores.items()}
        return weights

    weights = _get_dynamic_weights(classified_keywords)
    logging.info(f"   ⚖️  동적 가중치 적용: {weights}")

    weighted_panel_ids = []
    weighted_scores = {}
    if union_set:
        weighted_scores = {pid: (weights['panel_id1'] if pid in panel_id1 else 0) + 
                                (weights['panel_id2'] if pid in panel_id2 else 0) + 
                                (weights['panel_id3'] if pid in panel_id3 else 0) for pid in union_set}
        weighted_panel_ids = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)
    
    results['weighted'] = { 'panel_ids': weighted_panel_ids, 'count': len(weighted_panel_ids), 'scores': weighted_scores, 'weights': weights }
    
    logging.info(f"📊 검색 결과 요약: Welcome(Obj)={len(panel_id1):,}, Welcome(Subj)={len(panel_id2):,}, QPoll={len(panel_id3):,}")
    logging.info(f"   -> 교집합={results['intersection']['count']:,}, 합집합={results['union']['count']:,}, 가중치={results['weighted']['count']:,}")
    
    # 6. Limit 처리 
    if limit is not None and limit > 0:
        logging.info(f"🎯 {limit}명 목표 충족 로직 실행...")
        final_panel_ids = []; match_scores = {}; added_panel_ids_set = set()
        intersection_ids = results['intersection']['panel_ids']
        weighted_scores_map = results['weighted']['scores']
        sorted_intersection_ids = sorted(intersection_ids, key=lambda pid: weighted_scores_map.get(pid, 0), reverse=True)
        
        for panel_id in sorted_intersection_ids:
            if len(final_panel_ids) < limit:
                final_panel_ids.append(panel_id); added_panel_ids_set.add(panel_id)
                match_scores[panel_id] = weighted_scores_map.get(panel_id, 0.0)
            else: break
        logging.info(f"   1순위(교집합) 충족: {len(final_panel_ids):,} / {limit:,}명")
        
        if len(final_panel_ids) < limit:
            weighted_ids = results['weighted']['panel_ids']
            for panel_id in weighted_ids:
                if len(final_panel_ids) >= limit: break
                if panel_id not in added_panel_ids_set:
                    final_panel_ids.append(panel_id); added_panel_ids_set.add(panel_id)
                    match_scores[panel_id] = weighted_scores_map.get(panel_id, 0.0)
            logging.info(f"   2순위(가중치) 충족: {len(final_panel_ids):,} / {limit:,}명")
    else:
        logging.info(f"ℹ️  Limit 미지정. '{search_mode}' 모드 결과 반환.")
        if search_mode == 'intersection': final_panel_ids = results['intersection']['panel_ids']; match_scores = results['intersection']['scores']
        elif search_mode == 'union': final_panel_ids = results['union']['panel_ids']; match_scores = results['union']['scores']
        else: final_panel_ids = results['weighted']['panel_ids']; match_scores = results['weighted']['scores']
    
    return {
        "panel_id1": panel_id1, "panel_id2": panel_id2, "panel_id3": panel_id3,
        "final_panel_ids": final_panel_ids, "match_scores": match_scores,
        "results": results, "two_stage_used": two_stage_used
    }