import os
import re
import time
import logging
from typing import Optional, Tuple, List, Set
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# 수정된 import
from search_helpers import (
    initialize_embeddings, embedding_lock,
    search_welcome_objective, search_welcome_subjective, search_qpoll
)

load_dotenv()

def hybrid_search_parallel(
    classified_keywords: dict, 
    search_mode: str = "all", 
    limit: Optional[int] = None
) -> dict:
    """
    하이브리드 검색 (병렬 실행)
    1. Welcome 객관식 검색 (필터 셋 생성)
    2. 객관식 결과를 필터로 사용하여 Welcome 주관식 / QPoll 병렬 검색
    3. 결과 통합
    """
    welcome_obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
    welcome_subj_keywords = classified_keywords.get('welcome_keywords', {}).get('subjective', [])
    qpoll_data = classified_keywords.get('qpoll_keywords', {})
    
    logging.info("📌 2단계: 하이브리드 검색 (병렬 실행)")
    start_time = time.time()
    
    # 1. 객관식 검색 (필터 셋)을 *먼저* 실행
    logging.info("   🔄 Welcome 객관식 검색 (필터 셋 생성)...")
    panel_id1 = search_welcome_objective(welcome_obj_keywords)
    logging.info(f"   ✅ Welcome 객관식 완료: {len(panel_id1):,}명 (필터 셋)")
    
    # 2. 주관식/QPoll을 병렬 실행 (필터 셋 전달)
    panel_id2 = set()
    panel_id3 = set()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        
        if welcome_subj_keywords:
            logging.info("   ⚡ Welcome 주관식 시작 (필터 적용)")
            def subjective_search_with_lock(*args, **kwargs):
                # 임베딩 모델 로딩은 스레드 안전해야 함
                with embedding_lock:
                    return search_welcome_subjective(*args, **kwargs)
            futures['subjective'] = executor.submit(
                subjective_search_with_lock, 
                welcome_subj_keywords,
                pre_filter_panel_ids=panel_id1
            )
        
        qpoll_keywords = qpoll_data.get('keywords')
        if qpoll_keywords:
            logging.info("   ⚡ QPoll 시작 (필터 적용)")
            def qpoll_search_with_lock(*args, **kwargs):
                with embedding_lock:
                    return search_qpoll(*args, **kwargs)
            futures['qpoll'] = executor.submit(
                qpoll_search_with_lock,
                qpoll_data.get('survey_type'),
                qpoll_keywords,
                pre_filter_panel_ids=panel_id1
            )
        
        # 결과 수집
        for key, future in futures.items():
            try:
                result = future.result(timeout=60) # 타임아웃 60초로 연장
                if key == 'subjective':
                    panel_id2 = result
                    logging.info(f"   ✅ Welcome 주관식 완료: {len(panel_id2):,}명")
                elif key == 'qpoll':
                    panel_id3 = result
                    logging.info(f"   ✅ QPoll 완료: {len(panel_id3):,}명")
            except Exception as e:
                if isinstance(e, TimeoutError):
                    logging.error(f"   ❌ {key} 검색 시간 초과 (60초 이상 소요)", exc_info=False)

                logging.error(f"   ❌ {key} 검색 실패: {e}", exc_info=True)
                if key == 'subjective':
                    panel_id2 = set()
                elif key == 'qpoll':
                    panel_id3 = set()
    
    elapsed = time.time() - start_time
    logging.info(f"⚡ 병렬 검색 완료: {elapsed:.2f}초")
    
    # 3. 결과 통합
    all_sets = [s for s in [panel_id1, panel_id2, panel_id3] if s]
    results = {}
    
    # 교집합
    if not all_sets:
        intersection_panel_ids = []
        intersection_scores = {}
    elif len(all_sets) == 1:
        intersection_panel_ids = list(all_sets[0])
        intersection_scores = {panel_id: 1.0 for panel_id in intersection_panel_ids}
    else:
        intersection_set = set.intersection(*all_sets)
        intersection_panel_ids = list(intersection_set)
        intersection_scores = {panel_id: float(len(all_sets)) for panel_id in intersection_panel_ids}
    
    results['intersection'] = {
        'panel_ids': intersection_panel_ids,
        'count': len(intersection_panel_ids),
        'scores': intersection_scores
    }
    
    # 합집합
    union_set = set.union(*all_sets) if all_sets else set()
    union_scores = {
        panel_id: sum([1 if panel_id in s else 0 for s in [panel_id1, panel_id2, panel_id3]]) 
        for panel_id in union_set
    }
    union_panel_ids = sorted(union_set, key=lambda x: union_scores[x], reverse=True)
    
    results['union'] = {
        'panel_ids': union_panel_ids,
        'count': len(union_panel_ids),
        'scores': union_scores
    }
    
    # 가중치
    weights = {'panel_id1': 0.4, 'panel_id2': 0.3, 'panel_id3': 0.3}
    weighted_panel_ids = []
    weighted_scores = {}
    if union_set:
        for panel_id in union_set:
            score = 0.0
            if panel_id in panel_id1:
                score += weights['panel_id1']
            if panel_id in panel_id2:
                score += weights['panel_id2']
            if panel_id in panel_id3:
                score += weights['panel_id3']
            if score > 0: 
                weighted_scores[panel_id] = score
        
        weighted_panel_ids = sorted(
            weighted_scores.keys(), 
            key=lambda x: weighted_scores[x], 
            reverse=True
        )
    
    results['weighted'] = {
        'panel_ids': weighted_panel_ids,
        'count': len(weighted_panel_ids),
        'scores': weighted_scores,
        'weights': weights
    }
    
    # 검색 결과 요약 로그
    logging.info(f"📊 검색 결과 요약: Welcome(Obj)={len(panel_id1):,}, Welcome(Subj)={len(panel_id2):,}, QPoll={len(panel_id3):,}")
    logging.info(f"   -> 교집합={results['intersection']['count']:,}, 합집합={results['union']['count']:,}, 가중치={results['weighted']['count']:,}")
    
    # 4. Limit 처리
    if limit is not None and limit > 0:
        logging.info(f"🎯 {limit}명 목표 충족 로직 실행...")
        
        final_panel_ids = []
        match_scores = {}
        added_panel_ids_set = set()
        
        intersection_ids = results['intersection']['panel_ids']
        weighted_scores_map = results['weighted']['scores']
        
        # 1순위: 교집합 (가중치 점수순 정렬)
        sorted_intersection_ids = sorted(
            intersection_ids,
            key=lambda pid: weighted_scores_map.get(pid, 0), 
            reverse=True
        )
        
        for panel_id in sorted_intersection_ids:
            if len(final_panel_ids) < limit:
                final_panel_ids.append(panel_id)
                added_panel_ids_set.add(panel_id)
                match_scores[panel_id] = weighted_scores_map.get(panel_id, 0.0)
            else:
                break
        
        logging.info(f"   1순위(교집합) 충족: {len(final_panel_ids):,} / {limit:,}명")
        
        # 2순위: 가중치
        if len(final_panel_ids) < limit:
            weighted_ids = results['weighted']['panel_ids']
            for panel_id in weighted_ids:
                if len(final_panel_ids) >= limit:
                    break
                if panel_id not in added_panel_ids_set:
                    final_panel_ids.append(panel_id)
                    added_panel_ids_set.add(panel_id)
                    match_scores[panel_id] = weighted_scores_map.get(panel_id, 0.0)
            
            logging.info(f"   2순위(가중치) 충족: {len(final_panel_ids):,} / {limit:,}명")
    
    else:
        # Limit 없으면 search_mode에 따라 결과 선택
        logging.info(f"ℹ️  Limit 미지정. '{search_mode}' 모드 결과 반환.")
        if search_mode == 'intersection':
            final_panel_ids = results['intersection']['panel_ids']
            match_scores = results['intersection']['scores']
        elif search_mode == 'union':
            final_panel_ids = results['union']['panel_ids']
            match_scores = results['union']['scores']
        else:  # 'weighted' 또는 'all'
            final_panel_ids = results['weighted']['panel_ids']
            match_scores = results['weighted']['scores']
    
    return {
        "panel_id1": panel_id1,
        "panel_id2": panel_id2,
        "panel_id3": panel_id3,
        "final_panel_ids": final_panel_ids,
        "match_scores": match_scores,
        "results": results,
        "two_stage_used": False # 2단계 검색 로직은 병렬화로 대체됨
    }