import os
import re
import time
import logging
from typing import Optional, Tuple, List, Set
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from db import get_qdrant_client 
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
    - (수정) Qdrant 클라이언트를 미리 생성하여 스레드에 전달
    """
    welcome_obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
    welcome_subj_keywords = classified_keywords.get('welcome_keywords', {}).get('subjective', [])
    qpoll_data = classified_keywords.get('qpoll_keywords', {})
    
    logging.info("📌 2단계: 하이브리드 검색 (병렬 실행)")
    start_time = time.time()
    
    # 1. 객관식 검색
    logging.info("   🔄 Welcome 객관식 검색...")
    panel_id1 = search_welcome_objective(welcome_obj_keywords)
    logging.info(f"   ✅ Welcome 객관식 완료: {len(panel_id1):,}명")
    
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
    
    with embedding_lock:
        try:
            embeddings = initialize_embeddings() 
            if welcome_subj_keywords:
                def flatten(items):
                    flat = []
                    for item in items:
                        if isinstance(item, list): flat.extend(flatten(item))
                        elif item is not None: flat.append(str(item))
                    return flat
                subj_query_text = " ".join(flatten(welcome_subj_keywords))
                if subj_query_text:
                    subjective_vector = embeddings.embed_query(subj_query_text)
            
            qpoll_keywords = qpoll_data.get('keywords')
            if qpoll_keywords:
                qpoll_query_text = " ".join(qpoll_keywords)
                if qpoll_query_text:
                    qpoll_vector = embeddings.embed_query(qpoll_query_text)
        except Exception as e:
            logging.error(f"   ❌ 임베딩 생성 중 오류: {e}", exc_info=True)

    # 4. 네트워크 I/O 작업만 병렬 실행 (클라이언트 전달)
    panel_id2 = set()
    panel_id3 = set()
    
    # qdrant_client가 성공적으로 생성되었을 때만 실행
    if qdrant_client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            if subjective_vector:
                logging.info("   ⚡ Welcome 주관식 시작 (Top-K)")
                futures['subjective'] = executor.submit(
                    search_welcome_subjective, 
                    query_vector=subjective_vector,
                    qdrant_client=qdrant_client, 
                    keywords=welcome_subj_keywords
                )
            
            if qpoll_vector:
                logging.info("   ⚡ QPoll 시작 (Top-K)")
                futures['qpoll'] = executor.submit(
                    search_qpoll,
                    query_vector=qpoll_vector,
                    qdrant_client=qdrant_client, 
                    keywords=qpoll_keywords
                )
            
            for key, future in futures.items():
                try:
                    result = future.result(timeout=60)
                    if key == 'subjective': panel_id2 = result; logging.info(f"   ✅ Welcome 주관식 완료: {len(panel_id2):,}명")
                    elif key == 'qpoll': panel_id3 = result; logging.info(f"   ✅ QPoll 완료: {len(panel_id3):,}명")
                except Exception as e:
                    if isinstance(e, TimeoutError): logging.error(f"   ❌ {key} 검색 시간 초과", exc_info=False)
                    logging.error(f"   ❌ {key} 검색 실패: {e}", exc_info=True)
                    if key == 'subjective': panel_id2 = set()
                    elif key == 'qpoll': panel_id3 = set()
    
    elapsed = time.time() - start_time
    logging.info(f"⚡ 병렬 검색 완료: {elapsed:.2f}초")

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
    
    weights = {'panel_id1': 0.4, 'panel_id2': 0.3, 'panel_id3': 0.3}
    weighted_panel_ids = []
    weighted_scores = {}
    if union_set:
        for panel_id in union_set:
            score = 0.0
            if panel_id in panel_id1: score += weights['panel_id1']
            if panel_id in panel_id2: score += weights['panel_id2']
            if panel_id in panel_id3: score += weights['panel_id3']
            if score > 0: weighted_scores[panel_id] = score
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
        "results": results, "two_stage_used": False
    }