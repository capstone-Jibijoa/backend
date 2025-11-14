"""
병렬 검색을 사용한 search_logic.py 최적화 버전
- Welcome 객관식/주관식, QPoll 검색을 병렬로 실행
- 예상 개선: 1.5초 → 0.8초 (순차 실행 대비 40% 단축)
"""
import os
import re
import time
from typing import Optional, Tuple, List, Set
import threading
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from db_logic_optimized import get_db_connection_context, get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchAny
from langchain_huggingface import HuggingFaceEmbeddings

# 기존 search_logic.py의 함수들 import
from search_logic import (
    EMBEDDINGS, CURRENT_YEAR, CATEGORY_MAPPING, VALID_REGIONS,
    expand_keywords, initialize_embeddings, extract_panel_id_from_payload,
    ConditionBuilder, search_welcome_objective, search_welcome_subjective,
    search_qpoll
)

load_dotenv()

embedding_lock = threading.Lock()


def hybrid_search_parallel(
    classified_keywords: dict, 
    search_mode: str = "all", 
    limit: Optional[int] = None
) -> dict:
    """
    하이브리드 검색 (병렬 실행)
    
    개선점:
    1. Welcome 객관식/주관식, QPoll을 동시에 실행
    2. 병렬 실행으로 전체 시간 단축
    3. ThreadPoolExecutor 사용
    
    예상 개선: 1.5초 → 0.8초 (40% 단축)
    
    Args:
        classified_keywords: LLM 분류 결과
        search_mode: 검색 모드 (all/weighted/union/intersection)
        limit: 인원 수 제한
    
    Returns:
        검색 결과 딕셔너리
    """
    welcome_obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
    welcome_subj_keywords = classified_keywords.get('welcome_keywords', {}).get('subjective', [])
    qpoll_data = classified_keywords.get('qpoll_keywords', {})
    
    use_two_stage = len(welcome_obj_keywords) > 0 and len(welcome_subj_keywords) > 0
    
    print(f"\n📌 2단계: 하이브리드 검색 (병렬 실행)")
    start_time = time.time()
    
    # 병렬 실행
    # ⭐️ 1. [수정] 객관식 검색(필터 셋)을 *먼저* 실행
    print(f"   🔄 Welcome 객관식 검색 (필터 셋 생성)...")
    panel_id1 = search_welcome_objective(welcome_obj_keywords)
    print(f"   ✅ Welcome 객관식 완료: {len(panel_id1):,}명 (필터 셋)")
    
    # ⭐️ 2. [수정] 주관식/QPoll을 병렬 실행 (필터 셋 전달)
    panel_id2 = set()
    panel_id3 = set()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        
        # Welcome 주관식 (병렬)
        if welcome_subj_keywords:
            print(f"   ⚡ Welcome 주관식 시작 (필터 적용)")
            
            def subjective_search_with_lock(*args, **kwargs):
                with embedding_lock:
                    print("   (Lock 획득: Welcome 주관식)")
                    return search_welcome_subjective(*args, **kwargs)

            futures['subjective'] = executor.submit(
                subjective_search_with_lock, 
                welcome_subj_keywords,
                pre_filter_panel_ids=panel_id1  # ⭐️ 필터 전달
            )
        
        # QPoll (병렬)
        qpoll_keywords = qpoll_data.get('keywords')
        if qpoll_keywords:
            print(f"   ⚡ QPoll 시작 (필터 적용)")

            def qpoll_search_with_lock(*args, **kwargs):
                with embedding_lock:
                    print("   (Lock 획득: QPoll)")
                    return search_qpoll(*args, **kwargs)

            futures['qpoll'] = executor.submit(
                qpoll_search_with_lock,
                qpoll_data.get('survey_type'),
                qpoll_keywords,
                pre_filter_panel_ids=panel_id1  # ⭐️ 필터 전달
            )
        
        # 결과 수집
        for key, future in futures.items():
            try:
                result = future.result(timeout=30)
                if key == 'subjective':
                    panel_id2 = result
                    print(f"   ✅ Welcome 주관식 완료: {len(panel_id2):,}명")
                elif key == 'qpoll':
                    panel_id3 = result
                    print(f"   ✅ QPoll 완료: {len(panel_id3):,}명")
            except Exception as e:
                print(f"   ❌ {key} 검색 실패: {e}")
                if key == 'subjective':
                    panel_id2 = set()
                elif key == 'qpoll':
                    panel_id3 = set()
    
    elapsed = time.time() - start_time
    print(f"\n⚡ 병렬 검색 완료: {elapsed:.2f}초\n")
    
    # ⭐️ 3. [수정] 결과 통합 (위 3단계와 동일한 로직)
    
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
    if not all_sets:
        union_panel_ids = []
        union_scores = {}
    else:
        union_set = set.union(*all_sets)
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
    if union_set: # 객관식 결과가 있을 때만 가중치 계산
        
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
    
    # 최종 요약
    print(f"{'='*70}")
    print(f"📊 검색 결과 요약")
    print(f"{'='*70}")
    print(f"Welcome 객관식: {len(panel_id1):,}명")
    print(f"Welcome 주관식: {len(panel_id2):,}명")
    print(f"QPoll: {len(panel_id3):,}명")
    print(f"")
    print(f"교집합: {results['intersection']['count']:,}명")
    print(f"합집합: {results['union']['count']:,}명")
    print(f"가중치: {results['weighted']['count']:,}명")
    print(f"{'='*70}\n")
    
    # limit 처리 (기존 로직)
    if limit is not None and limit > 0:
        print(f"🎯 {limit}명 목표 충족 로직 실행...")
        
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
        
        print(f"   1순위(교집합) 충족: {len(final_panel_ids):,} / {limit:,}명")
        
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
            
            print(f"   2순위(가중치) 충족: {len(final_panel_ids):,} / {limit:,}명")
    
    else:
        # limit 없으면 search_mode에 따라 결과 선택
        print(f"ℹ️  Limit 미지정. '{search_mode}' 모드 결과 반환.")
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
        "two_stage_used": False
    }


def hybrid_search_with_cache(
    classified_keywords: dict, 
    search_mode: str = "all", 
    limit: Optional[int] = None,
    use_cache: bool = True
) -> dict:
    """
    캐싱을 지원하는 하이브리드 검색
    
    개선점:
    - 동일한 키워드 조합은 Redis에서 캐시된 결과 반환
    - 병렬 검색 + 캐싱으로 최대 성능
    
    Args:
        classified_keywords: LLM 분류 결과
        search_mode: 검색 모드
        limit: 인원 수 제한
        use_cache: 캐시 사용 여부
    
    Returns:
        검색 결과 딕셔너리
    """
    if not use_cache:
        return hybrid_search_parallel(classified_keywords, search_mode, limit)
    
    try:
        import redis
        import hashlib
        import pickle
        
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            socket_connect_timeout=1
        )
        redis_client.ping()
    except:
        # Redis 사용 불가시 캐싱 없이 실행
        print("⚠️  Redis 사용 불가, 캐싱 없이 검색")
        return hybrid_search_parallel(classified_keywords, search_mode, limit)
    
    # 캐시 키 생성
    cache_data = {
        'keywords': classified_keywords,
        'mode': search_mode,
        'limit': limit
    }
    cache_key_str = str(sorted(str(cache_data).encode()))
    cache_key = f"search:{hashlib.md5(cache_key_str.encode()).hexdigest()}"
    
    # 캐시 확인
    try:
        cached = redis_client.get(cache_key)
        if cached:
            print("✅ 검색 결과 캐시 히트!")
            return pickle.loads(cached)
    except Exception as e:
        print(f"⚠️  캐시 조회 실패: {e}")
    
    # 실제 검색 (병렬)
    result = hybrid_search_parallel(classified_keywords, search_mode, limit)
    
    # 캐시 저장 (10분 TTL)
    try:
        redis_client.setex(cache_key, 600, pickle.dumps(result))
        print("💾 검색 결과 캐싱 완료 (TTL: 10분)")
    except Exception as e:
        print(f"⚠️  캐싱 실패: {e}")
    
    return result