"""
DB 집계 쿼리를 사용한 analysis_logic.py 최적화 버전
- 전체 DB 조회 대신 PostgreSQL의 집계 함수 사용
- 예상 개선: 0.2~1초 → 0.02~0.05초 (10~20배 개선!)
"""

import json
from typing import List, Dict, Any, Tuple
from collections import Counter
from utils import (
    extract_field_values,
    calculate_distribution,
    find_top_category,
    FIELD_NAME_MAP,
    WELCOME_OBJECTIVE_FIELDS,
    EXCLUDED_RAW_FIELDS,
    get_panels_data_from_db
)


def get_field_distribution_from_db(field_name: str, limit: int = 10) -> Dict[str, float]:
    """
    PostgreSQL에서 직접 집계하여 필드 분포 조회
    
    개선점:
    - 전체 데이터를 Python으로 가져오지 않음
    - DB에서 GROUP BY로 집계 후 결과만 전송
    - 메모리 사용량 대폭 감소
    - 속도 10~20배 향상
    
    Args:
        field_name: 조회할 필드명
        limit: 상위 N개만 조회 (기본 10개)
    
    Returns:
        {값: 비율(%)} 딕셔너리
    """
    from db_logic import get_db_connection
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print(f"❌ DB 연결 실패")
            return {}
        
        cur = conn.cursor()
        
        # 연령대 계산이 필요한 경우
        if field_name == "birth_year":
            query = f"""
                WITH age_groups AS (
                    SELECT 
                        CASE 
                            WHEN (2025 - (structured_data->>'birth_year')::int) < 20 THEN '10대'
                            WHEN (2025 - (structured_data->>'birth_year')::int) < 30 THEN '20대'
                            WHEN (2025 - (structured_data->>'birth_year')::int) < 40 THEN '30대'
                            WHEN (2025 - (structured_data->>'birth_year')::int) < 50 THEN '40대'
                            WHEN (2025 - (structured_data->>'birth_year')::int) < 60 THEN '50대'
                            ELSE '60대 이상'
                        END as age_group
                    FROM welcome_meta2
                    WHERE structured_data->>'birth_year' IS NOT NULL
                        AND structured_data->>'birth_year' ~ '^[0-9]+$'
                )
                SELECT 
                    age_group,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
                FROM age_groups
                GROUP BY age_group
                ORDER BY percentage DESC
                LIMIT {limit}
            """
        else:
            # 일반 필드 (region_major, gender 등)
            query = f"""
                SELECT 
                    structured_data->>'{field_name}' as value,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
                FROM welcome_meta2
                WHERE structured_data->>'{field_name}' IS NOT NULL
                    AND structured_data->>'{field_name}' != ''
                GROUP BY structured_data->>'{field_name}'
                ORDER BY percentage DESC
                LIMIT {limit}
            """
        
        cur.execute(query)
        rows = cur.fetchall()
        
        # {값: 비율} 딕셔너리 생성
        distribution = {}
        for row in rows:
            value = row[0]
            percentage = float(row[2])
            if value and percentage > 0:
                distribution[value] = percentage
        
        cur.close()
        
        print(f"   📊 DB 집계 완료: {field_name} ({len(distribution)}개 카테고리)")
        return distribution
        
    except Exception as e:
        print(f"❌ DB 집계 실패 ({field_name}): {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        if conn:
            conn.close()


def get_multiple_field_distributions(field_names: List[str], limit: int = 10) -> Dict[str, Dict[str, float]]:
    """
    여러 필드의 분포를 한 번에 조회 (병렬 처리)
    
    Args:
        field_names: 조회할 필드명 리스트
        limit: 각 필드당 상위 N개
    
    Returns:
        {필드명: {값: 비율}} 딕셔너리
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 각 필드별로 병렬 실행
        future_to_field = {
            executor.submit(get_field_distribution_from_db, field, limit): field
            for field in field_names
        }
        
        for future in as_completed(future_to_field):
            field = future_to_field[future]
            try:
                distribution = future.result(timeout=5)
                if distribution:
                    results[field] = distribution
            except Exception as e:
                print(f"❌ {field} 조회 실패: {e}")
    
    return results


def create_chart_data_optimized(
    keyword: str,
    field_name: str,
    korean_name: str,
    panels_data: List[Dict],
    use_full_db: bool = False,
    max_categories: int = 10
) -> Dict:
    """
    차트 데이터 생성 (최적화 버전)
    
    개선점:
    - use_full_db=True일 때 DB 집계 쿼리 사용
    - Python으로 전체 데이터를 가져오지 않음
    """
    # 전체 DB 기반 분석 (최적화!)
    if use_full_db:
        print(f"      → DB 집계로 '{field_name}' 분석 (최적화)")
        distribution = get_field_distribution_from_db(field_name, max_categories)
        
        if not distribution:
            return {
                "topic": korean_name,
                "description": f"'{keyword}' 관련 전체 데이터를 조회할 수 없습니다.",
                "ratio": "0.0%",
                "chart_data": []
            }
        
        top_category, top_ratio = find_top_category(distribution)
        description_prefix = f"전체 데이터 기준 '{keyword}' 분석:"
        
        return {
            "topic": f"{korean_name} 분포",
            "description": f"{description_prefix} {top_ratio}%가 '{top_category}'입니다.",
            "ratio": f"{top_ratio}%",
            "chart_data": [{
                "label": korean_name,
                "values": distribution
            }]
        }
    
    # 검색 결과 기반 분석 (기존 로직)
    else:
        values = extract_field_values(panels_data, field_name)
        
        if not values:
            return {
                "topic": korean_name,
                "description": f"'{keyword}' 관련 데이터가 부족합니다.",
                "ratio": "0.0%",
                "chart_data": []
            }
        
        # 분포 계산
        distribution = calculate_distribution(values)
        filtered_distribution = {k: v for k, v in distribution.items() if v > 0.0}
        
        if not filtered_distribution:
            return {
                "topic": korean_name,
                "description": f"'{keyword}' 관련 데이터가 부족합니다.",
                "ratio": "0.0%",
                "chart_data": []
            }
        
        # 상위 N개만 + 기타
        if len(filtered_distribution) > max_categories:
            sorted_items = sorted(filtered_distribution.items(), key=lambda x: x[1], reverse=True)
            top_items = dict(sorted_items[:max_categories - 1])
            other_sum = sum(v for k, v in sorted_items[max_categories - 1:])
            if other_sum > 0:
                top_items['기타'] = round(other_sum, 1)
            final_distribution = top_items
        else:
            final_distribution = filtered_distribution
        
        top_category, top_ratio = find_top_category(final_distribution)
        
        return {
            "topic": f"{korean_name} 분포",
            "description": f"'{keyword}' 검색 결과: {top_ratio}%가 '{top_category}'입니다.",
            "ratio": f"{top_ratio}%",
            "chart_data": [{
                "label": korean_name,
                "values": final_distribution
            }]
        }


def find_high_ratio_fields_optimized(
    panels_data: List[Dict], 
    exclude_fields: List[str], 
    threshold: float = 50.0,
    max_charts: int = 3
) -> List[Dict]:
    """
    높은 비율 필드 찾기 (병렬 처리)
    
    개선점:
    - 여러 필드를 병렬로 분석
    - 불필요한 필드는 미리 필터링
    """
    candidate_fields = []
    
    for field_name, korean_name in WELCOME_OBJECTIVE_FIELDS:
        # 제외 조건
        if field_name in EXCLUDED_RAW_FIELDS or field_name in exclude_fields:
            continue
        candidate_fields.append((field_name, korean_name))
    
    if not candidate_fields:
        return []
    
    print(f"   🔍 {len(candidate_fields)}개 필드 병렬 분석 중...")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    high_ratio_results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_field = {}
        
        for field_name, korean_name in candidate_fields:
            future = executor.submit(
                extract_field_values, panels_data, field_name
            )
            future_to_field[future] = (field_name, korean_name)
        
        for future in as_completed(future_to_field):
            field_name, korean_name = future_to_field[future]
            
            try:
                values = future.result(timeout=2)
                if not values:
                    continue
                
                distribution = calculate_distribution(values)
                filtered_distribution = {k: v for k, v in distribution.items() if v > 0.0}
                
                if not filtered_distribution:
                    continue
                
                top_category, top_ratio = find_top_category(filtered_distribution)
                
                if top_ratio >= threshold:
                    # 카테고리가 많으면 상위 10개만
                    if len(filtered_distribution) > 10:
                        sorted_items = sorted(filtered_distribution.items(), key=lambda x: x[1], reverse=True)
                        top_items = dict(sorted_items[:9])
                        other_sum = sum(v for k, v in sorted_items[9:])
                        if other_sum > 0:
                            top_items['기타'] = round(other_sum, 1)
                        filtered_distribution = top_items
                    
                    high_ratio_results.append({
                        "field": field_name,
                        "korean_name": korean_name,
                        "distribution": filtered_distribution,
                        "top_category": top_category,
                        "top_ratio": top_ratio
                    })
            except Exception as e:
                print(f"   ⚠️  {field_name} 분석 실패: {e}")
                continue
    
    # 비율 높은 순 정렬
    high_ratio_results.sort(key=lambda x: x["top_ratio"], reverse=True)
    
    return high_ratio_results[:max_charts]


def analyze_search_results_optimized(
    query: str,
    classified_keywords: dict,
    panel_id_list: List[str]
) -> Tuple[Dict, int]:
    """
    검색 결과 분석 (최적화 버전)
    
    개선점:
    1. DB 집계 쿼리 사용 (전체 DB 기준 차트)
    2. 병렬 처리 (높은 비율 필드 분석)
    3. 불필요한 데이터 로딩 최소화
    
    예상 속도: 1~2초 → 0.2~0.5초
    """
    print(f"\n{'='*70}")
    print(f"📊 분석 시작 (최적화)")
    print(f"   질의: {query}")
    print(f"   panel_id 수: {len(panel_id_list)}개")
    print(f"{'='*70}\n")
    
    if not panel_id_list:
        return {
            "query": query,
            "total_count": 0,
            "main_summary": "검색 결과가 없습니다.",
            "charts": []
        }, 200
    
    try:
        # 1단계: 패널 데이터 조회
        print("📌 1단계: 패널 데이터 조회")
        panels_data = get_panels_data_from_db(panel_id_list)
        
        if not panels_data:
            return {
                "query": query,
                "total_count": 0,
                "main_summary": "패널 데이터를 조회할 수 없습니다.",
                "charts": []
            }, 200
        
        print(f"✅ {len(panels_data)}개 패널 데이터 조회 완료\n")
        
        # 2단계: ranked_keywords 추출
        print("📌 2단계: 키워드 우선순위 확인")
        ranked_keywords = classified_keywords.get('ranked_keywords', [])
        
        search_used_fields = set()
        for kw_info in ranked_keywords:
            field = kw_info.get('field', '')
            if field:
                search_used_fields.add(field)
        
        if not ranked_keywords:
            obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
            ranked_keywords = []
            for kw in obj_keywords[:5]:
                field = _guess_field_from_keyword(kw)
                korean_name = FIELD_NAME_MAP.get(field, field)
                ranked_keywords.append({
                    'keyword': kw,
                    'field': field,
                    'description': korean_name,
                    'priority': len(ranked_keywords) + 1
                })
                search_used_fields.add(field)
        
        if not ranked_keywords:
            return {
                "query": query,
                "total_count": len(panels_data),
                "main_summary": f"총 {len(panels_data)}명의 데이터를 조회했으나 분석할 키워드가 없습니다.",
                "charts": []
            }, 200
        
        ranked_keywords.sort(key=lambda x: x.get('priority', 999))
        
        print(f"✅ 키워드 목록: {[k.get('keyword') for k in ranked_keywords]}")
        print(f"✅ 검색 조건 필드: {list(search_used_fields)}\n")
        
        # 3단계: ranked_keywords 기반 차트 생성 (DB 집계 사용!)
        print("📌 3단계: 주요 키워드 차트 생성 (DB 집계, 최적화)")
        charts = []
        used_fields = []
        
        objective_fields = set([f[0] for f in WELCOME_OBJECTIVE_FIELDS])
        
        chart_count = 0
        for kw_info in ranked_keywords:
            if chart_count >= 2:
                break
            
            keyword = kw_info.get('keyword', '')
            field = kw_info.get('field', '')
            korean_name = kw_info.get('description', FIELD_NAME_MAP.get(field, field))
            
            if not field or not keyword:
                continue
            
            if field in EXCLUDED_RAW_FIELDS:
                print(f"   ⏭️  '{keyword}' (필드: {field}) - raw 필드, 스킵")
                continue
            
            if field not in objective_fields:
                print(f"   ⏭️  '{keyword}' (필드: {field}) - 객관식 아님, 스킵")
                continue
            
            if field in used_fields:
                print(f"   ⏭️  '{keyword}' (필드: {field}) - 이미 사용됨, 스킵")
                continue
            
            # ✅ DB 집계 쿼리 사용 (최적화!)
            chart = create_chart_data_optimized(
                keyword, field, korean_name, panels_data, use_full_db=True
            )
            
            if not chart.get('chart_data') or chart.get('ratio') == '0.0%':
                print(f"   ⏭️  '{keyword}' (필드: {field}) - 데이터 없음, 스킵")
                continue
            
            charts.append(chart)
            used_fields.append(field)
            chart_count += 1
            print(f"   ✅ [{chart_count}] '{keyword}' → {korean_name} 차트 생성 (DB 집계)")
        
        print()
        
        # 4단계: 높은 비율 필드 찾기 (병렬 처리!)
        print("📌 4단계: 높은 비율 필드 차트 생성 (병렬 처리)")
        needed_charts = 5 - len(charts)
        
        exclude_fields_for_step4 = list(set(used_fields) | search_used_fields)
        print(f"   🚫 제외할 필드: {exclude_fields_for_step4}")
        
        if needed_charts > 0:
            high_ratio_fields = find_high_ratio_fields_optimized(
                panels_data, 
                exclude_fields=exclude_fields_for_step4,
                threshold=50.0,
                max_charts=needed_charts
            )
            
            if not high_ratio_fields:
                print(f"   ⚠️  50% 이상 비율을 가진 필드를 찾지 못했습니다.")
            
            for field_info in high_ratio_fields:
                if len(charts) >= 5:
                    break
                
                distribution = field_info['distribution']
                top_category = field_info['top_category']
                top_ratio = field_info['top_ratio']
                
                chart = {
                    "topic": f"{field_info['korean_name']} 분포",
                    "description": f"{top_ratio:.1f}%가 '{top_category}'로 뚜렷한 패턴을 보입니다.",
                    "ratio": f"{top_ratio:.1f}%",
                    "chart_data": [{
                        "label": field_info['korean_name'],
                        "values": distribution
                    }]
                }
                charts.append(chart)
                print(f"   ✅ [{len(charts)}] {field_info['korean_name']} ({top_ratio:.1f}%) 차트 생성")
        
        print()
        
        # 5단계: 요약 생성
        print("📌 5단계: 요약 생성")
        main_summary = f"총 {len(panels_data)}명의 응답자 데이터를 분석했습니다. "
        
        if charts:
            top_chart = charts[0]
            main_summary += f"주요 분석 결과: {top_chart.get('topic', '')}에서 {top_chart.get('ratio', '0%')}의 비율을 보입니다."
        
        if len(charts) > 2:
            main_summary += f" 추가로 {len(charts) - 2}개의 뚜렷한 패턴이 발견되었습니다."
        
        result = {
            "query": query,
            "total_count": len(panels_data),
            "main_summary": main_summary,
            "charts": charts
        }
        
        print(f"✅ 분석 완료: {len(charts)}개 차트 생성 (최적화)")
        print(f"{'='*70}\n")
        
        return result, 200
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return {
            "query": query,
            "total_count": 0,
            "main_summary": f"분석 중 오류 발생: {str(e)}",
            "charts": []
        }, 500


def _guess_field_from_keyword(keyword: str) -> str:
    """키워드로부터 필드명 추정"""
    kw = keyword.strip().lower()
    
    if kw in ['남자', '남성', '남', '여자', '여성', '여']:
        return 'gender'
    elif '대' in keyword and keyword[:-1].isdigit():
        return 'birth_year'
    elif keyword in ['서울', '경기', '부산', '대구', '인천', '광주', '대전', '울산', '세종', 
                     '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']:
        return 'region_major'
    elif keyword.endswith(('시', '구', '군')):
        return 'region_minor'
    elif kw in ['미혼', '싱글', '기혼', '결혼', '이혼', '돌싱']:
        return 'marital_status'
    elif kw in ['고소득', '저소득']:
        return 'income_personal_monthly'
    else:
        return 'gender'