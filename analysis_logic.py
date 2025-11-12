"""
검색 결과 분석 및 차트 데이터 생성
- hybrid_logic.py에서 받은 ranked_keywords 사용 (LLM 호출 없음)
- 상위 2개 키워드 + 높은 비율 필드 = 최대 5개 차트 생성
- 3단계: ranked_keywords 기반 (전체 DB, 검색 조건 포함 OK)
- 4단계: 높은 비율 필드 (검색 결과, 검색 조건 제외)
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
    get_panels_data_from_db,
    get_all_panels_data_from_db,
    get_db_distribution
)


def find_high_ratio_fields(
    panels_data: List[Dict], 
    exclude_fields: List[str], 
    threshold: float = 50.0,
    max_charts: int = 3
) -> List[Dict]:
    """
    검색된 PID들에서 특정 카테고리가 높은 비율을 차지하는 필드 찾기
    
    Args:
        panels_data: 검색된 패널 데이터 리스트
        exclude_fields: 이미 차트로 만든 필드 (제외)
        threshold: 높은 비율 판단 기준 (기본 50%)
        max_charts: 최대 차트 개수
        
    Returns:
        [{field, korean_name, distribution, top_category, top_ratio}, ...]
    """
    high_ratio_results = []
    
    for field_name, korean_name in WELCOME_OBJECTIVE_FIELDS:
        # ✅ raw 필드 제외
        if field_name in EXCLUDED_RAW_FIELDS:
            continue
        
        # 제외 필드는 스킵
        if field_name in exclude_fields:
            continue
        
        # 분포 계산
        values = extract_field_values(panels_data, field_name)
        if not values:
            continue
        
        distribution = calculate_distribution(values)
        if not distribution:
            continue

        # ✅ 0% 비율 데이터 필터링
        filtered_distribution = {k: v for k, v in distribution.items() if v > 0.0}
        
        if not filtered_distribution:
            continue
        
        top_category, top_ratio = find_top_category(filtered_distribution)
        
        # threshold 이상만 추가
        if top_ratio >= threshold:
            # ✅ 카테고리가 너무 많으면 상위 10개만 + 기타
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
                "distribution": filtered_distribution,  # ✅ 필터링 및 상위 10개
                "top_category": top_category,
                "top_ratio": top_ratio
            })
    
    # 비율 높은 순 정렬
    high_ratio_results.sort(key=lambda x: x["top_ratio"], reverse=True)
    
    return high_ratio_results[:max_charts]


def create_chart_data(
    keyword: str,
    field_name: str,
    korean_name: str,
    panels_data: List[Dict],
    use_full_db: bool = False,
    max_categories: int = 10
) -> Dict:
    """
    특정 키워드/필드에 대한 차트 데이터 생성 (버그 수정됨)
    """
    
    final_distribution = {}
    description_prefix = ""

    if use_full_db:
        print(f"DB에서 직접 '{field_name}' 집계")
        final_distribution = get_db_distribution(field_name) 
        
        description_prefix = f"전체 데이터 기준 '{keyword}' 분석:"
        
        if not final_distribution:
            return {
                "topic": korean_name,
                "description": f"'{keyword}' 관련 전체 데이터를 DB에서 집계할 수 없습니다.",
                "ratio": "0.0%",
                "chart_data": []
            }

    else:
        analysis_data = panels_data
        description_prefix = f"'{keyword}' 검색 결과:"
        
        # 필드값 추출
        values = extract_field_values(analysis_data, field_name)
        
        if not values:
            return {
                "topic": korean_name,
                "description": f"'{keyword}' 관련 데이터가 부족합니다.",
                "ratio": "0.0%",
                "chart_data": []
            }
        
        # 분포 계산
        distribution = calculate_distribution(values)

        # 0% 비율 항목 필터링
        filtered_distribution = {k: v for k, v in distribution.items() if v > 0.0}
        
        if not filtered_distribution:
            return {
                "topic": korean_name,
                "description": f"'{keyword}' 관련 데이터가 부족합니다.",
                "ratio": "0.0%",
                "chart_data": []
            }
     
        if len(filtered_distribution) > max_categories:
            sorted_items = sorted(filtered_distribution.items(), key=lambda x: x[1], reverse=True)
            top_items = dict(sorted_items[:max_categories - 1])
            other_sum = sum(v for k, v in sorted_items[max_categories - 1:])
            if other_sum > 0:
                top_items['기타'] = round(other_sum, 1)
            final_distribution = top_items
            print(f"      → {len(filtered_distribution)}개 카테고리 중 상위 {max_categories}개만 표시")
        else:
            final_distribution = filtered_distribution 
    
    top_category, top_ratio = find_top_category(final_distribution)
    
    # 차트 데이터 생성
    return {
        "topic": f"{korean_name} 분포",
        "description": f"{description_prefix} {top_ratio}%가 '{top_category}'입니다.",
        "ratio": f"{top_ratio}%",
        "chart_data": [{
            "label": korean_name,
            "values": final_distribution
        }]
    }

def analyze_search_results(
    query: str,
    classified_keywords: dict,
    panel_id_list: List[str]
) -> Tuple[Dict, int]:
    """
    검색 결과를 분석하여 최대 5개의 차트 데이터 생성
    
    프로세스:
    1. DB에서 패널 데이터 조회
    2. ranked_keywords에서 Welcome 객관식 필드만 차트 생성 (1, 2순위, 전체 DB 기반)
       → 검색 조건 필드도 포함 (전체 DB 기준이므로 의미 있음)
    3. 높은 비율(50% 이상) 필드 찾아서 나머지 차트 생성 (검색 결과 기반)
       → 검색 조건 필드는 제외 (당연한 결과 제외)
    4. 최대 5개 차트 반환
    
    Args:
        query: 원본 자연어 질의
        classified_keywords: hybrid_logic에서 분류된 키워드 (ranked_keywords 포함)
        panel_id_list: 검색된 panel_id 리스트 (문자열)
        
    Returns:
        (analysis_result, status_code)
    """
    print(f"\n{'='*70}")
    print(f"📊 분석 시작")
    print(f"   질의: {query}")
    print(f"   panel_id 수: {len(panel_id_list)}개")
    print(f"{'='*70}\n")
    
    # 결과가 없는 경우
    if not panel_id_list:
        return {
            "query": query,
            "total_count": 0,
            "main_summary": "검색 결과가 없습니다.",
            "charts": []
        }, 200
    
    try:
        # 1단계: DB에서 패널 데이터 조회
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
        
        # ✅ 검색 조건 필드 추출 (4단계에서만 사용)
        search_used_fields = set()
        for kw_info in ranked_keywords:
            field = kw_info.get('field', '')
            if field:
                search_used_fields.add(field)
        
        if not ranked_keywords:
            # ranked_keywords가 없으면 기본 방식 사용
            print("⚠️  ranked_keywords 없음, welcome_keywords 사용")
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
        
        # ranked_keywords를 priority 순으로 정렬
        ranked_keywords.sort(key=lambda x: x.get('priority', 999))
        
        print(f"✅ 키워드 목록: {[k.get('keyword') for k in ranked_keywords]}")
        print(f"✅ 검색 조건 필드: {list(search_used_fields)} (4단계에서만 제외)\n")
        
        # 3단계: ranked_keywords 기반 차트 생성 (검색 조건 포함 OK!)
        print("📌 3단계: 주요 키워드 차트 생성 (전체 DB 기준, 검색 조건 포함)")
        charts = []
        used_fields = []
        
        # Welcome 객관식 필드 목록
        objective_fields = set([f[0] for f in WELCOME_OBJECTIVE_FIELDS])
        
        chart_count = 0
        for kw_info in ranked_keywords:
            if chart_count >= 2:  # 1, 2순위만
                break
            
            keyword = kw_info.get('keyword', '')
            field = kw_info.get('field', '')
            korean_name = kw_info.get('description', FIELD_NAME_MAP.get(field, field))
            
            if not field or not keyword:
                continue
            
            # ✅ raw 필드 제외
            if field in EXCLUDED_RAW_FIELDS:
                print(f"   ⏭️  '{keyword}' (필드: {field}) - raw 필드, 스킵")
                continue
            
            # Welcome 객관식 필드가 아니면 스킵
            if field not in objective_fields:
                print(f"   ⏭️  '{keyword}' (필드: {field}) - 객관식 아님, 스킵")
                continue
            
            # 이미 사용한 필드는 스킵
            if field in used_fields:
                print(f"   ⏭️  '{keyword}' (필드: {field}) - 이미 사용됨, 스킵")
                continue
            
            # ✅ 전체 DB 기반으로 차트 생성
            chart = create_chart_data(keyword, field, korean_name, panels_data, use_full_db=True)
            
            if not chart.get('chart_data') or chart.get('ratio') == '0.0%':
                print(f"   ⏭️  '{keyword}' (필드: {field}) - 데이터 없음, 스킵")
                continue
            
            charts.append(chart)
            used_fields.append(field)
            chart_count += 1
            print(f"   ✅ [{chart_count}] '{keyword}' → {korean_name} 차트 생성 (전체 DB 기준)")
        
        print()
        
        # 4단계: 높은 비율 필드 찾기 (검색 조건 제외!)
        print("📌 4단계: 높은 비율 필드 차트 생성 (검색 결과 기준, 검색 조건 제외)")
        needed_charts = 5 - len(charts)
        
        # ✅ 4단계에서만 검색 조건 필드 제외
        exclude_fields_for_step4 = list(set(used_fields) | search_used_fields)
        print(f"   🚫 제외할 필드: {exclude_fields_for_step4}")
        
        if needed_charts > 0:
            high_ratio_fields = find_high_ratio_fields(
                panels_data, 
                exclude_fields=exclude_fields_for_step4,  # ✅ 검색 조건 + 이미 사용한 필드
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
        
        # 최종 결과
        result = {
            "query": query,
            "total_count": len(panels_data),
            "main_summary": main_summary,
            "charts": charts
        }
        
        print(f"✅ 분석 완료: {len(charts)}개 차트 생성")
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
    """
    키워드로부터 필드명 추정 (fallback용)
    """
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
    elif kw in ['술먹는', '음주', '술', '술안먹는', '금주']:
        return 'drinking_experience'
    elif kw in ['흡연', '담배', '비흡연', '금연']:
        return 'smoking_experience'
    elif kw in ['차있음', '자가용', '차량보유', '차없음']:
        return 'car_ownership'
    else:
        return 'gender'  # 기본값