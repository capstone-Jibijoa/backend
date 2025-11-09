"""
검색 결과 분석 및 차트 데이터 생성
- hybrid_logic.py에서 받은 ranked_keywords 사용 (LLM 호출 없음)
- 상위 2개 키워드 + 높은 비율 필드 = 최대 5개 차트 생성
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
    get_panels_data_from_db,
    get_all_panels_data_from_db
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

        # ⚠️ 0% 비율 데이터 필터링 (find_top_category에 넘기기 전에 실행)
        # find_top_category도 필터링된 distribution을 받아야 합니다.
        filtered_distribution = {k: v for k, v in distribution.items() if v > 0.0}
        
        if not filtered_distribution:
            continue
        
        top_category, top_ratio = find_top_category(distribution)
        
        # threshold 이상만 추가
        if top_ratio >= threshold:
            high_ratio_results.append({
                "field": field_name,
                "korean_name": korean_name,
                "distribution": distribution,  # 전체 분포 (파이차트용)
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
    use_full_db: bool = False
) -> Dict:
    """
    특정 키워드/필드에 대한 차트 데이터 생성
    
    Args:
        keyword: 키워드
        field_name: 필드명
        korean_name: 한글명
        panels_data: 검색된 패널 데이터 (use_full_db=True면 사용 안 함)
        use_full_db: True면 전체 DB 데이터로 분석
    
    Returns:
        {
            "topic": "차트 제목",
            "description": "설명",
            "ratio": "XX.X%",
            "chart_data": [{"label": "...", "values": {...}}]
        }
    """
    # 전체 DB 기반 분석 옵션
    if use_full_db:
        print(f"      → 전체 DB 데이터로 '{field_name}' 분석")
        full_data = get_all_panels_data_from_db()
        if not full_data:
            return {
                "topic": korean_name,
                "description": f"'{keyword}' 관련 전체 데이터를 조회할 수 없습니다.",
                "ratio": "0.0%",
                "chart_data": []
            }
        analysis_data = full_data
        description_prefix = f"전체 데이터 기준 '{keyword}' 분석:"
    else:
        analysis_data = panels_data
        description_prefix = f"'{keyword}' 검색 결과:"
    
    # 필드 값 추출
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

    # 🌟 추가된 로직: 비율이 0.0%인 항목은 필터링
    # 부동 소수점 비교의 안전성을 위해 0.0보다 큰 값으로 필터링
    filtered_distribution = {k: v for k, v in distribution.items() if v > 0.0}
    
    if not filtered_distribution:
        # 필터링 후 데이터가 없으면 차트 생성 스킵
        return {
            "topic": korean_name,
            "description": f"'{keyword}' 관련 데이터가 부족합니다.",
            "ratio": "0.0%",
            "chart_data": []
        }
    
    top_category, top_ratio = find_top_category(distribution)
    
    # 차트 데이터 생성
    return {
        "topic": f"{korean_name} 분포",
        "description": f"{description_prefix} {top_ratio}%가 '{top_category}'입니다.",
        "ratio": f"{top_ratio}%",
        "chart_data": [{
            "label": korean_name,
            "values": distribution
        }]
    }


def analyze_search_results(
    query: str,
    classified_keywords: dict,
    pid_list: List[int]
) -> Tuple[Dict, int]:
    """
    검색 결과를 분석하여 최대 5개의 차트 데이터 생성
    
    프로세스:
    1. DB에서 패널 데이터 조회
    2. ranked_keywords에서 Welcome 객관식 필드만 차트 생성 (1, 2순위는 전체 DB 기반)
    3. 높은 비율(70% 이상) 필드 찾아서 나머지 차트 생성
    4. 최대 5개 차트 반환
    
    Args:
        query: 원본 자연어 질의
        classified_keywords: hybrid_logic에서 분류된 키워드 (ranked_keywords 포함)
        pid_list: 검색된 PID 리스트
        
    Returns:
        (analysis_result, status_code)
    """
    print(f"\n{'='*70}")
    print(f"📊 분석 시작")
    print(f"   질의: {query}")
    print(f"   PID 수: {len(pid_list)}개")
    print(f"{'='*70}\n")
    
    # 결과가 없는 경우
    if not pid_list:
        return {
            "query": query,
            "total_count": 0,
            "main_summary": "검색 결과가 없습니다.",
            "charts": []
        }, 200
    
    try:
        # 1단계: DB에서 패널 데이터 조회
        print("📌 1단계: 패널 데이터 조회")
        panels_data = get_panels_data_from_db(pid_list)
        
        if not panels_data:
            return {
                "query": query,
                "total_count": 0,
                "main_summary": "패널 데이터를 조회할 수 없습니다.",
                "charts": []
            }, 200
        
        print(f"✅ {len(panels_data)}개 패널 데이터 조회 완료\n")
        
        # 2단계: ranked_keywords 추출 (hybrid_logic에서 LLM이 이미 판단)
        print("📌 2단계: 키워드 우선순위 확인")
        ranked_keywords = classified_keywords.get('ranked_keywords', [])
        
        if not ranked_keywords:
            # ranked_keywords가 없으면 기본 방식 사용
            print("⚠️  ranked_keywords 없음, welcome_keywords 사용")
            obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
            
            # 기본 필드 매핑
            ranked_keywords = []
            for kw in obj_keywords[:5]:  # 최대 5개 시도
                field = _guess_field_from_keyword(kw)
                korean_name = FIELD_NAME_MAP.get(field, field)
                ranked_keywords.append({
                    'keyword': kw,
                    'field': field,
                    'description': korean_name,
                    'priority': len(ranked_keywords) + 1
                })
        
        if not ranked_keywords:
            return {
                "query": query,
                "total_count": len(panels_data),
                "main_summary": f"총 {len(panels_data)}명의 데이터를 조회했으나 분석할 키워드가 없습니다.",
                "charts": []
            }, 200
        
        # ranked_keywords를 priority 순으로 정렬
        ranked_keywords.sort(key=lambda x: x.get('priority', 999))
        
        print(f"✅ 키워드 목록: {[k.get('keyword') for k in ranked_keywords]}\n")
        
        # 3단계: Welcome 객관식 필드만 차트 생성
        print("📌 3단계: 주요 키워드 차트 생성 (Welcome 객관식만)")
        charts = []
        used_fields = []
        
        # Welcome 객관식 필드 목록
        objective_fields = set([f[0] for f in WELCOME_OBJECTIVE_FIELDS])
        
        chart_count = 0
        for kw_info in ranked_keywords:
            if chart_count >= 2:  # ✅ 1, 2순위만 전체 DB 기반으로 생성
                break
            
            keyword = kw_info.get('keyword', '')
            field = kw_info.get('field', '')
            korean_name = kw_info.get('description', FIELD_NAME_MAP.get(field, field))
            
            if not field or not keyword:
                continue
            
            # Welcome 객관식 필드가 아니면 스킵
            if field not in objective_fields:
                print(f"   ⏭️  '{keyword}' (필드: {field}) - 객관식 아님, 스킵")
                continue
            
            # 이미 사용한 필드는 스킵
            if field in used_fields:
                print(f"   ⏭️  '{keyword}' (필드: {field}) - 이미 사용됨, 스킵")
                continue
            
            # ✅ 전체 DB 기반으로 차트 생성 (use_full_db=True)
            chart = create_chart_data(keyword, field, korean_name, panels_data, use_full_db=True)
            
            # 차트 데이터가 없으면 스킵
            if not chart.get('chart_data') or chart.get('ratio') == '0.0%':
                print(f"   ⏭️  '{keyword}' (필드: {field}) - 데이터 없음, 스킵")
                continue
            
            charts.append(chart)
            used_fields.append(field)
            chart_count += 1
            print(f"   ✅ [{chart_count}] '{keyword}' → {korean_name} 차트 생성 (전체 DB 기반)")
        
        print()
        
        # 4단계: 높은 비율 필드 찾기 (최대 5개까지)
        print("📌 4단계: 높은 비율 필드 차트 생성 (검색 결과 기반)")
        needed_charts = 5 - len(charts)
        
        if needed_charts > 0:
            high_ratio_fields = find_high_ratio_fields(
                panels_data, 
                exclude_fields=used_fields,
                threshold=50.0,  # 50% 이상 비율
                max_charts=needed_charts
            )
            
            for field_info in high_ratio_fields:
                if len(charts) >= 5:  # 최대 5개
                    break
                
                # 검색 결과의 전체 분포 표시 (파이차트용)
                distribution = field_info['distribution']
                top_category = field_info['top_category']
                top_ratio = field_info['top_ratio']
                
                chart = {
                    "topic": f"{field_info['korean_name']} 분포",
                    "description": f"{top_ratio:.1f}%가 '{top_category}'로 뚜렷한 패턴을 보입니다.",
                    "ratio": f"{top_ratio:.1f}%",
                    "chart_data": [{
                        "label": field_info['korean_name'],
                        "values": distribution  # ✅ 검색 결과의 전체 분포 (파이차트용)
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


# ====================================================================
# 테스트 코드
# ====================================================================

if __name__ == "__main__":
    # 테스트 데이터
    test_query = "경기 30대 남자 술을 먹은 사람"
    
    # hybrid_logic.py에서 받은 것처럼 ranked_keywords 포함
    test_classified = {
        "ranked_keywords": [
            {"keyword": "술", "field": "drinking_experience", "description": "음주 경험", "priority": 1},
            {"keyword": "30대", "field": "birth_year", "description": "연령대", "priority": 2},
            {"keyword": "경기", "field": "region_minor", "description": "거주 지역", "priority": 3},
            {"keyword": "남자", "field": "gender", "description": "성별", "priority": 4}
        ],
        "welcome_keywords": {
            "objective": ["경기", "30대", "남자", "술먹는"],
            "subjective": []
        },
        "qpoll_keywords": {
            "survey_type": "lifestyle",
            "keywords": ["음주", "술"]
        }
    }
    
    # 🚨 수정된 부분: 테스트용 PID를 문자열 리스트로 변경
    # 실제 환경에서는 이 작업이 get_panels_data_from_db 내부에서 이루어져야 합니다.
    test_pids = ['1', '2', '3', '4', '5'] 
    
    print("="*70)
    print("🧪 analysis_logic.py 테스트")
    print("="*70)
    
    result, status = analyze_search_results(test_query, test_classified, test_pids)
    
    if status == 200:
        print("\n✅ 테스트 성공")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\n❌ 테스트 실패 (Status: {status})")
        print(result)