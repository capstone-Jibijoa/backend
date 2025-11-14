import json
import logging
import re 
from typing import List, Dict, Any, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (
    extract_field_values,
    calculate_distribution,
    find_top_category,
    FIELD_NAME_MAP,
    WELCOME_OBJECTIVE_FIELDS,
    get_panels_data_from_db 
)
from db import get_db_connection_context

# 1. 정적 매핑 규칙 (Python 코드로 관리)
FIELD_MAPPING_RULES = [
    # --- type: "filter" (객관식 필터용) ---
    (re.compile(r'^\d{2}대$'), 
     {"field": "birth_year", "description": "연령대", "type": "filter"}),
    (re.compile(r'^\d{2}~\d{2}대$'), 
     {"field": "birth_year", "description": "연령대", "type": "filter"}),
    (re.compile(r'젊은층|청년|MZ세대'), 
     {"field": "birth_year", "description": "연령대", "type": "filter"}),
    
    (re.compile(r'^(서울|경기|부산|인천|대구|광주|대전|울산|세종|강원|충북|충남|전북|전남|경북|경남|제주)(특별)?(자?치)?(시|도|광역)?$', re.IGNORECASE), 
     {"field": "region_major", "description": "거주 지역", "type": "filter"}),
    
    (re.compile(r'.*(시|구|군)$'), 
     {"field": "region_minor", "description": "세부 거주 지역", "type": "filter"}),

    (re.compile(r'^(남|남자|남성)$', re.IGNORECASE), 
     {"field": "gender", "description": "성별", "type": "filter"}),
    (re.compile(r'^(여|여자|여성)$', re.IGNORECASE), 
     {"field": "gender", "description": "성별", "type": "filter"}),
    
    ("미혼", {"field": "marital_status", "description": "결혼 여부", "type": "filter"}),
    ("기혼", {"field": "marital_status", "description": "결혼 여부", "type": "filter"}),
    
    ("흡연", {"field": "smoking_experience", "description": "흡연 경험", "type": "filter"}),
    ("비흡연", {"field": "smoking_experience", "description": "흡연 경험", "type": "filter"}),
    
    ("음주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("금주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    
    ("차량보유", {"field": "car_ownership", "description": "차량 보유", "type": "filter"}),
    ("차없음", {"field": "car_ownership", "description": "차량 보유", "type": "filter"}),
    # 개념/주관식 키워드는 모두 제거 -> 'unknown' 처리되어 벡터 검색으로 유도
]

def get_field_mapping(keyword: str) -> Dict[str, str]:
    """
    [수정됨] 키워드를 받아 매핑되는 필드 정보와 "타입"을 반환합니다.
    필터 규칙에 없으면 'unknown'으로 반환합니다.
    """
    keyword_for_string_match = keyword.lower() 
    
    for pattern, mapping_info in FIELD_MAPPING_RULES:
        
        # [수정] 'type'을 명시적으로 확인 (기본값 'filter')
        rule_type = mapping_info.get("type", "filter")
        if rule_type != "filter": # (혹시 모를 실수를 방지하기 위해)
            continue
            
        field = mapping_info["field"]
        description = FIELD_NAME_MAP.get(field, mapping_info["description"])

        if isinstance(pattern, re.Pattern):
            if pattern.match(keyword): 
                return {"field": field, 
                        "description": description, 
                        "type": "filter"}
        elif isinstance(pattern, str):
            if pattern == keyword_for_string_match:
                return {"field": field, 
                        "description": description, 
                        "type": "filter"}
            
    # '간호직', 'OTT', 'it' 등 필터 규칙에 없는 모든 키워드는 'unknown'으로 처리
    # (주의: search.py의 파서가 'unknown' 타입을 'vector'로 해석해야 함)
    logging.warning(f" ⚠️  '{keyword}'에 대한 매핑 규칙 없음. 'unknown'(벡터)으로 처리.")
    return {"field": "unknown", "description": keyword, "type": "unknown"}


def get_field_distribution_from_db(field_name: str, limit: int = 10) -> Dict[str, float]:
    """
    PostgreSQL에서 직접 집계하여 필드 분포 조회 (전체 DB 대상)
    (캐시 롤백 버전)
    """
    try:
        with get_db_connection_context() as conn:
            if not conn:
                logging.error("DB 집계: 연결 실패")
                return {}
            
            cur = conn.cursor()
            
            if field_name == "birth_year":
                query = f"""
                    WITH age_groups AS (
                        SELECT 
                            CASE 
                                WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 20 THEN '10대'
                                WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 30 THEN '20대'
                                WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 40 THEN '30대'
                                WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 50 THEN '40대'
                                WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 60 THEN '50대'
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
            
            distribution = {}
            for row in rows:
                value = row[0]
                percentage = float(row[2])
                if value and percentage > 0:
                    distribution[value] = percentage
            
            cur.close()
        
        logging.info(f"   📊 DB 집계 완료: {field_name} ({len(distribution)}개 카테고리)")
        return distribution
        
    except Exception as e:
        logging.error(f"   DB 집계 실패 ({field_name}): {e}", exc_info=True)
        return {}

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
    """
    # 전체 DB 기반 분석
    if use_full_db:
        logging.info(f"       → DB 집계로 '{field_name}' 분석 (최적화)")
        distribution = get_field_distribution_from_db(field_name, max_categories)
        
        if not distribution:
            return {
                "topic": korean_name,
                "description": f"'{keyword}' 관련 전체 데이터를 조회할 수 없습니다.",
                "ratio": "0.0%",
                "chart_data": []
            }
        
        top_category, top_ratio = find_top_category(distribution)
        description_prefix = f"전체 데이터 기준 '{korean_name}' 분석:"
        
        return {
            "topic": f"{korean_name} 분포",
            "description": f"{description_prefix} {top_ratio}%가 '{top_category}'입니다.",
            "ratio": f"{top_ratio}%",
            "chart_data": [{"label": korean_name, "values": distribution}]
        }
    
    # 검색 결과 기반 분석
    else:
        values = extract_field_values(panels_data, field_name)
        
        if not values:
            return { "topic": korean_name, "description": "데이터 부족", "ratio": "0.0%", "chart_data": [] }
        
        distribution = calculate_distribution(values)
        filtered_distribution = {k: v for k, v in distribution.items() if v > 0.0}
        
        if not filtered_distribution:
            return { "topic": korean_name, "description": "데이터 부족", "ratio": "0.0%", "chart_data": [] }
        
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

def create_crosstab_chart(
    panels_data: List[Dict],
    field1: str,  # 주축 (e.g., 'birth_year')
    field2: str,  # 세그먼트 (e.g., 'gender')
    field1_korean: str,
    field2_korean: str,
    max_categories: int = 5
) -> Dict:
    """
    교차 분석 차트 데이터를 생성합니다. (예: 연령대별 성별 분포)
    """
    logging.info(f"       → 교차 분석으로 '{field1}' vs '{field2}' 분석")
    from utils import get_age_group

    # 1. 두 필드에 대한 데이터 추출
    crosstab_data = {}
    for item in panels_data:
        val1 = item.get(field1)
        val2 = item.get(field2)

        if val1 is None or val2 is None:
            continue

        # 값 처리 (연령대 변환 등)
        key1 = get_age_group(val1) if field1 == 'birth_year' else str(val1)
        key2 = str(val2)

        if key1 not in crosstab_data:
            crosstab_data[key1] = []
        crosstab_data[key1].append(key2)

    if not crosstab_data:
        return {}

    # 2. 각 주축 카테고리별로 세그먼트 분포 계산
    chart_values = {}
    for key1, values2 in crosstab_data.items():
        distribution = calculate_distribution(values2)
        chart_values[key1] = distribution

    # 3. 주축 카테고리가 너무 많으면 상위 N개만 선택
    if len(chart_values) > max_categories:
        # 각 카테고리의 전체 개수를 기준으로 정렬
        sorted_keys = sorted(chart_values.keys(), key=lambda k: sum(crosstab_data[k].count(v) for v in set(crosstab_data[k])), reverse=True)
        chart_values = {k: chart_values[k] for k in sorted_keys[:max_categories]}

    if not chart_values:
        return {}

    return {
        "topic": f"{field1_korean}별 {field2_korean} 분포",
        "description": f"'{field1_korean}'에 따른 '{field2_korean}'의 상대적 분포를 보여줍니다.",
        "chart_type": "crosstab",
        "chart_data": [{"label": f"{field1_korean}별 {field2_korean}", "values": chart_values}]
    }


def _analyze_fields_in_parallel(panels_data: List[Dict], candidate_fields: List[Tuple[str, str]]) -> List[Dict]:
    """
    [리팩토링] panels_data를 한 번만 순회하여 모든 후보 필드의 값을 집계하고 분포를 계산합니다.
    """
    field_values = {field_name: [] for field_name, _ in candidate_fields}
    field_map = dict(candidate_fields)

    # 데이터를 한 번만 순회하여 모든 필드의 값을 추출
    for item in panels_data:
        for field_name in field_values.keys():
            value = item.get(field_name)
            if value is None:
                continue

            if field_name == "birth_year":
                from utils import get_age_group
                field_values[field_name].append(get_age_group(value))
            elif isinstance(value, list):
                field_values[field_name].extend(value)
            else:
                field_values[field_name].append(value)

    results = []
    for field_name, values in field_values.items():
        if not values:
            continue
        
        try:
            distribution = calculate_distribution(values)
            filtered_distribution = {k: v for k, v in distribution.items() if v > 0.0}
            if not filtered_distribution:
                continue

            results.append({
                "field": field_name,
                "korean_name": field_map[field_name],
                "distribution": filtered_distribution,
            })
        except Exception as e:
            logging.warning(f"   ⚠️  {field_name} 분포 계산 실패: {e}")

    return results


def find_high_ratio_fields_optimized(
    panels_data: List[Dict], 
    exclude_fields: List[str], 
    threshold: float = 50.0,
    max_charts: int = 3
) -> List[Dict]:
    """
    검색 결과(panels_data) 내에서 높은 비율을 차지하는 필드 찾기 (병렬 처리)
    """
    candidate_fields = []
    
    for field_name, korean_name in WELCOME_OBJECTIVE_FIELDS:
        if field_name not in exclude_fields:
            candidate_fields.append((field_name, korean_name))
    
    if not candidate_fields:
        return []
    
    logging.info(f"   🔍 {len(candidate_fields)}개 필드 병렬 분석 중... (제외 필드: {exclude_fields})")
    
    analysis_results = _analyze_fields_in_parallel(panels_data, candidate_fields)
    
    high_ratio_results = []
    for result in analysis_results:
        distribution = result['distribution']
        
        # 100% 단일 카테고리 스킵
        if len(distribution) == 1:
            top_category, top_ratio = find_top_category(distribution) 
            logging.info(f"   ⚠️  [{result['korean_name']}] 스킵: {top_category} {top_ratio}% (단일 카테고리 100%)")
            continue
        
        top_category, top_ratio = find_top_category(distribution)
        
        # 50% 이상이라는 1차 임계값 통과 시
        if top_ratio >= threshold:
        
            final_distribution = distribution
            if len(distribution) > 10:
                sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
                top_items = dict(sorted_items[:9])
                other_sum = sum(v for k, v in sorted_items[9:])
                if other_sum > 0:
                    top_items['기타'] = round(other_sum, 1)
                final_distribution = top_items
            
            high_ratio_results.append({
                "field": result['field'],
                "korean_name": result['korean_name'],
                "distribution": final_distribution,
                "top_category": top_category,
                "top_ratio": top_ratio
            })
    
    high_ratio_results.sort(key=lambda x: x["top_ratio"], reverse=True)
    
    return high_ratio_results[:max_charts]

def analyze_search_results_optimized(
    query: str,
    classified_keywords: dict,
    panel_id_list: List[str]
) -> Tuple[Dict, int]:
    """
    검색 결과 분석 (최적화 버전)
    """
    logging.info(f"📊 분석 시작 (최적화) - panel_id 수: {len(panel_id_list)}개")
    
    if not panel_id_list:
        return {"main_summary": "검색 결과가 없습니다.", "charts": []}, 200
    
    try:
        # 1단계: 패널 데이터 조회
        logging.info("   1단계: 패널 데이터 조회 (utils.py 사용)")
        panels_data = get_panels_data_from_db(panel_id_list)
        
        if not panels_data:
            return {"main_summary": "패널 데이터를 조회할 수 없습니다.", "charts": []}, 200
        
        logging.info(f"   ✅ {len(panels_data)}개 패널 데이터 조회 완료")
        
        # 2단계: ranked_keywords 추출 및 매핑
        # (주의: search.py에서 LLM 대신 규칙 기반 파서로 대체되었다면
        #       classified_keywords['ranked_keywords_raw']가 올바르게 전달되어야 함)
        raw_keywords = classified_keywords.get('ranked_keywords_raw', [])
        ranked_keywords = []
        search_used_fields = set()
        
        if raw_keywords:
            logging.info(f"   2a단계: (규칙 기반) 키워드 {raw_keywords} 매핑 시작")
            for i, keyword in enumerate(raw_keywords):
                
                mapping = get_field_mapping(keyword) 
                kw_type = mapping.get("type", "unknown")
                
                ranked_keywords.append({
                    "keyword": keyword, 
                    "field": mapping["field"],
                    "description": mapping["description"], 
                    "type": kw_type, # [수정] type 정보 저장
                    "priority": i + 1
                })
                
                # [수정] 'filter' 타입이고 'unknown'이 아닌 경우에만 '뻔한 필드'로 추가
                if kw_type == 'filter' and mapping["field"] != 'unknown':
                    search_used_fields.add(mapping["field"])
        
        if not ranked_keywords:
            logging.warning("   ⚠️  'ranked_keywords_raw' 없음. (Fallback 로직 실행)")
            # (Fallback 로직은 LLM을 사용하지 않으므로 수정이 필요할 수 있음)
            obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
            for i, kw in enumerate(obj_keywords[:5]):
                mapping = get_field_mapping(kw) # [수정] 수정된 함수 호출
                ranked_keywords.append({
                    'keyword': kw, 
                    'field': mapping["field"],
                    'description': mapping["description"], 
                    'type': mapping.get("type", "unknown"), # [수정] type 정보 저장
                    'priority': i + 1
                })
                if mapping["type"] == 'filter' and mapping["field"] != 'unknown': # [수정] type 체크
                    search_used_fields.add(mapping["field"])
            
        if not ranked_keywords:
            return { "main_summary": f"총 {len(panels_data)}명 조회, 분석할 키워드 없음.", "charts": [] }, 200
        
        ranked_keywords.sort(key=lambda x: x.get('priority', 999))
        logging.info(f"   ✅ 분석 키워드: {[k.get('keyword') for k in ranked_keywords]}")
        logging.info(f"   ✅ 검색 사용 필드 (뻔한 인사이트 제외용): {search_used_fields}")
        
        # 3단계: ranked_keywords 기반 차트 생성
        logging.info("   3단계: 주요 키워드 차트 생성 (DB 집계, 병렬)")
        charts = []
        used_fields = []
        objective_fields = set([f[0] for f in WELCOME_OBJECTIVE_FIELDS])
        
        # 1. 생성할 차트 작업 목록 정의
        chart_tasks = [] 
        chart_count = 0
        for kw_info in ranked_keywords:
            if chart_count >= 2: break
            
            field = kw_info.get('field', '')
            kw_type = kw_info.get('type', 'unknown') # [수정] type 가져오기
            
            # [수정] 'filter' 타입인 키워드만 3단계 차트 생성
            if kw_type != 'filter' or not field or field == 'unknown' or field not in objective_fields or field in used_fields:
                # ('it', '간호직' 등 벡터/unknown 키워드는 여기서 차트 생성 안 함)
                continue
            
            chart_tasks.append(kw_info)
            used_fields.append(field) 
            chart_count += 1

        # 2. ThreadPoolExecutor로 차트 생성 병렬 실행
        if chart_tasks:
            with ThreadPoolExecutor(max_workers=len(chart_tasks) or 1) as executor:
                
                def create_chart_task(kw_info):
                    field = kw_info.get('field', '')
                    logging.info(f"   ⚡ [{field}] 차트 DB 집계 스레드 시작...")
                    return create_chart_data_optimized(
                        kw_info.get('keyword', ''), 
                        field, 
                        kw_info.get('description', FIELD_NAME_MAP.get(field, field)),
                        panels_data, 
                        use_full_db=True
                    )

                futures = {executor.submit(create_chart_task, kw_info): kw_info for kw_info in chart_tasks}
                
                for future in as_completed(futures):
                    kw_info_original = futures[future] 
                    field_name = kw_info_original.get('field', 'unknown')
                    try:
                        chart = future.result() 
                        if chart.get('chart_data') and chart.get('ratio') != '0.0%':
                            chart['priority'] = kw_info_original.get('priority', 99)
                            charts.append(chart)
                            logging.info(f"   ✅ [{field_name}] 차트 생성 완료 (DB 집계)")
                        else:
                            logging.warning(f"   ⚠️  [{field_name}] 차트 데이터가 비어있음 (DB 집계)")
                    except Exception as e:
                        logging.error(f"   ❌ [{field_name}] 차트 생성 실패: {e}", exc_info=True)
            
            charts.sort(key=lambda x: x.get('priority', 99))
            
            for chart in charts:
                if 'priority' in chart:
                    del chart['priority']

        # 3.5단계: 교차 분석 차트 생성
        logging.info("   3.5단계: 교차 분석 차트 생성")
        if len(charts) < 5 and len(ranked_keywords) > 0:
            
            CROSSTAB_CANDIDATES = [
                ('gender', '성별'),
                ('birth_year', '연령대'),
                ('marital_status', '결혼 여부'),
                ('income_personal_monthly', '소득 수준'),
                ('job_duty_raw', '직무'), 
                ('job_title_raw', '직업'),
            ]
            
            primary_kw = None
            for kw in ranked_keywords:
                if kw.get("type") == "filter":
                    primary_kw = kw
                    break 
            
            primary_field = None
            primary_korean_name = None
            
            if primary_kw:
                primary_field = primary_kw.get('field')
                primary_korean_name = primary_kw.get('description')
            
            secondary_field = None
            secondary_korean_name = None
            
            if primary_field and primary_field != 'unknown' and primary_field in objective_fields:
                for field, korean in CROSSTAB_CANDIDATES:
                    if field == primary_field or field in search_used_fields:
                        continue
                    
                    secondary_field = field
                    secondary_korean_name = korean
                    logging.info(f"   ✨ 새 교차분석 축 발견: '{primary_korean_name}' vs '{secondary_korean_name}'")
                    break
            
            if secondary_field:
                crosstab_chart = create_crosstab_chart(
                    panels_data,
                    primary_field, secondary_field,
                    primary_korean_name, secondary_korean_name
                )
                if crosstab_chart:
                    charts.append(crosstab_chart)
                    if primary_field not in used_fields:
                         used_fields.append(primary_field) 
                    # [수정] 교차분석 보조축도 제외 목록에 추가
                    if secondary_field not in used_fields:
                        used_fields.append(secondary_field)
                    logging.info(f"   ✅ [{len(charts)}] 교차 분석 차트 생성 ({primary_korean_name} vs {secondary_korean_name})")
            else:
                logging.warning("   ⚠️  교차 분석 스킵: 1순위 필터 키워드가 없거나, 적절한 보조축 후보가 없음 (모두 검색어에 포함됨)")
                
        # 4단계: 높은 비율 필드 찾기
        logging.info("   4단계: 높은 비율 필드 차트 생성 (검색 결과 기반)")
        needed_charts = 5 - len(charts)
        # 3.5단계에서 보조축까지 제외됨
        exclude_fields_for_step4 = list(set(used_fields) | search_used_fields)
        
        if needed_charts > 0:
            high_ratio_fields = find_high_ratio_fields_optimized(
                panels_data, 
                # [롤백] 검색에 사용된 필드와 이미 차트로 만들어진 필드를 제외
                exclude_fields=exclude_fields_for_step4,
                threshold=50.0,
                max_charts=needed_charts
            )
            
            for field_info in high_ratio_fields:
                if len(charts) >= 5: break
                
                chart = {
                    "topic": f"{field_info['korean_name']} 분포",
                    "description": f"{field_info['top_ratio']:.1f}%가 '{field_info['top_category']}'로 뚜렷한 패턴을 보입니다.",
                    "ratio": f"{field_info['top_ratio']:.1f}%",
                    "chart_data": [{"label": field_info['korean_name'], "values": field_info['distribution']}]
                }
                charts.append(chart)
                logging.info(f"   ✅ [{len(charts)}] {field_info['korean_name']} ({field_info['top_ratio']:.1f}%) 차트 생성")
        
        # 5단계: 요약 생성
        main_summary = f"총 {len(panels_data)}명의 응답자 데이터를 분석했습니다. "
        if charts:
            top_chart = charts[0]
            summary_desc = top_chart.get('description', '')
            if '전체 데이터 기준' in summary_desc:
                summary_desc = summary_desc.split(':', 1)[-1].strip()
            elif ':' in summary_desc:
                 summary_desc = summary_desc.split(':', 1)[-1].strip()
            
            main_summary += f"주요 분석 결과: {top_chart.get('topic', '')}에서 {top_chart.get('ratio', '0%')}의 비율을 보입니다."
        
        result = {
            "query": query,
            "total_count": len(panels_data),
            "main_summary": main_summary,
            "charts": charts
        }
        
        logging.info(f"✅ 분석 완료: {len(charts)}개 차트 생성 (최적화)")
        return result, 200
        
    except Exception as e:
        logging.error(f"❌ 분석 실패: {e}", exc_info=True)
        return {"main_summary": f"분석 중 오류 발생: {str(e)}", "charts": []}, 500