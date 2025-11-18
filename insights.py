import os
import logging
import re 
from typing import List, Dict, Any, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client.models import Filter, FieldCondition, MatchValue

from utils import (
    extract_field_values,
    calculate_distribution,
    find_top_category,
    FIELD_NAME_MAP,
    WELCOME_OBJECTIVE_FIELDS,
    get_panels_data_from_db 
)
from mapping_rules import get_field_mapping, QPOLL_FIELD_TO_TEXT # get_field_mapping import
from db import get_db_connection_context, get_qdrant_client
from functools import lru_cache


@lru_cache(maxsize=64)
def get_field_distribution_from_db(field_name: str, limit: int = 10) -> Dict[str, float]:
    """
    PostgreSQL에서 직접 집계하여 필드 분포를 조회합니다. (전체 DB 대상)
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
    
@lru_cache(maxsize=64)
def get_qpoll_distribution_from_db(qpoll_field: str, limit: int = 10) -> Dict[str, float]:
    """
    Qdrant에서 Q-Poll 질문에 대한 응답 분포를 조회합니다.
    """
    question_text = QPOLL_FIELD_TO_TEXT.get(qpoll_field)
    if not question_text:
        logging.error(f"Q-Poll DB 집계: '{qpoll_field}'에 해당하는 질문 원문을 찾을 수 없습니다.")
        return {}
    
    client = get_qdrant_client()
    if not client:
        logging.error("Q-Poll Qdrant 집계: Qdrant 클라이언트 연결 실패.")
        return {}
        
    try:
        COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_QPOLL_NAME", "qpoll_vectors_v2")
        
        query_filter = Filter(
            must=[
                FieldCondition(key="question", match=MatchValue(value=question_text))
            ]
        )
        
        all_points = []
        next_offset = None
        
        while True:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=query_filter,
                limit=1000, 
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if next_offset is None:
                break
                
        total_count = len(all_points)
        
        if total_count == 0:
            return {}

        sentence_counts = Counter(p.payload.get("sentence") for p in all_points if p.payload and p.payload.get("sentence"))
        
        distribution = {
            sentence: round((count / total_count) * 100, 1)
            for sentence, count in sentence_counts.most_common(limit)
        }
        
        logging.info(f"   📊 Q-Poll Qdrant 집계 완료: {qpoll_field} ({len(distribution)}개 카테고리)")
        return distribution
        
    except Exception as e:
        logging.error(f"   Q-Poll Qdrant 집계 실패: {e}", exc_info=True)
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
    차트 데이터를 생성합니다.
    """
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
    
    else:
        values = extract_field_values(panels_data, field_name)
        
        if not values:
            return { "topic": korean_name, "description": "데이터 부족", "ratio": "0.0%", "chart_data": [] }
        
        distribution = calculate_distribution(values)
        filtered_distribution = {k: v for k, v in distribution.items() if v > 0.0}
        
        if not filtered_distribution:
            return { "topic": korean_name, "description": "데이터 부족", "ratio": "0.0%", "chart_data": [] }
        
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

def create_qpoll_chart_data(
    qpoll_field: str,
    max_categories: int = 10
) -> Dict:
    """
    Q-Poll 데이터 기반으로 차트 데이터를 생성합니다.
    """
    question_text = QPOLL_FIELD_TO_TEXT.get(qpoll_field, qpoll_field) 
    logging.info(f"       → Q-Poll Qdrant 집계로 '{qpoll_field}' 분석")
    
    distribution = get_qpoll_distribution_from_db(qpoll_field, max_categories)
    
    if not distribution:
        return {
            "topic": question_text,
            "description": f"'{question_text}' 관련 Q-Poll 데이터를 조회할 수 없습니다.",
            "ratio": "0.0%",
            "chart_data": []
        }
    
    final_distribution = distribution
    
    top_category, top_ratio = find_top_category(final_distribution)
    
    is_array_type = "모두 선택해주세요" in question_text

    if is_array_type:
        description = f"Q-Poll 응답자 기준, 가장 많은 응답은 '{top_category}'로 {top_ratio}%입니다. (복수 응답 가능)"
    else:
        description = f"Q-Poll 응답자 기준, {top_ratio}%가 '{top_category}'입니다."
        
    return {
        "topic": question_text, 
        "description": description,
        "ratio": f"{top_ratio}%",
        "chart_data": [{
            "label": question_text,
            "values": final_distribution
        }]
    }

def create_crosstab_chart(
    panels_data: List[Dict],
    field1: str,
    field2: str,
    field1_korean: str,
    field2_korean: str,
    max_categories: int = 5
) -> Dict:
    """
    교차 분석 차트 데이터를 생성합니다. (예: 연령대별 성별 분포)
    """
    logging.info(f"       → 교차 분석으로 '{field1}' vs '{field2}' 분석")
    from utils import get_age_group

    crosstab_data = {}
    for item in panels_data:
        val1 = item.get(field1)
        val2 = item.get(field2)

        if val1 is None or val2 is None:
            continue

        key1 = get_age_group(val1) if field1 == 'birth_year' else str(val1)
        key2 = str(val2)

        if key1 not in crosstab_data:
            crosstab_data[key1] = []
        crosstab_data[key1].append(key2)

    if not crosstab_data:
        return {}

    chart_values = {}
    for key1, values2 in crosstab_data.items():
        distribution = calculate_distribution(values2)
        chart_values[key1] = distribution

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
    panels_data를 한 번만 순회하여 모든 후보 필드의 값을 집계하고 분포를 계산합니다.
    """
    field_values = {field_name: [] for field_name, _ in candidate_fields}
    field_map = dict(candidate_fields)

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
    검색 결과(panels_data) 내에서 높은 비율을 차지하는 필드를 찾습니다.
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
        
        top_category, top_ratio = find_top_category(distribution)
        
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

def _generate_no_results_tips(classified_keywords: dict) -> str:
    """
    검색 결과가 없을 때 사용자에게 보여줄 동적 도움말을 생성합니다.
    """
    tips = []
    objective_kws = classified_keywords.get('objective_keywords', [])
    mandatory_kws = classified_keywords.get('mandatory_keywords', [])
    vector_kws = classified_keywords.get('vector_keywords', [])
    
    total_filter_kws = len(objective_kws) + len(mandatory_kws)

    if total_filter_kws > 3:
        combined_kws = objective_kws + mandatory_kws
        kws_str = ', '.join(f"'{k}'" for k in combined_kws[:3])
        tips.append(f"필터 조건({kws_str} 등)이 너무 많을 수 있습니다. 조건을 줄여보세요.")

    if vector_kws:
        tips.append(f"'{', '.join(vector_kws)}'와 같은 키워드가 너무 구체적일 수 있습니다. 더 일반적인 단어로 바꿔보세요.")

    tips.append("검색어에 오타가 없는지 확인하거나, '젊은층' 대신 '20대'처럼 더 명확한 키워드를 사용해 보세요.")
    
    summary = "검색 결과가 없습니다. 더 나은 결과를 위해 다음 팁을 확인해 보세요:\n"
    for i, tip in enumerate(tips, 1):
        summary += f"\n{i}. {tip}"
        
    return summary

def analyze_search_results_optimized(
    query: str,
    classified_keywords: dict,
    panel_id_list: List[str]
) -> Tuple[Dict, int]:
    """
    검색 결과를 분석하여 차트와 요약을 생성합니다.
    """
    logging.info(f"📊 분석 시작 (최적화) - panel_id 수: {len(panel_id_list)}개")
    
    if not panel_id_list:
        summary = _generate_no_results_tips(classified_keywords)
        return {"main_summary": summary, "charts": []}, 200
    
    try:
        logging.info("   1단계: 패널 데이터 조회 (utils.py 사용)")
        panels_data = get_panels_data_from_db(panel_id_list)
        
        if not panels_data:
            return {"main_summary": "패널 데이터를 조회할 수 없습니다.", "charts": []}, 200
        
        logging.info(f"   ✅ {len(panels_data)}개 패널 데이터 조회 완료")
        
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
                    "type": kw_type,
                    "priority": i + 1
                })
                
                if kw_type == 'filter' and mapping["field"] != 'unknown':
                    search_used_fields.add(mapping["field"])
        
        if not ranked_keywords:
            logging.warning("   ⚠️  'ranked_keywords_raw' 없음. (Fallback 로직 실행)")
            obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
            for i, kw in enumerate(obj_keywords[:5]):
                mapping = get_field_mapping(kw)
                ranked_keywords.append({
                    'keyword': kw, 
                    'field': mapping["field"],
                    'description': mapping["description"], 
                    'type': mapping.get("type", "unknown"),
                    'priority': i + 1
                })
                if mapping["type"] == 'filter' and mapping["field"] != 'unknown': # [수정] type 체크
                    search_used_fields.add(mapping["field"])
            
        if not ranked_keywords:
            return { "main_summary": f"총 {len(panels_data)}명 조회, 분석할 키워드 없음.", "charts": [] }, 200
        
        ranked_keywords.sort(key=lambda x: x.get('priority', 999))
        logging.info(f"   ✅ 분석 키워드: {[k.get('keyword') for k in ranked_keywords]}")
        logging.info(f"   ✅ 검색 사용 필드 (뻔한 인사이트 제외용): {search_used_fields}")
        
        logging.info("   3단계: 주요 키워드 차트 생성 (DB 집계, 병렬)")
        charts = []
        used_fields = []
        objective_fields = set([f[0] for f in WELCOME_OBJECTIVE_FIELDS])
        
        chart_tasks = [] 
        chart_count = 0
        for kw_info in ranked_keywords:
            if chart_count >= 2: break
            
            field = kw_info.get('field', '')
            kw_type = kw_info.get('type', 'unknown')
            
            if field in used_fields:
                continue

            if kw_type == 'filter':
                if field in objective_fields and field != 'unknown':
                    if panels_data:
                        chart_tasks.append({"type": "filter", "kw_info": kw_info})
                        used_fields.append(field)
                        chart_count += 1
            
            elif kw_type == 'qpoll': # Q-Poll 분석은 전체 DB 대상
                chart_tasks.append({"type": "qpoll", "kw_info": kw_info})
                used_fields.append(field)
                chart_count += 1
        if chart_tasks:
            with ThreadPoolExecutor(max_workers=len(chart_tasks) or 1) as executor:
                
                def run_chart_creation(task):
                    kw_info = task["kw_info"]
                    field = kw_info.get('field', '')
                    korean_name = kw_info.get('description', field)
                    logging.info(f"   ⚡ [{korean_name}] 차트 DB 집계 스레드 시작 ({task['type']})...")
                    
                    if task["type"] == "filter":
                        return create_chart_data_optimized(
                            kw_info.get('keyword', ''), 
                            field, 
                            korean_name,
                            panels_data,
                            use_full_db=False
                        )
                    elif task["type"] == "qpoll":
                        return create_qpoll_chart_data( 
                            field
                        )
                    
                    return None

                futures = {executor.submit(run_chart_creation, task): task for task in chart_tasks}
                
                for future in as_completed(futures):
                    kw_info_original = futures[future]["kw_info"]
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

        logging.info("   3.5단계: 교차 분석 차트 생성")
        if len(charts) < 5 and len(ranked_keywords) > 0:
            
            DEFAULT_CROSSTAB_AXES = [
                ('birth_year', '연령대'),
                ('gender', '성별'),
            ]

            if not search_used_fields:
                topic_kw = ranked_keywords[0]
                topic_field = topic_kw.get('field')
                topic_korean_name = topic_kw.get('description')

                if topic_field and topic_field != 'unknown':
                    # 기본 축(연령대, 성별)으로 교차 분석 실행
                    for axis_field, axis_korean_name in DEFAULT_CROSSTAB_AXES:
                        if len(charts) >= 5: break
                        
                        if topic_field == axis_field: continue

                        crosstab_chart = create_crosstab_chart(
                            panels_data, axis_field, topic_field, axis_korean_name, topic_korean_name)
                        if crosstab_chart and crosstab_chart.get("chart_data"):
                            charts.append(crosstab_chart)
                            logging.info(f"   ✅ [{len(charts)}] 교차 분석 차트 생성 ({axis_korean_name} vs {topic_korean_name})")

            else:
                CROSSTAB_CANDIDATES = [
                    ('gender', '성별'), ('birth_year', '연령대'), ('marital_status', '결혼 여부'),
                    ('income_personal_monthly', '소득 수준'), ('job_duty_raw', '직무'), ('job_title_raw', '직업'),
                ]

                primary_kw = next((kw for kw in ranked_keywords if kw.get("type") == "filter"), None)

                if primary_kw:
                    primary_field = primary_kw.get('field')
                    primary_korean_name = primary_kw.get('description')

                    secondary_field_info = next((
                        (field, korean) for field, korean in CROSSTAB_CANDIDATES 
                        if field != primary_field and field not in search_used_fields
                    ), None)

                    if secondary_field_info:
                        secondary_field, secondary_korean_name = secondary_field_info
                        if primary_field in search_used_fields and secondary_field in search_used_fields:
                            logging.warning(f"   ⚠️  교차 분석 스킵: 주축({primary_korean_name})과 보조축({secondary_korean_name})이 모두 검색 조건에 포함됨")
                            pass

                        logging.info(f"   ✨ 새 교차분석 축 발견: '{primary_korean_name}' vs '{secondary_korean_name}'")
                        
                        crosstab_chart = create_crosstab_chart(
                            panels_data,
                            primary_field, secondary_field,
                            primary_korean_name, secondary_korean_name
                        )
                        if crosstab_chart and crosstab_chart.get("chart_data"):
                            charts.append(crosstab_chart)
                            if primary_field not in used_fields: used_fields.append(primary_field)
                            if secondary_field not in used_fields: used_fields.append(secondary_field)
                            logging.info(f"   ✅ [{len(charts)}] 교차 분석 차트 생성 ({primary_korean_name} vs {secondary_korean_name})")
                    else:
                        logging.warning("   ⚠️  교차 분석 스킵: 적절한 보조축 후보가 없음 (모두 검색어에 포함됨)")
                else:
                    logging.warning("   ⚠️  교차 분석 스킵: 1순위 필터 키워드가 없음")
                
        logging.info("   4단계: 높은 비율 필드 차트 생성 (검색 결과 기반)")
        needed_charts = 5 - len(charts)
        exclude_fields_for_step4 = list(set(used_fields) | search_used_fields)
        
        if needed_charts > 0:
            high_ratio_fields = find_high_ratio_fields_optimized(
                panels_data, 
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