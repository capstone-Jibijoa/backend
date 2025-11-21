import os
import logging
import re 
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import numpy as np
from sklearn.cluster import DBSCAN
from search_helpers import initialize_embeddings 

from utils import (
    extract_field_values,
    calculate_distribution,
    find_top_category,
    FIELD_NAME_MAP,
    WELCOME_OBJECTIVE_FIELDS,
    get_panels_data_from_db,
    get_age_group
)
# [필수] 템플릿 import
from mapping_rules import get_field_mapping, QPOLL_FIELD_TO_TEXT, QPOLL_ANSWER_TEMPLATES
from db import get_db_connection_context, get_qdrant_client


def _clean_label(text: Any, max_length: int = 12) -> str:
    """기본 라벨 정제 함수"""
    if not text: return ""
    text_str = str(text)
    cleaned = re.sub(r'\([^)]*\)', '', text_str).strip()
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_length:
        return cleaned[:max_length] + ".."
    return cleaned

def _extract_core_value(field_name: str, sentence: str) -> str:
    """
    [지능형 수정] 템플릿을 역설계하여 핵심 답변만 자동 추출합니다.
    """
    if not sentence: return ""
    
    # 1. 숫자 데이터 등 특수 처리가 필요한 필드는 기존 로직 유지 (정확도 최우선)
    if field_name == "ott_count":
        match = re.search(r'(\d+개|이용 안 함|없음)', sentence)
        if match: return match.group(1)
    elif field_name == "skincare_spending":
        match = re.search(r'(\d+만\s*원|\d+~\d+만\s*원|\d+원)', sentence)
        if match: return match.group(1)

    # 2. QPOLL_ANSWER_TEMPLATES를 이용한 동적 추출
    template = QPOLL_ANSWER_TEMPLATES.get(field_name)
    if template:
        try:
            # 2-1. 템플릿을 정규식 패턴으로 변환하기 위해 특수문자 이스케이프
            # 예: "이사할 때 {answer_str}(으)로..." -> "이사할\ 때\ \{answer_str\}\(으\)로\.\.\."
            pattern_str = re.escape(template)

            # 2-2. {answer_str} 부분을 캡처 그룹 (.*?) 으로 변경
            # re.escape로 인해 \{answer_str\} 형태가 되었을 것임
            pattern_str = pattern_str.replace(re.escape("{answer_str}"), r"(.*?)")

            # 2-3. 한국어 조사 유연성 처리
            # 템플릿의 (이)다 -> (?:이)?다 ( '이'는 있어도 되고 없어도 됨)
            pattern_str = pattern_str.replace(r"\(이\)다", r"(?:이)?다")
            pattern_str = pattern_str.replace(r"\(으\)로", r"(?:으)?로")
            pattern_str = pattern_str.replace(r"\(가\)", r"(?:가)?")
            
            # 2-4. 공백 유연성 (템플릿과 실제 데이터의 띄어쓰기가 다를 수 있음)
            pattern_str = pattern_str.replace(r"\ ", r"\s*")

            # 2-5. 매칭 시도
            match = re.search(pattern_str, sentence)
            if match:
                # 캡처된 내용(핵심 답변) 반환
                extracted = match.group(1)
                return _clean_label(extracted)
        except Exception as e:
            logging.warning(f"템플릿 추출 실패 ({field_name}): {e}")

    # 3. 템플릿 매칭 실패 시 기본 정제 반환
    return _clean_label(sentence)


def _limit_distribution_top_k(distribution: Dict[str, float], k: int = 7) -> Dict[str, float]:
    """상위 K개 + 기타로 제한"""
    if not distribution or len(distribution) <= k:
        return distribution
    
    sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    top_items = dict(sorted_items[:k])
    
    other_sum = sum(v for _, v in sorted_items[k:])
    if other_sum > 0:
        top_items['기타'] = round(other_sum, 1)
            
    return top_items


def get_field_distribution_from_db(field_name: str, limit: int = 10) -> Dict[str, float]:
    """PostgreSQL 직접 집계"""
    try:
        with get_db_connection_context() as conn:
            if not conn: return {}
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
                    )
                    SELECT age_group, COUNT(*), ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)
                    FROM age_groups GROUP BY age_group ORDER BY 3 DESC LIMIT {limit}
                """
            else:
                query = f"""
                    SELECT structured_data->>'{field_name}', COUNT(*), ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)
                    FROM welcome_meta2
                    WHERE structured_data->>'{field_name}' IS NOT NULL
                    GROUP BY 1 ORDER BY 3 DESC LIMIT {limit}
                """
            
            cur.execute(query)
            rows = cur.fetchall()
            cur.close()
            return {row[0]: float(row[2]) for row in rows if row[0]}
            
    except Exception as e:
        logging.error(f"DB 집계 실패 ({field_name}): {e}")
        return {}
    
def get_qpoll_distribution_from_db(qpoll_field: str, limit: int = 10) -> Dict[str, float]:
    """Qdrant 집계 (자동 추출 적용)"""
    question_text = QPOLL_FIELD_TO_TEXT.get(qpoll_field)
    if not question_text: return {}
    
    client = get_qdrant_client()
    if not client: return {}
        
    try:
        COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_QPOLL_NAME", "qpoll_vectors_v2")
        query_filter = Filter(must=[FieldCondition(key="question", match=MatchValue(value=question_text))])
        
        all_points = []
        next_offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME, scroll_filter=query_filter, limit=1000, offset=next_offset, with_payload=True, with_vectors=False
            )
            all_points.extend(points)
            if next_offset is None: break
        
        if not all_points: return {}

        extracted_values = []
        for p in all_points:
            if p.payload and p.payload.get("sentence"):
                raw_sentence = p.payload.get("sentence")
                # [핵심] 템플릿 기반 자동 추출 실행
                core_val = _extract_core_value(qpoll_field, raw_sentence)
                if core_val:
                    extracted_values.append(core_val)
        
        total_count = len(extracted_values)
        if total_count == 0: return {}

        val_counts = Counter(extracted_values)
        
        return {
            k: round((v / total_count) * 100, 1)
            for k, v in val_counts.most_common(limit)
        }
        
    except Exception as e:
        logging.error(f"Q-Poll 집계 실패: {e}")
        return {}

def create_chart_data_optimized(
    keyword: str,
    field_name: str,
    korean_name: str,
    panels_data: List[Dict],
    use_full_db: bool = False,
    max_categories: int = 7
) -> Dict:
    """차트 데이터 생성"""
    if use_full_db:
        logging.info(f"       → DB 집계로 '{field_name}' 분석 (최적화)")
        distribution = get_field_distribution_from_db(field_name, max_categories)
        
        if not distribution:
            return {"topic": korean_name, "ratio": "0.0%", "chart_data": [], "description": "데이터 없음"}
        
        cleaned_distribution = defaultdict(float)
        for k, v in distribution.items():
            cleaned_distribution[_clean_label(k)] += v
            
        final_distribution = _limit_distribution_top_k(dict(cleaned_distribution), max_categories)
        top_category, top_ratio = find_top_category(final_distribution)
        
        return {
            "topic": f"{korean_name} 분포",
            "description": f"전체 기준: {top_ratio}%가 '{top_category}'입니다.",
            "ratio": f"{top_ratio}%",
            "chart_data": [{"label": korean_name, "values": final_distribution}],
            "field": field_name 
        }
    
    else:
        values = []
        raw_values = [item.get(field_name) for item in panels_data if item.get(field_name)]
        
        for val in raw_values:
            if isinstance(val, list):
                for v in val:
                    cleaned = _clean_label(v)
                    if cleaned: values.append(cleaned)
            elif val is not None:
                cleaned = _clean_label(val)
                if cleaned: values.append(cleaned)
        
        if not values:
            return { "topic": korean_name, "description": "데이터 부족", "ratio": "0.0%", "chart_data": [], "field": field_name }
        
        distribution = calculate_distribution(values)
        final_distribution = _limit_distribution_top_k(distribution, max_categories)
        top_category, top_ratio = find_top_category(final_distribution)
        
        return {
            "topic": f"{korean_name} 분포",
            "description": f"검색 결과: {top_ratio}%가 '{top_category}'입니다.",
            "ratio": f"{top_ratio}%",
            "chart_data": [{"label": korean_name, "values": final_distribution}],
            "field": field_name
        }

def create_qpoll_chart_data(qpoll_field: str, max_categories: int = 7) -> Dict:
    """Q-Poll 차트 데이터 생성"""
    question_text = QPOLL_FIELD_TO_TEXT.get(qpoll_field, qpoll_field) 
    
    distribution = get_qpoll_distribution_from_db(qpoll_field, max_categories)
    
    if not distribution:
        return {"topic": question_text, "ratio": "0.0%", "chart_data": [], "description": "데이터 없음", "field": qpoll_field}
    
    top_category, top_ratio = find_top_category(distribution)
    
    # 설명(Description)은 템플릿을 사용해 자연스럽게
    template = QPOLL_ANSWER_TEMPLATES.get(qpoll_field)
    if template and top_category != "기타":
        try:
            formatted_answer = template.format(answer_str=f"'{top_category}'")
            description = f"가장 많은 응답자는 {formatted_answer} ({top_ratio}%)"
        except:
            description = f"가장 많은 응답은 '{top_category}'({top_ratio}%)입니다."
    else:
        description = f"가장 많은 응답은 '{top_category}'({top_ratio}%)입니다."

    return {
        "topic": question_text, 
        "description": description,
        "ratio": f"{top_ratio}%",
        "chart_data": [{"label": question_text, "values": distribution}],
        "field": qpoll_field
    }

def create_crosstab_chart(
    panels_data: List[Dict],
    field1: str,
    field2: str,
    field1_korean: str,
    field2_korean: str,
    max_categories: int = 5
) -> Dict:
    """교차 분석 차트 생성"""
    logging.info(f"       → 교차 분석: '{field1}' vs '{field2}'")
    
    all_values_field2 = []
    for item in panels_data:
        val2 = item.get(field2)
        if not val2: continue
        
        if isinstance(val2, list):
            for v in val2:
                cleaned = _clean_label(v)
                if cleaned: all_values_field2.append(cleaned)
        else:
            cleaned = _clean_label(val2)
            if cleaned: all_values_field2.append(cleaned)
            
    if not all_values_field2:
        return {}

    global_counter = Counter(all_values_field2)
    top_7_keys = [k for k, v in global_counter.most_common(7)]
    top_7_set = set(top_7_keys)

    crosstab_data = {} 

    for item in panels_data:
        val1 = item.get(field1)
        val2 = item.get(field2)
        
        if val1 is None or val2 is None: continue

        raw_key1 = get_age_group(val1) if field1 == 'birth_year' else str(val1)
        key1 = _clean_label(raw_key1)
        
        if key1 not in crosstab_data:
            crosstab_data[key1] = []
            
        values_to_process = val2 if isinstance(val2, list) else [val2]
        
        for v in values_to_process:
            cleaned_v = _clean_label(v)
            if not cleaned_v: continue
            
            if cleaned_v in top_7_set:
                crosstab_data[key1].append(cleaned_v)
            else:
                crosstab_data[key1].append("기타")

    if not crosstab_data:
        return {}

    if len(crosstab_data) <= 1:
        only_group = list(crosstab_data.keys())[0]
        distribution = calculate_distribution(crosstab_data[only_group])
        final_distribution = _limit_distribution_top_k(distribution, k=7)
        
        return {
            "topic": f"{field1_korean}별 {field2_korean} 분포 ({only_group})",
            "description": f"'{only_group}' 집단의 '{field2_korean}' 분포입니다.",
            "chart_type": "pie", 
            "chart_data": [{"label": field2_korean, "values": final_distribution}],
            "fields": [field1, field2]
        }

    chart_values = {}
    sorted_groups = sorted(crosstab_data.keys(), key=lambda k: len(crosstab_data[k]), reverse=True)
    target_groups = sorted_groups[:max_categories]

    for group in target_groups:
        items = crosstab_data[group]
        distribution = calculate_distribution(items)
        chart_values[group] = _limit_distribution_top_k(distribution, k=7)

    return {
        "topic": f"{field1_korean}별 {field2_korean} 분포",
        "description": f"'{field1_korean}'에 따른 주요 '{field2_korean}' 분포입니다.",
        "chart_type": "crosstab",
        "chart_data": [{"label": f"{field1_korean}별 {field2_korean}", "values": chart_values}],
        "fields": [field1, field2] 
    }

def _analyze_fields_in_parallel(panels_data: List[Dict], candidate_fields: List[Tuple[str, str]]) -> List[Dict]:
    """병렬 필드 분석"""
    field_values = {fname: [] for fname, _ in candidate_fields}
    field_map = dict(candidate_fields)

    for item in panels_data:
        for fname in field_values.keys():
            val = item.get(fname)
            if val is None: continue

            if fname == "birth_year":
                field_values[fname].append(get_age_group(val))
            elif isinstance(val, list):
                for v in val:
                    cleaned = _clean_label(v)
                    if cleaned: field_values[fname].append(cleaned)
            else:
                cleaned = _clean_label(val)
                if cleaned: field_values[fname].append(cleaned)

    results = []
    for fname, vals in field_values.items():
        if not vals: continue
        try:
            dist = calculate_distribution(vals)
            final_dist = _limit_distribution_top_k(dist, k=7)
            if not final_dist: continue
            
            results.append({
                "field": fname,
                "korean_name": field_map[fname],
                "distribution": final_dist,
            })
        except: pass
    return results


def find_high_ratio_fields_optimized(
    panels_data: List[Dict], 
    exclude_fields: List[str], 
    threshold: float = 50.0,
    max_charts: int = 3
) -> List[Dict]:
    """높은 비율 필드 찾기"""
    candidate_fields = []
    for fname, kname in WELCOME_OBJECTIVE_FIELDS:
        if fname not in exclude_fields:
            candidate_fields.append((fname, kname))
    
    if not candidate_fields: return []
    
    analysis_results = _analyze_fields_in_parallel(panels_data, candidate_fields)
    
    high_ratio_results = []
    for result in analysis_results:
        distribution = result['distribution']
        top_category, top_ratio = find_top_category(distribution)
        
        if top_ratio >= threshold:
            if top_ratio >= 98.0:
                continue

            high_ratio_results.append({
                "field": result['field'],
                "korean_name": result['korean_name'],
                "distribution": distribution,
                "top_category": top_category,
                "top_ratio": top_ratio
            })
    
    high_ratio_results.sort(key=lambda x: x["top_ratio"], reverse=True)
    return high_ratio_results[:max_charts]

def _generate_no_results_tips(classified_keywords: dict) -> str:
    tips = []
    if len(classified_keywords.get('objective_keywords', [])) > 3:
        tips.append("필터 조건이 너무 많을 수 있습니다.")
    tips.append("더 일반적인 키워드를 사용해 보세요.")
    return "\n".join(tips)

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

        # [기존 로직] 검색에 사용된 필드 추출
        demographic_filters = classified_keywords.get('demographic_filters', {})
        if demographic_filters:
            logging.debug(f"   ✅ demographic_filters에서 검색 필터 추출: {demographic_filters}")
            if 'age_range' in demographic_filters:
                search_used_fields.add('birth_year')  
            for key in demographic_filters.keys():
                if key != 'age_range':
                    search_used_fields.add(key)

        structured_filters = classified_keywords.get('structured_filters', [])
        if structured_filters:
            for filter_item in structured_filters:
                field = filter_item.get('field')
                if field and field != 'age':
                    search_used_fields.add(field)
                elif field == 'age':
                    search_used_fields.add('birth_year')

        if raw_keywords:
            logging.debug(f"   2a단계: (규칙 기반) 키워드 {raw_keywords} 매핑 시작")
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
            # Fallback 로직 (기존 코드 유지)
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
                if mapping["type"] == 'filter' and mapping["field"] != 'unknown':
                    search_used_fields.add(mapping["field"])
            
        ranked_keywords.sort(key=lambda x: x.get('priority', 999))
        
        logging.info("   3단계: 주요 키워드 차트 생성 (DB 집계, 병렬)")
        charts = []
        used_fields = []
        objective_fields = set([f[0] for f in WELCOME_OBJECTIVE_FIELDS])

        is_single_household = False
        
        # 1. demographic_filters 확인
        fam_val = demographic_filters.get('family_size') or demographic_filters.get('household_size')
        if fam_val:
            if isinstance(fam_val, list):
                if any(str(v).startswith('1') for v in fam_val): is_single_household = True
            elif str(fam_val).startswith('1'):
                is_single_household = True
                
        # 2. structured_filters 확인
        if not is_single_household:
            for f in structured_filters:
                if f.get('field') in ['family_size', 'household_size']:
                    val = f.get('value')
                    if isinstance(val, list):
                        if any(str(v).startswith('1') for v in val): is_single_household = True
                    elif str(val).startswith('1'):
                        is_single_household = True
                        
        if is_single_household:
            logging.info("   ℹ️ 1인 가구 감지: '가구 소득', '혼인 여부' 필드를 인사이트에서 제외합니다.")
            # used_fields에 미리 추가해두면, 이후 단계(4단계)에서 중복 필드로 간주되어 생성되지 않음
            used_fields.append('income_household_monthly')
            used_fields.append('marital_status')
        # ----------------------------------------------------------------------

        chart_tasks = []
        for kw_info in ranked_keywords:
            if len(chart_tasks) >= 3: break 

            field = kw_info.get('field', '')
            kw_type = kw_info.get('type', 'unknown')

            if field in used_fields:
                continue

            if kw_type == 'qpoll': 
                chart_tasks.append({"type": "qpoll", "kw_info": kw_info})
                used_fields.append(field)
            elif kw_type == 'filter' and field not in search_used_fields:
                if field in objective_fields and field != 'unknown':
                    if panels_data:
                        chart_tasks.append({"type": "filter", "kw_info": kw_info})
                        used_fields.append(field)
                        
        if chart_tasks:
            with ThreadPoolExecutor(max_workers=len(chart_tasks) or 1) as executor:
                
                def run_chart_creation(task):
                    kw_info = task["kw_info"]
                    field = kw_info.get('field', '')
                    korean_name = kw_info.get('description', field)
                    
                    if task["type"] == "filter":
                        return create_chart_data_optimized(
                            kw_info.get('keyword', ''), field, korean_name, panels_data, use_full_db=False
                        )
                    elif task["type"] == "qpoll":
                        return create_qpoll_chart_data(field)
                    return None

                futures = {executor.submit(run_chart_creation, task): task for task in chart_tasks}
                
                for future in as_completed(futures):
                    kw_info_original = futures[future]["kw_info"]
                    try:
                        chart = future.result() 
                        if chart.get('chart_data') and chart.get('ratio') != '0.0%':
                            chart['priority'] = kw_info_original.get('priority', 99)
                            charts.append(chart)
                    except Exception as e:
                        logging.error(f"차트 생성 실패: {e}", exc_info=True)
            
            charts.sort(key=lambda x: x.get('priority', 99))
            for chart in charts:
                if 'priority' in chart: del chart['priority']

        logging.debug("   3.2단계: 연관 필드 심층 분석")
        needed_charts_after_main = 5 - len(charts)
        if needed_charts_after_main > 0:
            if 'region_major' in search_used_fields and 'region_minor' not in used_fields:
                region_minor_chart = create_chart_data_optimized(
                    "세부 지역", "region_minor", "세부 지역(구/군)", panels_data, use_full_db=False, max_categories=15 
                )
                if region_minor_chart and region_minor_chart.get('chart_data'):
                    charts.append(region_minor_chart)
                    used_fields.append('region_minor')

            if 'birth_year' in search_used_fields and 'marital_status' not in used_fields:
                marital_chart = create_chart_data_optimized(
                    "혼인 상태", "marital_status", "혼인 상태", panels_data, use_full_db=False, max_categories=10
                )
                if marital_chart and marital_chart.get('chart_data') and len(charts) < 5:
                    charts.append(marital_chart)
                    used_fields.append('marital_status')

        logging.debug("   3.5단계: 교차 분석 차트 생성")
        if len(charts) < 5:
            topic_field_info = next((kw for kw in ranked_keywords if kw.get('type') == 'qpoll'), None)
            if not topic_field_info:
                topic_field_info = next((kw for kw in ranked_keywords if kw.get('type') == 'filter'), None)

            if topic_field_info:
                topic_field = topic_field_info.get('field')
                topic_korean_name = topic_field_info.get('description')

                priority_axes = []
                for field in search_used_fields:
                    if field != topic_field:
                        korean_name = FIELD_NAME_MAP.get(field, field)
                        priority_axes.append((field, korean_name, 'priority'))

                standard_axes = [
                    ('birth_year', '연령대', 'standard'),
                    ('gender', '성별', 'standard'),
                    ('region_major', '지역', 'standard'),
                    ('job_title_raw', '직업', 'standard'),
                    ('marital_status', '혼인 상태', 'standard'),
                ]

                all_axes = priority_axes + [ax for ax in standard_axes if ax[0] not in search_used_fields]

                for axis_field, axis_korean_name, axis_type in all_axes:
                    if len(charts) >= 5: break
                    if topic_field == axis_field: continue

                    crosstab_chart = create_crosstab_chart(
                        panels_data, axis_field, topic_field, axis_korean_name, topic_korean_name
                    )

                    if crosstab_chart and crosstab_chart.get("chart_data"):
                        charts.append(crosstab_chart)
                        used_fields.append(axis_field)
                        used_fields.append(topic_field)
                
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
                    "description": f"{field_info['top_ratio']:.1f}%가 '{field_info['top_category']}'입니다.",
                    "ratio": f"{field_info['top_ratio']:.1f}%",
                    "chart_data": [{"label": field_info['korean_name'], "values": field_info['distribution']}]
                }
                charts.append(chart)
        
        main_summary = f"총 {len(panels_data)}명의 응답자 데이터를 분석했습니다. "
        if charts:
            top_chart = charts[0]
            summary_desc = top_chart.get('description', '')
            if ':' in summary_desc: summary_desc = summary_desc.split(':', 1)[-1].strip()
            main_summary += f"주요 분석 결과: {top_chart.get('topic', '')}에서 {top_chart.get('ratio', '0%')}의 비율을 보입니다."
        
        result = {
            "query": query,
            "total_count": len(panels_data),
            "main_summary": main_summary,
            "charts": charts
        }
        
        logging.info(f"✅ 분석 완료: {len(charts)}개 차트 생성")
        return result, 200
        
    except Exception as e:
        logging.error(f"❌ 분석 실패: {e}", exc_info=True)
        return {"main_summary": f"분석 중 오류 발생: {str(e)}", "charts": []}, 500

async def generate_dynamic_insight(panel_ids: List[str], target_field: str, field_desc: str) -> Dict:
    if not panel_ids or not target_field: return {}
    logging.info(f"📊 동적 인사이트 생성 중... (Field: {target_field})")
    panels_data = get_panels_data_from_db(panel_ids)
    
    cleaned_answers = []
    for p in panels_data:
        val = p.get(target_field)
        if val:
            if isinstance(val, list):
                for v in val:
                    cleaned = _clean_label(v)
                    if cleaned: cleaned_answers.append(cleaned)
            else:
                cleaned = _clean_label(val)
                if cleaned: cleaned_answers.append(cleaned)
    
    if not cleaned_answers: return {"error": "데이터 부족"}

    unique_answers = list(set(cleaned_answers))
    chart_data = {}
    
    if len(unique_answers) <= 15:
        chart_data = calculate_distribution(cleaned_answers)
    else:
        chart_data = _group_answers_with_vectors(cleaned_answers, threshold=0.82)

    final_chart_data = _limit_distribution_top_k(chart_data, k=7)
    if not final_chart_data: return {}

    top_category, top_ratio = find_top_category(final_chart_data)
    
    return {
        "topic": f"{field_desc} 분석",
        "description": f"'{field_desc}'에 대해 '{top_category}'({top_ratio}%) 응답이 가장 많았습니다.",
        "ratio": f"{top_ratio}%",
        "chart_data": [{"label": field_desc, "values": final_chart_data}]
    }

def _group_answers_with_vectors(answers: List[str], threshold: float = 0.75) -> Dict[str, float]:
    if not answers: return {}
    embeddings_model = initialize_embeddings()
    unique_answers = list(set(answers))
    if len(unique_answers) < 2: return calculate_distribution(answers)

    try:
        vectors = embeddings_model.embed_documents(unique_answers)
        vectors = np.array(vectors)
        clustering = DBSCAN(eps=1-threshold, min_samples=1, metric='cosine').fit(vectors)
        labels = clustering.labels_
        
        cluster_map = {}
        for i, label in enumerate(labels):
            if label not in cluster_map: cluster_map[label] = []
            cluster_map[label].append(unique_answers[i])
            
        total_counts = Counter(answers)
        cluster_to_repr = {}
        for label, group_members in cluster_map.items():
            repr_word = max(group_members, key=lambda x: (total_counts[x], -len(x)))
            cluster_to_repr[label] = repr_word
            
        ans_to_label = {ans: labels[i] for i, ans in enumerate(unique_answers)}
        mapped_answers = [cluster_to_repr[ans_to_label[ans]] for ans in answers]
            
        return calculate_distribution(mapped_answers)
    except Exception as e:
        logging.error(f"벡터 클러스터링 실패: {e}", exc_info=True)
        return calculate_distribution(answers)
        logging.error(f"벡터 클러스터링 실패: {e}", exc_info=True)
        return calculate_distribution(answers)