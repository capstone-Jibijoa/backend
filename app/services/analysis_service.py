import logging
import pandas as pd
import numpy as np
import asyncio
from typing import List, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from app.repositories.panel_repo import PanelRepository
from app.repositories.qpoll_repo import QpollRepository
from app.services.llm_service import LLMService
from app.core.semantic_router import router  
from app.utils.common import (
    find_target_columns_dynamic, 
    clean_label, 
    calculate_distribution, 
    get_age_group, 
    filter_merged_panels
)
from app.constants.mapping import (
    FIELD_NAME_MAP, 
    QPOLL_FIELD_TO_TEXT, 
    WELCOME_OBJECTIVE_FIELDS,
    VALUE_TRANSLATION_MAP,
    QPOLL_FIELDS,
)

class AnalysisService:
    def __init__(self):
        self.panel_repo = PanelRepository()
        self.qpoll_repo = QpollRepository()
        self.llm_service = LLMService()

    async def get_insight_summary(self, panel_ids: List[str], question: str) -> Dict[str, Any]:
        """[Lite 모드] 요약"""
        target_ids = panel_ids[:1000]
        panels_data = self.panel_repo.get_panels_by_ids(target_ids)
        
        if not panels_data:
            return {"summary": "분석할 데이터가 없습니다.", "used_fields": []}

        df = pd.DataFrame(panels_data)
        target_columns = find_target_columns_dynamic(question)
        
        if not target_columns:
            stats_context = self._calculate_column_stats(df, ['gender', 'birth_year', 'region_major'])
            target_columns = ['기본 인구통계']
        else:
            stats_context = self._calculate_column_stats(df, target_columns)

        summary_text = await self.llm_service.generate_insight_summary(question, stats_context)

        return {
            "summary": summary_text,
            "used_fields": target_columns
        }

    async def analyze_search_results(self, query: str, classification: Dict, panel_ids: List[str]) -> Tuple[Dict, str]:
        """[Pro 모드] 심층 분석 및 차트 생성"""
        if not panel_ids:
            return {"main_summary": "검색 결과가 없습니다.", "charts": []}, "검색 결과 없음"

        # 분석용 데이터 확보
        panels_data = self.panel_repo.get_panels_data_from_db(panel_ids[:5000])

        if not panels_data:
            return {"main_summary": "데이터 없음", "charts": []}, "데이터 없음"

        # 스마트 필터링 적용
        filters = classification.get('demographic_filters', {}).copy()
        if 'region_major' in filters:
            filters['region'] = filters.pop('region_major')

        panels_data = filter_merged_panels(panels_data, filters)

        if not panels_data:
             return {"main_summary": "조건에 맞는 데이터가 없습니다.", "charts": []}, "조건 불일치"

        # 차트 생성
        charts, used_fields = await self._generate_charts_optimized(query, classification, panels_data)

        # 요약 생성
        summary_text = await self._generate_result_overview(query, panel_ids, classification, panels_data)

        return {
            "query": query,
            "total_count": len(panels_data),
            "charts": charts
        }, summary_text

    async def _generate_charts_optimized(self, query: str, classification: Dict, panels_data: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        [최적화 V3] Pandas 연산 + 스마트 인사이트
        """
        charts = []
        used_fields = []
        chart_tasks = []
        search_used_fields = set()

        # 분석 대상 제한 (속도 최적화)
        stat_panels = panels_data[:1200] 

        # [A] 검색 필터 식별
        demographic_filters = classification.get('demographic_filters', {})
        if 'age_range' in demographic_filters: 
            search_used_fields.add('birth_year')
        for key in demographic_filters:
            if key != 'age_range': 
                search_used_fields.add(key)
        
        semantic_conditions = classification.get('semantic_conditions', [])
        if 'children_count' in demographic_filters or 'children_count' in search_used_fields:
            search_used_fields.add('marital_status')

        # [Logic 2] 지역 세분화
        if 'region_minor' not in used_fields:
            region_in_filters = 'region_major' in demographic_filters or 'region' in demographic_filters
            region_values = [p.get('region_major') for p in stat_panels if p.get('region_major')]
            unique_regions = set(region_values) if region_values else set()
            
            if region_in_filters or len(unique_regions) >= 2:
                chart_tasks.append({"type": "filter", "field": "region_minor", "priority": 0})
                used_fields.append("region_minor")
                search_used_fields.add('region_major')

        # [B] Target Field 처리
        target_field = classification.get('target_field')
        if target_field and target_field != 'unknown' and target_field not in used_fields:
            if target_field not in search_used_fields:
                priority = 0
                if target_field in QPOLL_FIELD_TO_TEXT:
                    chart_tasks.append({"type": "qpoll", "field": target_field, "priority": priority})
                    used_fields.append(target_field)
                    
                    basic_demos = ['gender', 'birth_year']
                    if 'region_minor' not in used_fields and 'region_major' not in search_used_fields:
                        basic_demos.append('region_major')
                    for field in basic_demos:
                        if field not in used_fields and field not in search_used_fields:
                            chart_tasks.append({"type": "filter", "field": field, "priority": 1})
                            used_fields.append(field)
                else:
                    chart_tasks.append({"type": "filter", "field": target_field, "priority": priority})
                    used_fields.append(target_field)

        # [C] Semantic Conditions 처리
        for condition in semantic_conditions:
            original_keyword = condition.get('original_keyword')
            if not original_keyword: continue
            field_info = router.find_closest_field(original_keyword)
            if field_info:
                found_field = field_info['field']
                if found_field in used_fields: continue
                prio = 1 if found_field not in search_used_fields else 3
                if found_field in QPOLL_FIELD_TO_TEXT:
                    chart_tasks.append({"type": "qpoll", "field": found_field, "priority": prio})
                else:
                    if found_field not in search_used_fields:
                        chart_tasks.append({"type": "filter", "field": found_field, "priority": prio})
                used_fields.append(found_field)

        # [Logic 4] 차량 소유 비율
        car_ownership_values = [p.get('car_ownership') for p in stat_panels if p.get('car_ownership')]
        if car_ownership_values:
            flat_values = []
            car_map = VALUE_TRANSLATION_MAP.get('car_ownership', {})
            for v in car_ownership_values:
                cleaned = clean_label(v if isinstance(v, str) else str(v))
                flat_values.append(car_map.get(cleaned, cleaned))
            car_dist = calculate_distribution(flat_values)
            if car_dist.get('있음', 0) >= 70.0 or car_dist.get('있음(자가)', 0) >= 70.0:
                if 'car_model_raw' not in used_fields:
                    chart_tasks.append({"type": "filter", "field": "car_model_raw", "priority": 2})
                    used_fields.append("car_model_raw")

        # ✅ Step 1: 기본 차트 생성
        chart_tasks.sort(key=lambda x: x['priority'])
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for task in chart_tasks[:4]:
                if task['type'] == 'qpoll':
                    futures.append(executor.submit(self._create_qpoll_chart, task['field']))
                else:
                    korean_name = FIELD_NAME_MAP.get(task['field'], task['field'])
                    futures.append(executor.submit(self._create_basic_chart, task['field'], korean_name, stat_panels))
            
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res and res.get('chart_data'):
                        vals = list(res['chart_data'][0]['values'].values())
                        if vals and vals[0] > 95.0 and res.get('field') in search_used_fields:
                            continue 
                        charts.append(res)
                except Exception as e:
                    logging.error(f"   ❌ [Chart Error] {e}")

        # ✅ Step 2: 교차 분석 (Rich Insight & Speed Optimization)
        pivot_field = target_field if (target_field and target_field in used_fields) else (used_fields[0] if used_fields else None)
        
        if pivot_field and len(charts) < 5:
            pivot_name = QPOLL_FIELD_TO_TEXT.get(pivot_field, FIELD_NAME_MAP.get(pivot_field, pivot_field))
            
            # 1. 축 선정을 위한 샘플링 (300명)
            sample_panels = stat_panels[:300]
            sample_ids = [p['panel_id'] for p in sample_panels]
            all_qpoll_fields = [field for field, _ in QPOLL_FIELDS]
            
            logging.info(f"   📥 [Crosstab] 축 선정용 샘플 로드 ({len(sample_ids)}명)")
            sample_qpoll_data = await asyncio.to_thread(
                self.qpoll_repo.get_responses_for_table, sample_ids, all_qpoll_fields
            )
            for p in sample_panels:
                if p['panel_id'] in sample_qpoll_data: p.update(sample_qpoll_data[p['panel_id']])
            
            # 2. 중요 축 선정 (검색 조건에 사용된 필드도 허용)
            recommended_axes = self._select_dynamic_crosstab_axes(
                pivot_field, sample_panels, search_used_fields, used_fields
            )
            
            if recommended_axes:
                max_crosstab = min(2, 5 - len(charts))
                selected_axis_fields = [ax[0] for ax in recommended_axes[:max_crosstab] if ax[0] in QPOLL_FIELD_TO_TEXT]
                
                # [최적화] 선정된 축에 대해 '배치(Batch)'로 데이터 로드
                if selected_axis_fields:
                    target_ids = [p['panel_id'] for p in stat_panels]
                    batch_size = 400
                    chunks = [target_ids[i:i + batch_size] for i in range(0, len(target_ids), batch_size)]
                    
                    logging.info(f"   🚀 [Batch Load] 교차분석 데이터 로드 ({len(target_ids)}명)")
                    tasks = [
                        asyncio.to_thread(self.qpoll_repo.get_responses_for_table, chunk, selected_axis_fields)
                        for chunk in chunks
                    ]
                    results = await asyncio.gather(*tasks)
                    
                    full_qpoll_data = {}
                    for res in results: full_qpoll_data.update(res)
                    for p in stat_panels:
                        if p['panel_id'] in full_qpoll_data: p.update(full_qpoll_data[p['panel_id']])

                # 3. 차트 생성
                crosstab_added = 0
                for ax_field, ax_name in recommended_axes:
                    if crosstab_added >= max_crosstab: break
                    
                    crosstab = self._create_crosstab_chart(
                        stat_panels, ax_field, pivot_field, ax_name, pivot_name
                    )
                    
                    if crosstab and crosstab.get('chart_data'):
                        # 그룹이 2개 이상일 때만 추가
                        if len(crosstab['chart_data'][0]['values']) >= 2:
                             charts.append(crosstab)
                             crosstab_added += 1

        # ✅ Step 3: 특이점 자동 발굴
        if len(charts) < 5:
            exclude = list(set(used_fields) | search_used_fields)
            if 'region_minor' in used_fields: exclude.append('region_major')
            if 'children_count' in demographic_filters: exclude.append('marital_status')

            high_ratio_charts = self._find_high_ratio_fields(
                stat_panels, exclude_fields=exclude, max_charts=(5 - len(charts))
            )
            charts.extend(high_ratio_charts)

        return charts[:5], used_fields

    def _calculate_axis_importance(self, panels_data: List[Dict], target_field: str, 
                                    candidate_field: str) -> float:
        """
        [속도 최적화] Pandas를 사용하여 축 중요도 계산
        """
        try:
            df = pd.DataFrame(panels_data, columns=[target_field, candidate_field])
            df.dropna(inplace=True)
            if len(df) < 10: return 0.0

            # 리스트 데이터 평탄화 (Explode)
            if df[target_field].apply(lambda x: isinstance(x, list)).any():
                df = df.explode(target_field)
            if df[candidate_field].apply(lambda x: isinstance(x, list)).any():
                df = df.explode(candidate_field)
            
            # 연령대 변환
            if candidate_field == 'birth_year':
                df[candidate_field] = df[candidate_field].apply(get_age_group)

            # 교차표 생성
            cross_tab = pd.crosstab(df[candidate_field], df[target_field])
            
            axis_diversity = len(cross_tab.index)
            target_diversity = len(cross_tab.columns)

            if axis_diversity < 2 or target_diversity < 2: return 0.0
            
            # 엔트로피 기반 다양성 점수 계산 (빠름)
            probs = cross_tab.div(cross_tab.sum(axis=1), axis=0)
            variance_score = probs.var().mean() * 10 

            return float(variance_score)
        except Exception:
            return 0.0

    def _select_dynamic_crosstab_axes(self, target_field: str, panels_data: List[Dict], 
                                       search_used_fields: Set[str], used_fields: List[str]) -> List[Tuple[str, str]]:
        """검색 조건에 포함된 필드라도 분포가 다양하면 축으로 허용"""
        logging.info(f"   🔍 [Dynamic Crosstab] 타겟: {target_field}")
        all_candidate_fields = []
        
        # 이미 그려진 차트는 제외, 하지만 검색 조건(search_used_fields)은 허용
        for field, name in WELCOME_OBJECTIVE_FIELDS:
            if field in used_fields: continue
            if field == target_field: continue
            all_candidate_fields.append((field, name))
        
        if target_field not in QPOLL_FIELD_TO_TEXT:
            for field, desc in QPOLL_FIELDS:
                if field in used_fields: continue
                if field == target_field: continue
                all_candidate_fields.append((field, desc))
        
        if not all_candidate_fields: return []
        
        axis_scores = []
        for candidate_field, candidate_name in all_candidate_fields:
            score = self._calculate_axis_importance(panels_data, target_field, candidate_field)
            if score > 0.1:
                axis_scores.append({'field': candidate_field, 'name': candidate_name, 'score': score})
        
        axis_scores.sort(key=lambda x: x['score'], reverse=True)
        return [(a['field'], a['name']) for a in axis_scores[:4]]

    def _create_crosstab_chart(self, panels_data, field1, field2, name1, name2) -> Dict:
        """
        [Rich Insight] 스마트 설명 생성 + 리스트 처리
        """
        crosstab = {}
        
        # 1. 축 데이터 수집 & 평탄화
        vals1 = []
        for p in panels_data:
            val = p.get(field1)
            if not val: continue
            if field1 == 'birth_year': val = get_age_group(val)
            
            if isinstance(val, list): vals1.extend([clean_label(v) for v in val])
            else: vals1.append(clean_label(val))
        
        if not vals1: return {}
        top_groups = [k for k, v in Counter(vals1).most_common(5)]

        group_insights = [] # 설명을 위한 데이터

        for group in top_groups:
            # 그룹 필터링
            group_panels = []
            for p in panels_data:
                p_val1 = p.get(field1)
                if field1 == 'birth_year': p_val1 = get_age_group(p_val1)
                
                is_match = False
                if isinstance(p_val1, list):
                    if str(group) in [clean_label(x) for x in p_val1]: is_match = True
                else:
                    if p_val1 and str(clean_label(p_val1)) == str(group): is_match = True
                if is_match: group_panels.append(p)
            
            # 피벗 데이터
            vals2 = []
            for p in group_panels:
                v = p.get(field2)
                if v:
                    if isinstance(v, list): vals2.extend(v)
                    else: vals2.append(v)
            
            if vals2:
                dist = calculate_distribution([clean_label(v) for v in vals2])
                sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                crosstab[str(group)] = dict(sorted_dist[:5])
                
                # [스마트 설명] 1위 답변 추출
                if sorted_dist:
                    top_ans, top_pct = sorted_dist[0]
                    group_insights.append(f"{group}(은)는 '{top_ans}'({top_pct}%)")

        if not crosstab: return {}

        # [스마트 설명] 동적 텍스트 생성
        # 예: "20대(은)는 '넷플릭스'(60%)를, 30대(은)는 '왓챠'(40%)를 가장 선호합니다."
        desc_text = f"'{name1}'에 따른 차이: " + ", ".join(group_insights[:2]) + "..."
        
        return {
            "topic": f"{name1}별 {name2} 분포",
            "description": desc_text,
            "chart_type": "crosstab",
            "chart_data": [{"label": f"{name1}별 {name2}", "values": crosstab}],
            "fields": [field1, field2]
        }

    def _find_high_ratio_fields(self, panels_data, exclude_fields, max_charts) -> List[Dict]:
        results = []
        candidates = [f for f in WELCOME_OBJECTIVE_FIELDS if f[0] not in exclude_fields]
        
        for field, kname in candidates:
            if len(results) >= max_charts: break
            vals = []
            for p in panels_data:
                v = p.get(field)
                if v:
                    if isinstance(v, list): vals.extend(v)
                    else: vals.append(v)
            if not vals: continue
            
            dist = calculate_distribution([clean_label(str(x)) for x in vals])
            if not dist: continue
            top_k, top_v = sorted(dist.items(), key=lambda x: x[1], reverse=True)[0]
            
            if 40.0 <= top_v < 95.0:
                results.append({
                    "topic": f"{kname} 특징",
                    "description": f"전체의 {top_v}%가 '{top_k}'입니다.",
                    "ratio": f"{top_v}%",
                    "chart_data": [{"label": kname, "values": dict(list(dist.items())[:10])}],
                    "field": field
                })
        return results

    def _create_basic_chart(self, field_name: str, korean_name: str, panels_data: List[Dict]) -> Dict:
        if field_name in ["children_count", "birth_year"]:
            distribution = self.panel_repo.get_field_distribution(field_name)
        else:
            values = []
            for item in panels_data:
                val = item.get(field_name)
                if val:
                    if isinstance(val, list): values.extend([clean_label(v) for v in val])
                    else: values.append(clean_label(val))
            distribution = calculate_distribution(values)

        if not distribution: return {}
        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:10]
        final_dist = dict(sorted_items)
        top_k, top_v = list(final_dist.items())[0]
        
        return {
            "topic": f"{korean_name} 분포",
            "description": f"가장 많은 응답은 '{top_k}'({top_v}%) 입니다.",
            "ratio": f"{top_v}%",
            "chart_data": [{"label": korean_name, "values": final_dist}],
            "field": field_name
        }

    def _create_qpoll_chart(self, field_name: str) -> Dict:
        distribution = self.qpoll_repo.get_distribution(field_name)
        if not distribution: return {}
        question_text = QPOLL_FIELD_TO_TEXT.get(field_name, field_name)
        top_k, top_v = list(distribution.items())[0]
        return {
            "topic": question_text,
            "description": f"가장 많은 응답은 '{top_k}'({top_v}%) 입니다.",
            "ratio": f"{top_v}%",
            "chart_data": [{"label": question_text, "values": distribution}],
            "field": field_name
        }

    def _calculate_column_stats(self, df: pd.DataFrame, columns: List[str]) -> str:
        report = []
        for col in columns:
            if col not in df.columns: continue
            try:
                valid = df[col].dropna()
                if valid.empty: continue
                if valid.apply(lambda x: isinstance(x, list)).any():
                    counts = valid.explode().value_counts().head(5)
                else:
                    counts = valid.value_counts().head(5)
                korean_name = FIELD_NAME_MAP.get(col, col)
                lines = [f"[{korean_name}]"]
                for val, count in counts.items():
                    pct = (count / len(df)) * 100
                    lines.append(f"- {val}: {pct:.1f}%")
                report.append("\n".join(lines))
            except: pass
        return "\n\n".join(report)

    async def _generate_result_overview(self, query: str, panel_ids: List[str], classification: Dict, panels_data: List[Dict]) -> str:
        if not panels_data: return "데이터가 없습니다."
        df = pd.DataFrame(panels_data[:1000])
        
        filter_summary = []
        filters = classification.get('demographic_filters', {})
        for k, v in filters.items():
            filter_summary.append(f"- {FIELD_NAME_MAP.get(k, k)}: {v}")
        semantic = classification.get('semantic_conditions', [])
        for s in semantic:
            filter_summary.append(f"- 의도 조건: {s.get('original_keyword')}")
        filter_text = "\n".join(filter_summary)

        stats_context = []
        target_field = classification.get('target_field')
        if target_field and target_field in df.columns:
            counts = df[target_field].value_counts(normalize=True).head(3)
            if not counts.empty:
                items = [f"{k}({v*100:.1f}%)" for k, v in counts.items()]
                kname = FIELD_NAME_MAP.get(target_field, target_field)
                stats_context.append(f"[{kname} 분포]: {', '.join(items)}")

        if 'birth_year' in df.columns:
            df['age_group'] = df['birth_year'].apply(lambda x: get_age_group(x) if x else None)
            age_counts = df['age_group'].value_counts(normalize=True).head(3)
            if not age_counts.empty:
                age_desc = [f"{age}({ratio*100:.1f}%)" for age, ratio in age_counts.items()]
                stats_context.append(f"[연령대 분포]: {', '.join(age_desc)}")

        if 'income_personal_monthly' in df.columns:
            top_income = df['income_personal_monthly'].value_counts(normalize=True).head(2) 
            if not top_income.empty:
                income_strs = [f"{k}({v*100:.1f}%)" for k, v in top_income.items()]
                stats_context.append(f"[개인 월소득 분포]: {', '.join(income_strs)}")
        
        interest_fields = ['gender', 'region_major', 'job_title_raw', 'income_household_monthly', 'marital_status']
        for field in interest_fields:
            if field in df.columns:
                top = df[field].value_counts(normalize=True).head(1)
                if not top.empty:
                    val, ratio = top.index[0], top.values[0]
                    desc = f"{val}({ratio*100:.1f}%)"
                    if ratio >= 0.5: desc += " - 과반수 이상"
                    kname = FIELD_NAME_MAP.get(field, field)
                    stats_context.append(f"[{kname}]: {desc}")

        full_context = f"검색 조건:\n{filter_text}\n\n발견된 특징:\n{chr(10).join(stats_context)}"
        return await self.llm_service.generate_analysis_summary(query, full_context, len(panel_ids))