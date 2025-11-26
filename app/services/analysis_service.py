import logging
import pandas as pd
import asyncio
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from app.repositories.panel_repo import PanelRepository
from app.repositories.qpoll_repo import QpollRepository
from app.services.llm_service import LLMService
from app.core.semantic_router import router  
from app.utils.common import (
    find_target_columns_dynamic, 
    get_field_mapping, 
    clean_label, 
    calculate_distribution, 
    get_age_group, 
    truncate_text
)
from app.constants.mapping import (
    FIELD_NAME_MAP, 
    QPOLL_FIELD_TO_TEXT, 
    WELCOME_OBJECTIVE_FIELDS,
    VALUE_TRANSLATION_MAP,
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
        """[Pro 모드] 심층 분석 및 차트 생성 (Insights 로직 완벽 복원)"""
        if not panel_ids:
            return {"main_summary": "검색 결과가 없습니다.", "charts": []}, "검색 결과 없음"

        # 데이터 로드 (분석용으로 최대 5000명)
        panels_data = self.panel_repo.get_panels_by_ids(panel_ids[:5000])
        if not panels_data:
            return {"main_summary": "데이터 없음", "charts": []}, "데이터 없음"

        # 차트 생성 로직 
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
        심층 분석 우선순위 및 누락된 비즈니스 로직 복원:
        1. 타겟 필드 (Target Field)
        2. 의도 분석 (Semantic Analysis)
        3. 필터 & 파생 로직 (Region Minor, Single Household 등) -> 복원됨
        4. 데이터 기반 로직 (Car Ownership >= 70% -> Car Model) -> 복원됨
        5. 교차 분석 (Crosstab)
        6. 특이점 발견 (High Ratio)
        """
        charts = []
        used_fields = []
        chart_tasks = []
        search_used_fields = set()

        # [A] 필터 정보 추출 및 기록
        demographic_filters = classification.get('demographic_filters', {})
        if 'age_range' in demographic_filters: search_used_fields.add('birth_year')
        for key in demographic_filters:
            if key != 'age_range': search_used_fields.add(key)
        
        # (1) Target Field (0순위)
        target_field = classification.get('target_field')
        if target_field and target_field != 'unknown':
            if target_field in QPOLL_FIELD_TO_TEXT:
                chart_tasks.append({"type": "qpoll", "field": target_field, "priority": 0})
            else:
                chart_tasks.append({"type": "filter", "field": target_field, "priority": 0})
            used_fields.append(target_field)

            # [Logic 1 복원] Q-Poll 타겟인 경우 기본 인구통계(성별, 연령, 지역) 자동 추가
            if target_field in QPOLL_FIELD_TO_TEXT:
                basic_demos = ['gender', 'birth_year', 'region_major']
                for field in basic_demos:
                    if field not in used_fields and field not in search_used_fields:
                        chart_tasks.append({"type": "filter", "field": field, "priority": 1})
                        used_fields.append(field)

        # (2) Semantic Conditions (1순위 - 의도 분석)
        semantic_conditions = classification.get('semantic_conditions', [])
        for condition in semantic_conditions:
            original_keyword = condition.get('original_keyword')
            if not original_keyword: continue
            
            field_info = router.find_closest_field(original_keyword)
            if field_info:
                found_field = field_info['field']
                if found_field in used_fields: continue
                
                if found_field in QPOLL_FIELD_TO_TEXT:
                    chart_tasks.append({"type": "qpoll", "field": found_field, "priority": 1})
                else:
                    chart_tasks.append({"type": "filter", "field": found_field, "priority": 1})
                used_fields.append(found_field)

        # (3) Demographic Filters 및 파생 로직
        # 지역 필터(region_major)가 있으면 세부 지역(region_minor) 자동 추가
        if 'region_major' in demographic_filters and 'region_minor' not in used_fields:
            chart_tasks.append({"type": "filter", "field": "region_minor", "priority": 2})
            used_fields.append("region_minor")

        for key in demographic_filters:
            if key not in used_fields and key != 'age_range':
                chart_tasks.append({"type": "filter", "field": key, "priority": 2})
                used_fields.append(key)
        
        if 'age_range' in demographic_filters and 'birth_year' not in used_fields:
            chart_tasks.append({"type": "filter", "field": "birth_year", "priority": 2})
            used_fields.append('birth_year')

        # 1인 가구 로직
        is_single_household = False
        fam_val = demographic_filters.get('family_size') or demographic_filters.get('household_size')
        if fam_val:
            if isinstance(fam_val, list) and any(str(v).startswith('1') for v in fam_val): is_single_household = True
            elif str(fam_val).startswith('1'): is_single_household = True
            
        if is_single_household and 'income_household_monthly' not in used_fields:
            chart_tasks.append({"type": "filter", "field": "income_household_monthly", "priority": 2})
            used_fields.append('income_household_monthly')

        # 차량 소유 비율 70% 이상 시 '소유 여부' 대신 '차종' 차트 노출
        car_ownership_values = [p.get('car_ownership') for p in panels_data if p.get('car_ownership')]
        if car_ownership_values:
            flat_values = []
            car_map = VALUE_TRANSLATION_MAP.get('car_ownership', {})
            
            for v in car_ownership_values:
                if isinstance(v, list):
                    for sub_v in v:
                        cleaned = clean_label(sub_v)
                        flat_values.append(car_map.get(cleaned, cleaned))
                else:
                    cleaned = clean_label(v)
                    flat_values.append(car_map.get(cleaned, cleaned))
            
            if flat_values:
                car_dist = calculate_distribution(flat_values)
                # '있음' 관련 응답이 70% 이상이면 차종 차트 추가
                if car_dist.get('있음', 0) >= 70.0 or car_dist.get('있음(자가)', 0) >= 70.0:
                    if 'car_model_raw' not in used_fields:
                        chart_tasks.append({"type": "filter", "field": "car_model_raw", "priority": 1})
                        used_fields.append("car_model_raw")
                    
                    # 차량 소유 여부 차트는 중복되므로 제거를 위해 used_fields에 등록만 해둠
                    if 'car_ownership' not in used_fields:
                        used_fields.append("car_ownership")

        # 우선순위 정렬 (낮은 숫자가 먼저)
        chart_tasks.sort(key=lambda x: x['priority'])

        # 차트 데이터 생성 (병렬 처리)
        final_charts = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for task in chart_tasks[:6]: # 상위 6개까지만 시도
                if task['type'] == 'qpoll':
                    futures.append(executor.submit(self._create_qpoll_chart, task['field']))
                else:
                    korean_name = FIELD_NAME_MAP.get(task['field'], task['field'])
                    futures.append(executor.submit(self._create_basic_chart, task['field'], korean_name, panels_data))
            
            for future in as_completed(futures):
                try:
                    chart = future.result()
                    if chart and chart.get('chart_data'):
                        final_charts.append(chart)
                except Exception:
                    pass
        
        charts.extend(final_charts[:5]) # 최종 5개 선정

        # (5) 차트가 부족하면 교차 분석(Crosstab) 추가
        if len(charts) < 5:
            pivot_field = target_field if (target_field and target_field in used_fields) else (used_fields[0] if used_fields else None)
            
            if pivot_field:
                pivot_name = QPOLL_FIELD_TO_TEXT.get(pivot_field, FIELD_NAME_MAP.get(pivot_field, pivot_field))
                standard_axes = [('birth_year','연령대'), ('gender','성별'), ('region_major','지역'), ('job_title_raw','직업')]
                
                for ax_field, ax_name in standard_axes:
                    if len(charts) >= 5: break
                    if ax_field == pivot_field or ax_field in search_used_fields: continue
                    
                    crosstab = self._create_crosstab_chart(panels_data, ax_field, pivot_field, ax_name, pivot_name)
                    if crosstab and crosstab.get('chart_data'):
                        charts.append(crosstab)

        # (6) 그래도 부족하면 특이점(High Ratio) 필드 자동 발굴
        if len(charts) < 5:
            high_ratio_charts = self._find_high_ratio_fields(panels_data, exclude_fields=used_fields, max_charts=5-len(charts))
            charts.extend(high_ratio_charts)

        return charts, used_fields

    def _create_crosstab_chart(self, panels_data, field1, field2, name1, name2) -> Dict:
        """교차 분석 차트 생성"""
        crosstab = {}
        
        vals1 = [p.get(field1) for p in panels_data if p.get(field1)]
        if field1 == 'birth_year': vals1 = [get_age_group(v) for v in vals1]
        if not vals1: return {}
        
        top_groups = [k for k, v in Counter(vals1).most_common(5)]

        for group in top_groups:
            group_panels = []
            for p in panels_data:
                p_val1 = p.get(field1)
                if field1 == 'birth_year': p_val1 = get_age_group(p_val1)
                if str(p_val1) == str(group):
                    group_panels.append(p)
            
            vals2 = []
            for p in group_panels:
                v = p.get(field2)
                if v:
                    if isinstance(v, list): vals2.extend(v)
                    else: vals2.append(v)
            
            if vals2:
                dist = calculate_distribution([clean_label(v) for v in vals2])
                crosstab[str(group)] = dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5])

        if not crosstab: return {}

        return {
            "topic": f"{name1}별 {name2} 분포",
            "description": f"'{name1}'에 따른 '{name2}' 응답 차이를 보여줍니다.",
            "chart_type": "crosstab",
            "chart_data": [{"label": f"{name1}별 {name2}", "values": crosstab}],
            "fields": [field1, field2]
        }

    def _find_high_ratio_fields(self, panels_data, exclude_fields, max_charts) -> List[Dict]:
        """특이점(High Ratio) 필드 자동 발굴"""
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
            
            if 50.0 <= top_v < 98.0:
                results.append({
                    "topic": f"{kname} 특징",
                    "description": f"전체의 {top_v}%가 '{top_k}'입니다.",
                    "ratio": f"{top_v}%",
                    "chart_data": [{"label": kname, "values": dict(list(dist.items())[:10])}],
                    "field": field
                })
        
        return results

    def _create_basic_chart(self, field_name: str, korean_name: str, panels_data: List[Dict]) -> Dict:
        """일반 필드(DB) 차트 데이터 생성"""
        if field_name in ["children_count", "birth_year"]:
            distribution = self.panel_repo.get_field_distribution(field_name)
        else:
            values = []
            for item in panels_data:
                val = item.get(field_name)
                if val:
                    if isinstance(val, list):
                        values.extend([clean_label(v) for v in val])
                    else:
                        values.append(clean_label(val))
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
        """Q-Poll 차트"""
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
        """
        검색 결과 요약 텍스트 생성 (연령대/소득 분석 로직 복원)
        """
        if not panels_data:
            return "데이터가 없습니다."

        df = pd.DataFrame(panels_data[:1000]) # 상위 1000명 샘플링
        stats_context = []
        
        # 타겟 필드 통계
        target_field = classification.get('target_field')
        if target_field and target_field in df.columns:
            counts = df[target_field].value_counts(normalize=True).head(3)
            items = [f"{k}({v*100:.1f}%)" for k, v in counts.items()]
            kname = FIELD_NAME_MAP.get(target_field, target_field)
            stats_context.append(f"[{kname}]: {', '.join(items)}")

        # 연령대(age_group) 변환 및 분석
        if 'birth_year' in df.columns:
            df['age_group'] = df['birth_year'].apply(lambda x: get_age_group(x) if x else None)
            age_counts = df['age_group'].value_counts(normalize=True).head(3)
            if not age_counts.empty:
                age_desc = [f"{age}({ratio*100:.1f}%)" for age, ratio in age_counts.items()]
                stats_context.append(f"[연령대]: {', '.join(age_desc)}")

        #  기본 인구통계 (성별, 지역)
        for field in ['gender', 'region_major']:
            if field in df.columns:
                top = df[field].value_counts(normalize=True).head(1)
                if not top.empty:
                    k, v = top.index[0], top.values[0]
                    kname = FIELD_NAME_MAP.get(field, field)
                    # 과반수 체크 등 텍스트 강화
                    desc = f"{k}({v*100:.1f}%)"
                    if v >= 0.5: desc += " - 과반수 이상"
                    stats_context.append(f"[{kname}]: {desc}")

        # 소득 수준 특징 발견 (30% 이상 쏠림 시)
        if 'income_personal_monthly' in df.columns:
            top_income = df['income_personal_monthly'].value_counts(normalize=True).head(1)
            if not top_income.empty and top_income.values[0] > 0.3:
                stats_context.append(f"[주요 소득구간]: {top_income.index[0]} ({top_income.values[0]*100:.1f}%)")

        full_stats = "\n".join(stats_context)
        
        # LLM 호출
        return await self.llm_service.generate_analysis_summary(query, full_stats, len(panel_ids))