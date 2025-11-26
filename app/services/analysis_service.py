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
from app.utils.text_utils import clean_label, calculate_distribution, get_age_group, truncate_text
from app.utils.common import find_target_columns_dynamic, get_field_mapping

# 매핑 규칙 및 상수
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

    # --------------------------------------------------------------------------
    # [핵심] Insights.py의 지능형 차트 추천 로직 이식
    # --------------------------------------------------------------------------
    async def _generate_charts_optimized(self, query: str, classification: Dict, panels_data: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        분석 우선순위:
        1. 타겟 필드 (Target Field)
        2. 의도 분석 (Semantic Analysis) - 검색어의 숨은 뜻 파악
        3. 검색 필터 (Demographic Filters)
        4. 교차 분석 (Crosstab) - 차트 부족 시 자동 생성
        5. 특이점 발견 (High Ratio) - 쏠림 현상이 있는 데이터 자동 발견
        """
        charts = []
        used_fields = []
        chart_tasks = []
        search_used_fields = set()

        # (1) Target Field (0순위)
        target_field = classification.get('target_field')
        if target_field and target_field != 'unknown':
            if target_field in QPOLL_FIELD_TO_TEXT:
                chart_tasks.append({"type": "qpoll", "field": target_field, "priority": 0})
            else:
                chart_tasks.append({"type": "filter", "field": target_field, "priority": 0})
            used_fields.append(target_field)

        # (2) Semantic Conditions (1순위 - 의도 분석)
        semantic_conditions = classification.get('semantic_conditions', [])
        for condition in semantic_conditions:
            original_keyword = condition.get('original_keyword')
            if not original_keyword: continue
            
            # Semantic Router 활용해 연관 필드 찾기
            field_info = router.find_closest_field(original_keyword)
            if field_info:
                found_field = field_info['field']
                if found_field in used_fields: continue
                
                logging.info(f"💡 [Insight] 2차 의도 발견: '{original_keyword}' -> '{field_info['description']}' ({found_field})")
                
                if found_field in QPOLL_FIELD_TO_TEXT:
                    chart_tasks.append({"type": "qpoll", "field": found_field, "priority": 1})
                else:
                    chart_tasks.append({"type": "filter", "field": found_field, "priority": 1})
                used_fields.append(found_field)

        # (3) Demographic Filters
        filters = classification.get('demographic_filters', {})
        for key in filters:
            if key not in used_fields and key != 'age_range':
                chart_tasks.append({"type": "filter", "field": key, "priority": 2})
                used_fields.append(key)
                search_used_fields.add(key)
        
        if 'age_range' in filters and 'birth_year' not in used_fields:
            chart_tasks.append({"type": "filter", "field": "birth_year", "priority": 2})
            used_fields.append('birth_year')

        # [실행] 차트 데이터 생성 (병렬 처리)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for task in chart_tasks:
                if task['type'] == 'qpoll':
                    futures.append(executor.submit(self._create_qpoll_chart, task['field']))
                else:
                    korean_name = FIELD_NAME_MAP.get(task['field'], task['field'])
                    futures.append(executor.submit(self._create_basic_chart, task['field'], korean_name, panels_data))
            
            for future in as_completed(futures):
                try:
                    chart = future.result()
                    if chart and chart.get('chart_data'):
                        charts.append(chart)
                except Exception as e:
                    logging.error(f"차트 생성 실패: {e}")

        # (4) [Fallback] 차트가 부족하면 교차 분석(Crosstab) 추가
        if len(charts) < 5:
            pivot_field = target_field if target_field in used_fields else (used_fields[0] if used_fields else None)
            
            if pivot_field:
                pivot_name = QPOLL_FIELD_TO_TEXT.get(pivot_field, FIELD_NAME_MAP.get(pivot_field, pivot_field))
                standard_axes = [('birth_year','연령대'), ('gender','성별'), ('region_major','지역'), ('job_title_raw','직업')]
                
                for ax_field, ax_name in standard_axes:
                    if len(charts) >= 5: break
                    if ax_field == pivot_field or ax_field in search_used_fields: continue
                    
                    crosstab = self._create_crosstab_chart(panels_data, ax_field, pivot_field, ax_name, pivot_name)
                    if crosstab:
                        charts.append(crosstab)

        # (5) [Fallback] 그래도 부족하면 특이점(High Ratio) 필드 자동 발굴
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
        df = pd.DataFrame(panels_data[:1000])
        stats_context = []
        target_field = classification.get('target_field')
        if target_field and target_field in df.columns:
            counts = df[target_field].value_counts(normalize=True).head(3)
            items = [f"{k}({v*100:.1f}%)" for k, v in counts.items()]
            kname = FIELD_NAME_MAP.get(target_field, target_field)
            stats_context.append(f"[{kname}]: {', '.join(items)}")
        for field in ['gender', 'region_major']:
            if field in df.columns:
                top = df[field].value_counts(normalize=True).head(1)
                if not top.empty:
                    k, v = top.index[0], top.values[0]
                    stats_context.append(f"[{FIELD_NAME_MAP.get(field)}]: {k}({v*100:.1f}%)")
        full_stats = "\n".join(stats_context)
        return await self.llm_service.generate_analysis_summary(query, full_stats, len(panel_ids))