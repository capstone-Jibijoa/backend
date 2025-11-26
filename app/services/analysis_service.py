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
    truncate_text,
    filter_merged_panels
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
        """[Pro 모드] 심층 분석 및 차트 생성 (Capstone Logic 복원 + 스마트 필터링)"""
        if not panel_ids:
            return {"main_summary": "검색 결과가 없습니다.", "charts": []}, "검색 결과 없음"

        panels_data = self.panel_repo.get_panels_data_from_db(panel_ids[:5000])

        if not panels_data:
            return {"main_summary": "데이터 없음", "charts": []}, "데이터 없음"

        # 메모리 상 스마트 필터링 적용
        # LLM이 추출한 필터(demographic_filters)를 가져옵니다.
        filters = classification.get('demographic_filters', {}).copy()
        
        if 'region_major' in filters:
            filters['region'] = filters.pop('region_major')

        # 필터링 수행
        panels_data = filter_merged_panels(panels_data, filters)

        # 필터링 후 데이터가 한 건도 없으면 종료
        if not panels_data:
             return {"main_summary": "조건에 맞는 데이터가 없습니다 (필터링 됨).", "charts": []}, "조건 불일치"

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
        1. 검색 조건(필터)에 사용된 필드는 'search_used_fields'로 등록하여 중복/당연한 차트 생성을 방지합니다.
        2. 파생 로직(지역->세부지역, 자녀->결혼무시, 차량->차종)을 우선 적용합니다.
        3. Semantic Intent 및 Target Field를 시각화합니다.
        """
        charts = []
        used_fields = []
        chart_tasks = []
        search_used_fields = set()

        # [A] 검색 필터(Demographic & Semantic) 식별 -> 차트 생성 방지용
        demographic_filters = classification.get('demographic_filters', {})
        if 'age_range' in demographic_filters: search_used_fields.add('birth_year')
        for key in demographic_filters:
            if key != 'age_range': search_used_fields.add(key)
        
        semantic_conditions = classification.get('semantic_conditions', [])
        # 의도 분석에 사용된 필드도 식별 (단, 시각화 가치가 있으면 아래에서 허용)

        # [Logic 1] 자녀 필터 있으면 결혼 상태 차트는 무의미하므로 제외 처리
        if 'children_count' in demographic_filters or 'children_count' in search_used_fields:
            # [FIX] 자녀가 있으면 99% 기혼이므로, 결혼 여부 차트는 정보 가치가 낮아 생성 제외 대상에 추가
            search_used_fields.add('marital_status')

        # [Logic 2] 지역 필터(Region Major) 있으면 -> 세부 지역(Region Minor) 차트 자동 추가 (0순위)
        if 'region_major' in demographic_filters and 'region_minor' not in used_fields:
            chart_tasks.append({"type": "filter", "field": "region_minor", "priority": 0})
            used_fields.append("region_minor")

        # [B] Target Field (0순위) - 사용자가 가장 궁금해하는 핵심 질문
        target_field = classification.get('target_field')
        if target_field and target_field != 'unknown' and target_field not in used_fields:
            # 타겟 필드가 검색 조건(필터)으로 쓰이지 않았을 때만 0순위 (쓰였으면 100%이므로 제외하거나 후순위)
            priority = 0 if target_field not in search_used_fields else 9
            
            if target_field in QPOLL_FIELD_TO_TEXT:
                chart_tasks.append({"type": "qpoll", "field": target_field, "priority": priority})
                used_fields.append(target_field)
                
                # [Logic 3] Q-Poll 타겟이면 기본 인구통계(성별, 연령, 지역) 자동 추가 (1순위)
                # 단, 이미 검색 조건으로 쓰인 필드는 제외
                basic_demos = ['gender', 'birth_year', 'region_major']
                for field in basic_demos:
                    if field not in used_fields and field not in search_used_fields:
                        chart_tasks.append({"type": "filter", "field": field, "priority": 1})
                        used_fields.append(field)
            else:
                # 타겟 필드가 정형 데이터인 경우
                if target_field not in search_used_fields:
                    chart_tasks.append({"type": "filter", "field": target_field, "priority": priority})
                    used_fields.append(target_field)

        # [C] Semantic Conditions (1순위) - 사용자의 의도 파악
        for condition in semantic_conditions:
            original_keyword = condition.get('original_keyword')
            if not original_keyword: continue
            
            field_info = router.find_closest_field(original_keyword)
            if field_info:
                found_field = field_info['field']
                if found_field in used_fields: continue
                
                # 검색 조건에 사용되었더라도 시각화 가치가 있을 수 있음 (하지만 우선순위 낮춤)
                prio = 1 if found_field not in search_used_fields else 3
                
                if found_field in QPOLL_FIELD_TO_TEXT:
                    chart_tasks.append({"type": "qpoll", "field": found_field, "priority": prio})
                else:
                    # 이미 필터로 쓴 정형 데이터는 차트에서 제외 (예: "남성" -> 성별 차트 X)
                    if found_field not in search_used_fields:
                        chart_tasks.append({"type": "filter", "field": found_field, "priority": prio})
                used_fields.append(found_field)

        # [Logic 4] 차량 소유 비율 70% 이상 시 '차종(Car Model)' 자동 분석
        car_ownership_values = [p.get('car_ownership') for p in panels_data if p.get('car_ownership')]
        if car_ownership_values:
            flat_values = []
            car_map = VALUE_TRANSLATION_MAP.get('car_ownership', {})
            for v in car_ownership_values:
                cleaned = clean_label(v if isinstance(v, str) else str(v))
                flat_values.append(car_map.get(cleaned, cleaned))
            
            car_dist = calculate_distribution(flat_values)
            if car_dist.get('있음', 0) >= 70.0 or car_dist.get('있음(자가)', 0) >= 70.0:
                if 'car_model_raw' not in used_fields:
                    chart_tasks.append({"type": "filter", "field": "car_model_raw", "priority": 1})
                    used_fields.append("car_model_raw")
                if 'car_ownership' not in used_fields:
                    pass 

        # 실행: Priority 순으로 정렬하여 차트 생성
        chart_tasks.sort(key=lambda x: x['priority'])
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            # 상위 5~6개만 시도
            for task in chart_tasks[:6]:
                if task['type'] == 'qpoll':
                    futures.append(executor.submit(self._create_qpoll_chart, task['field']))
                else:
                    korean_name = FIELD_NAME_MAP.get(task['field'], task['field'])
                    futures.append(executor.submit(self._create_basic_chart, task['field'], korean_name, panels_data))
            
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res and res.get('chart_data'):
                        # [검증] 값이 99% 이상 하나로 쏠려있고, 그것이 검색 필터라면 제외 (Obvious Chart)
                        vals = list(res['chart_data'][0]['values'].values())
                        # 예: 남성만 검색했는데 성별 차트가 나오면 100% 남성이므로 제외
                        if vals and vals[0] > 95.0 and res.get('field') in search_used_fields:
                            continue 
                        charts.append(res)
                except Exception:
                    pass

        # [D] 차트 부족 시: 교차 분석 (Crosstab)
        if len(charts) < 5:
            pivot_field = target_field if (target_field and target_field in used_fields) else (used_fields[0] if used_fields else None)
            if pivot_field:
                pivot_name = QPOLL_FIELD_TO_TEXT.get(pivot_field, FIELD_NAME_MAP.get(pivot_field, pivot_field))
                standard_axes = [('birth_year','연령대'), ('gender','성별'), ('region_major','지역'), ('job_title_raw','직업')]
                
                for ax_field, ax_name in standard_axes:
                    if len(charts) >= 5: break
                    # 축으로 사용할 필드가 이미 100% 쏠린 필드(검색 조건)라면 건너뜀
                    if ax_field == pivot_field or ax_field in search_used_fields: continue
                    
                    crosstab = self._create_crosstab_chart(panels_data, ax_field, pivot_field, ax_name, pivot_name)
                    if crosstab and crosstab.get('chart_data'):
                        charts.append(crosstab)

        # [E] 차트 부족 시: 특이점(High Ratio) 자동 발굴
        if len(charts) < 5:
            # 이미 사용된 필드와 검색 필터를 제외한 나머지 중에서 찾기
            exclude_for_high_ratio = list(set(used_fields) | search_used_fields)
            
            # [New Logic] '결혼 여부'도 자녀 조건이 있으면 제외 대상에 추가
            if 'children_count' in demographic_filters or 'children_count' in search_used_fields:
                exclude_for_high_ratio.append('marital_status')

            high_ratio_charts = self._find_high_ratio_fields(panels_data, exclude_fields=exclude_for_high_ratio, max_charts=5-len(charts))
            charts.extend(high_ratio_charts)

        return charts[:5], used_fields

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
            
            # 기준 완화: 40% 이상 ~ 95% 미만 (너무 뻔한 99% 데이터는 제외)
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
        검색 결과 요약: '검색 조건'과 '새로 발견된 특징'을 명확히 구분 (Capstone Logic)
        """
        if not panels_data:
            return "데이터가 없습니다."

        df = pd.DataFrame(panels_data[:1000])
        
        # 1. 검색 조건 텍스트 구성
        filter_summary = []
        filters = classification.get('demographic_filters', {})
        for k, v in filters.items():
            filter_summary.append(f"- {FIELD_NAME_MAP.get(k, k)}: {v}")
        
        semantic = classification.get('semantic_conditions', [])
        for s in semantic:
            filter_summary.append(f"- 의도 조건: {s.get('original_keyword')}")
            
        filter_text = "\n".join(filter_summary)

        # 2. 통계 데이터 수집 
        stats_context = []
        
        # 타겟 필드 통계 (Top 3)
        target_field = classification.get('target_field')
        if target_field and target_field in df.columns:
            counts = df[target_field].value_counts(normalize=True).head(3)
            if not counts.empty:
                items = [f"{k}({v*100:.1f}%)" for k, v in counts.items()]
                kname = FIELD_NAME_MAP.get(target_field, target_field)
                stats_context.append(f"[{kname} 분포]: {', '.join(items)}")

        # 연령대 통계 (Top 3)
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
        
        # 기타 주요 인구통계 (성별, 지역, 직업 등)
        interest_fields = ['gender', 'region_major', 'job_title_raw', 'income_household_monthly', 'marital_status']
        for field in interest_fields:
            if field in df.columns:
                top = df[field].value_counts(normalize=True).head(1) # 주요 특징만
                if not top.empty:
                    val, ratio = top.index[0], top.values[0]
                    desc = f"{val}({ratio*100:.1f}%)"
                    if ratio >= 0.5: desc += " - 과반수 이상"
                    kname = FIELD_NAME_MAP.get(field, field)
                    stats_context.append(f"[{kname}]: {desc}")

        # 3. LLM 요약 요청 (조건과 발견을 분리)
        full_context = f"""
        {filter_text}
        {chr(10).join(stats_context)}
        """
        
        return await self.llm_service.generate_analysis_summary(query, full_context, len(panel_ids))