import logging
import pandas as pd
import numpy as np
import asyncio
from typing import List, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict

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

        panels_data = self.panel_repo.get_panels_data_from_db(panel_ids[:5000])

        if not panels_data:
            return {"main_summary": "데이터 없음", "charts": []}, "데이터 없음"

        # 메모리 상 스마트 필터링 적용
        filters = classification.get('demographic_filters', {}).copy()
        
        if 'region_major' in filters:
            filters['region'] = filters.pop('region_major')

        panels_data = filter_merged_panels(panels_data, filters)

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
        완전 동적 차트 생성 (데이터 기반 교차 분석)
        교차 분석을 위한 Q-Poll 데이터 사전 로드 포함
        """
        
        # ✅ Step 0: Q-Poll 데이터 미리 로드 (교차 분석에 필요)
        panel_ids = [p['panel_id'] for p in panels_data]
        all_qpoll_fields = [field for field, _ in QPOLL_FIELDS]
        
        logging.info(f"   📥 [Data Load] Q-Poll 데이터 로드 시작: {len(panel_ids)}명, {len(all_qpoll_fields)}개 필드")
        
        qpoll_data = await asyncio.to_thread(
            self.qpoll_repo.get_responses_for_table, 
            panel_ids, 
            all_qpoll_fields
        )
        
        # panels_data에 Q-Poll 데이터 병합
        merged_count = 0
        for panel in panels_data:
            pid = panel['panel_id']
            if pid in qpoll_data:
                panel.update(qpoll_data[pid])
                merged_count += 1
        
        logging.info(f"   ✅ [Data Load] Q-Poll 데이터 병합 완료: {merged_count}/{len(panels_data)}명")
        
        # 기존 로직 시작
        charts = []
        used_fields = []
        chart_tasks = []
        search_used_fields = set()

        # [A] 검색 필터 식별
        demographic_filters = classification.get('demographic_filters', {})
        if 'age_range' in demographic_filters: 
            search_used_fields.add('birth_year')
        for key in demographic_filters:
            if key != 'age_range': 
                search_used_fields.add(key)
        
        semantic_conditions = classification.get('semantic_conditions', [])

        # [Logic 1] 자녀 필터 있으면 결혼 상태 제외
        if 'children_count' in demographic_filters or 'children_count' in search_used_fields:
            search_used_fields.add('marital_status')

        # [Logic 2] 지역 세분화 + region_major 완전 제외
        if 'region_minor' not in used_fields:
            region_in_filters = 'region_major' in demographic_filters or 'region' in demographic_filters
            
            region_values = [p.get('region_major') for p in panels_data if p.get('region_major')]
            unique_regions = set(region_values) if region_values else set()
            has_region_diversity = len(unique_regions) >= 2
            
            if region_in_filters or has_region_diversity:
                reasons = []
                if region_in_filters: reasons.append("필터")
                if has_region_diversity: reasons.append(f"분포({len(unique_regions)}개)")
                
                logging.info(f"   🗺️ [Logic 2] region_minor 추가: {', '.join(reasons)}")
                chart_tasks.append({"type": "filter", "field": "region_minor", "priority": 0})
                used_fields.append("region_minor")
                
                search_used_fields.add('region_major')
                logging.info("   🗺️ [Logic 2] region_major 제외 처리")

        # [B] Target Field
        target_field = classification.get('target_field')
        if target_field and target_field != 'unknown' and target_field not in used_fields:
            if target_field not in search_used_fields:
                priority = 0
                
                if target_field in QPOLL_FIELD_TO_TEXT:
                    chart_tasks.append({"type": "qpoll", "field": target_field, "priority": priority})
                    used_fields.append(target_field)
                    
                    # [Logic 3] 기본 인구통계 (Priority 1로 설정 - 타겟 다음)
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

        # [C] Semantic Conditions
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

        # [Logic 4] 차량 소유 비율 70% 이상 시 차종 분석
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
                    chart_tasks.append({"type": "filter", "field": "car_model_raw", "priority": 2})
                    used_fields.append("car_model_raw")

        # ✅ Step 1: 기본 차트 생성 (최대 3개만)
        chart_tasks.sort(key=lambda x: x['priority'])
        
        logging.info(f"   📊 [Chart Tasks] 총 {len(chart_tasks)}개 태스크 생성")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for task in chart_tasks[:4]:  # 기본 차트는 최대 4개만
                if task['type'] == 'qpoll':
                    futures.append(executor.submit(self._create_qpoll_chart, task['field']))
                else:
                    korean_name = FIELD_NAME_MAP.get(task['field'], task['field'])
                    futures.append(executor.submit(self._create_basic_chart, task['field'], korean_name, panels_data))
            
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res and res.get('chart_data'):
                        vals = list(res['chart_data'][0]['values'].values())
                        if vals and vals[0] > 95.0 and res.get('field') in search_used_fields:
                            logging.info(f"   ⏭️ [Chart Skip] '{res.get('topic')}' (95% 이상 쏠림)")
                            continue 
                        charts.append(res)
                        logging.info(f"   ✅ [Chart Added] '{res.get('topic')}' (현재: {len(charts)}개)")
                except Exception as e:
                    logging.error(f"   ❌ [Chart Error] {e}", exc_info=True)

        # ✅ Step 2: 교차 분석 무조건 생성 (최소 1개, 최대 2개)
        pivot_field = target_field if (target_field and target_field in used_fields) else (used_fields[0] if used_fields else None)
        
        if pivot_field and len(charts) < 5:
            pivot_name = QPOLL_FIELD_TO_TEXT.get(pivot_field, FIELD_NAME_MAP.get(pivot_field, pivot_field))
            
            logging.info(f"   🎯 [Crosstab] 피벗 필드: {pivot_field} ({pivot_name})")
            
            # 완전 동적 축 선택
            recommended_axes = self._select_dynamic_crosstab_axes(
                pivot_field, panels_data, search_used_fields, used_fields
            )
            
            if recommended_axes:
                crosstab_added = 0
                max_crosstab = min(2, 5 - len(charts))
                
                logging.info(f"   🎯 [Crosstab] 생성 목표: 최대 {max_crosstab}개")
                
                for ax_field, ax_name in recommended_axes:
                    if crosstab_added >= max_crosstab: break
                    
                    logging.info(f"   🔄 [Crosstab] 시도: {ax_name} x {pivot_name}")
                    
                    crosstab = self._create_crosstab_chart(
                        panels_data, ax_field, pivot_field, ax_name, pivot_name
                    )
                    
                    if crosstab and crosstab.get('chart_data'):
                        crosstab_values = crosstab['chart_data'][0]['values']
                        
                        # 검증: 최소 2개 그룹, 분포 차이 있어야 함
                        if len(crosstab_values) >= 2:
                            # 모든 그룹의 상위 답변이 동일하지 않은지 체크
                            top_answers = []
                            for group_dist in crosstab_values.values():
                                if isinstance(group_dist, dict) and group_dist:
                                    top_answer = max(group_dist.items(), key=lambda x: x[1])[0]
                                    top_answers.append(top_answer)
                            
                            # 그룹별 답변이 다르면 의미 있는 교차 분석
                            if len(set(top_answers)) > 1 or len(top_answers) == 0:  # 빈 경우도 허용
                                charts.append(crosstab)
                                crosstab_added += 1
                                logging.info(f"   📊 [Crosstab Added] {ax_name} x {pivot_name} ✅")
                            else:
                                logging.info(f"   ⏭️ [Crosstab Skip] {ax_name} x {pivot_name} (그룹별 차이 없음)")
                        else:
                            logging.info(f"   ⏭️ [Crosstab Skip] {ax_name} x {pivot_name} (그룹 부족: {len(crosstab_values)}개)")
                    else:
                        logging.info(f"   ⏭️ [Crosstab Skip] {ax_name} x {pivot_name} (데이터 없음)")
                
                if crosstab_added == 0:
                    logging.warning("   ⚠️ [Crosstab] 교차 분석 생성 실패 - 모든 축에서 데이터 없음")
            else:
                logging.info("   ⚠️ [Crosstab] 추천 축 없음")

        # ✅ Step 3: 차트 부족 시 특이점 자동 발굴
        current_crosstab_count = sum(1 for c in charts if c.get('chart_type') == 'crosstab')
        
        if len(charts) < 5:
            exclude_for_high_ratio = list(set(used_fields) | search_used_fields)
            
            if 'region_minor' in used_fields:
                exclude_for_high_ratio.append('region_major')
            
            if 'children_count' in demographic_filters or 'children_count' in search_used_fields:
                exclude_for_high_ratio.append('marital_status')

            max_high_ratio = 1 if current_crosstab_count >= 2 else (5 - len(charts))
            
            logging.info(f"   🔍 [High Ratio] 특이점 탐색 시작 (최대 {max_high_ratio}개)")
            
            high_ratio_charts = self._find_high_ratio_fields(
                panels_data, 
                exclude_fields=exclude_for_high_ratio, 
                max_charts=max_high_ratio
            )
            charts.extend(high_ratio_charts)

        logging.info(f"   🎉 [Result] 최종 차트 개수: {len(charts)}개 (교차분석: {current_crosstab_count}개)")
        
        return charts[:5], used_fields

    def _calculate_axis_importance(self, panels_data: List[Dict], target_field: str, 
                                    candidate_field: str) -> float:
        """
        타겟 필드와 후보 축 필드 간의 상관관계/중요도를 계산
        
        반환값: 0.0 ~ 1.0 (높을수록 의미 있는 교차 분석)
        """
        
        # 1. 데이터 수집
        pairs = []
        for panel in panels_data:
            target_val = panel.get(target_field)
            axis_val = panel.get(candidate_field)
            
            if target_val and axis_val:
                # birth_year는 연령대로 변환
                if candidate_field == 'birth_year':
                    axis_val = get_age_group(axis_val)
                
                # 리스트 값 처리
                if isinstance(target_val, list):
                    target_val = ', '.join(map(str, target_val))
                if isinstance(axis_val, list):
                    axis_val = ', '.join(map(str, axis_val))
                
                pairs.append((str(target_val), str(axis_val)))
        
        if len(pairs) < 10:
            return 0.0
        
        # 2. 교차표 생성
        cross_table = defaultdict(lambda: defaultdict(int))
        
        for target_val, axis_val in pairs:
            cross_table[axis_val][target_val] += 1
        
        # 3. 다양성 점수
        axis_diversity = len(cross_table)
        target_diversity = len(set(t for t, _ in pairs))
        
        if axis_diversity < 2 or axis_diversity > 20:
            return 0.0
        
        if target_diversity < 2:
            return 0.0
        
        # 4. 분포 균형도 (엔트로피)
        total_count = len(pairs)
        entropy_score = 0.0
        
        for axis_val, target_counts in cross_table.items():
            axis_ratio = sum(target_counts.values()) / total_count
            
            group_total = sum(target_counts.values())
            group_entropy = 0.0
            
            for target_val, count in target_counts.items():
                p = count / group_total
                if p > 0:
                    group_entropy -= p * np.log2(p)
            
            entropy_score += axis_ratio * group_entropy
        
        # 5. 차별성 점수 (각 축 그룹이 서로 다른 타겟 분포를 가지는가?)
        target_distributions = []
        for axis_val in cross_table.keys():
            target_counts = cross_table[axis_val]
            total = sum(target_counts.values())
            
            dist = {t: (c / total) for t, c in target_counts.items()}
            target_distributions.append(dist)
        
        if len(target_distributions) < 2:
            variance_score = 0.0
        else:
            all_targets = set()
            for dist in target_distributions:
                all_targets.update(dist.keys())
            
            variance_sum = 0.0
            for target in all_targets:
                values = [dist.get(target, 0.0) for dist in target_distributions]
                variance_sum += np.var(values)
            
            variance_score = variance_sum / len(all_targets)
        
        # 6. 최종 점수 계산
        diversity_penalty = 1.0
        if axis_diversity < 3:
            diversity_penalty = axis_diversity / 3.0
        elif axis_diversity > 10:
            diversity_penalty = 10.0 / axis_diversity
        
        max_entropy = np.log2(target_diversity)
        normalized_entropy = entropy_score / max_entropy if max_entropy > 0 else 0.0
        
        normalized_variance = min(variance_score * 10, 1.0)
        
        final_score = (
            0.3 * diversity_penalty +
            0.3 * normalized_entropy +
            0.4 * normalized_variance
        )
        
        return final_score

    def _select_dynamic_crosstab_axes(self, target_field: str, panels_data: List[Dict], 
                                       search_used_fields: Set[str], used_fields: List[str]) -> List[Tuple[str, str]]:
        """
        완전 동적으로 교차 분석 축을 선택 (하드코딩 없음)
        [수정] 검색 조건에 포함된 필드라도, 내부 분포가 다양하면(2개 그룹 이상) 축으로 허용
        """
        
        logging.info(f"   🔍 [Dynamic Crosstab] 타겟: {target_field}")
        
        # 1. 모든 가능한 축 후보 수집
        all_candidate_fields = []
        
        # Welcome 정형 데이터 필드
        for field, name in WELCOME_OBJECTIVE_FIELDS:
            # ✅ 수정됨: 이미 '차트'로 그려진(used_fields) 것만 제외하고, 
            # 검색 조건(search_used_fields)에 있더라도 후보군에 포함시킴.
            # (단일 값인지 여부는 뒤의 _calculate_axis_importance에서 axis_diversity < 2 로 걸러짐)
            if field in used_fields:
                continue
            
            if field == target_field:
                continue
            
            all_candidate_fields.append((field, name))
        
        # Q-Poll 필드도 후보에 추가 (타겟이 정형 데이터인 경우만)
        if target_field not in QPOLL_FIELD_TO_TEXT:
            for field, desc in QPOLL_FIELDS:
                # ✅ 수정됨: 여기도 동일하게 search_used_fields 조건 제거
                if field in used_fields: 
                    continue
                
                if field == target_field:
                    continue
                
                all_candidate_fields.append((field, desc))
        
        if not all_candidate_fields:
            logging.info("   ⚠️ [Dynamic Crosstab] 후보 축 없음")
            return []
        
        # 2. 각 후보 축의 중요도 점수 계산
        axis_scores = []
        
        for candidate_field, candidate_name in all_candidate_fields:
            try:
                # _calculate_axis_importance 내부에서 다양성(diversity)이 2 미만이면 0점 처리하므로
                # "20대"만 검색해서 데이터가 20대뿐이라면 자동으로 탈락됨.
                # "20대, 30대"를 검색했다면 다양성이 2가 되어 살아남음.
                score = self._calculate_axis_importance(panels_data, target_field, candidate_field)
                
                if score > 0.1:
                    axis_scores.append({
                        'field': candidate_field,
                        'name': candidate_name,
                        'score': score
                    })
                    logging.info(f"      📊 {candidate_name}: {score:.3f}")
            except Exception as e:
                logging.warning(f"      ❌ {candidate_name}: 계산 실패 ({e})")
                continue
        
        # 3. 점수 순으로 정렬
        axis_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 4. 상위 4개 선택
        selected_axes = [(a['field'], a['name']) for a in axis_scores[:4]]
        
        if axis_scores:
            top_scores_str = ', '.join([f"{a['name']}({a['score']:.2f})" for a in axis_scores[:4]])
            logging.info(f"   ✅ [Dynamic Crosstab] 선택된 축: [{top_scores_str}]")
        
        return selected_axes

    def _create_crosstab_chart(self, panels_data, field1, field2, name1, name2) -> Dict:
        """교차 분석 차트 생성 (리스트 타입 지원 수정)"""
        crosstab = {}
        
        # [수정 1] 축 데이터 수집 (리스트 평탄화 및 정제)
        vals1 = []
        for p in panels_data:
            val = p.get(field1)
            if not val: continue

            # 연령대 변환
            if field1 == 'birth_year': 
                val = get_age_group(val)
            
            # 리스트 처리 (Flatten: ['A', 'B'] -> 'A', 'B'로 분리)
            if isinstance(val, list):
                vals1.extend([clean_label(v) for v in val])
            else:
                vals1.append(clean_label(val))
        
        if not vals1: 
            return {}
        
        # 상위 5개 그룹 추출
        top_groups = [k for k, v in Counter(vals1).most_common(5)]

        for group in top_groups:
            group_panels = []
            
            # [수정 2] 그룹별 패널 필터링 (리스트 포함 여부 확인)
            for p in panels_data:
                p_val1 = p.get(field1)
                
                # 비교 값 전처리
                if field1 == 'birth_year': 
                    p_val1 = get_age_group(p_val1)
                
                is_match = False
                if isinstance(p_val1, list):
                    # 리스트인 경우: 해당 그룹 키워드가 리스트 안에 있는지 확인
                    # (데이터 정제 후 비교)
                    cleaned_list = [clean_label(x) for x in p_val1]
                    if str(group) in cleaned_list:
                        is_match = True
                else:
                    # 단일 값인 경우: 문자열 일치 확인
                    if p_val1 and str(clean_label(p_val1)) == str(group):
                        is_match = True
            
                if is_match:
                    group_panels.append(p)
            
            # 피벗 데이터 수집 (기존 로직)
            vals2 = []
            for p in group_panels:
                v = p.get(field2)
                if v:
                    if isinstance(v, list): 
                        vals2.extend(v)
                    else: 
                        vals2.append(v)
            
            if vals2:
                dist = calculate_distribution([clean_label(v) for v in vals2])
                crosstab[str(group)] = dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5])

        if not crosstab: 
            return {}

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
        검색 결과 요약
        """
        if not panels_data:
            return "데이터가 없습니다."

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

        full_context = f"""
검색 조건:
{filter_text}

발견된 특징:
{chr(10).join(stats_context)}
        """
        
        return await self.llm_service.generate_analysis_summary(query, full_context, len(panel_ids))




















