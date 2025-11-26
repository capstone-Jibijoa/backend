import logging
import pandas as pd
import asyncio
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.repositories.panel_repo import PanelRepository
from app.repositories.qpoll_repo import QpollRepository
from app.services.llm_service import LLMService
from app.utils.text_utils import clean_label, calculate_distribution

from app.constants.mapping import (
    FIELD_NAME_MAP, 
    QPOLL_FIELD_TO_TEXT, 
    WELCOME_OBJECTIVE_FIELDS,
    VALUE_TRANSLATION_MAP,
    get_field_mapping,
    find_target_columns_dynamic
)

class AnalysisService:
    def __init__(self):
        self.panel_repo = PanelRepository()
        self.qpoll_repo = QpollRepository()
        self.llm_service = LLMService()

    async def get_insight_summary(self, panel_ids: List[str], question: str) -> Dict[str, Any]:
        """
        [Lite 모드] 패널 ID와 질문을 받아 통계 요약을 반환
        """
        # 1. 데이터 로드 (최대 1000명 샘플링)
        target_ids = panel_ids[:1000]
        panels_data = self.panel_repo.get_panels_by_ids(target_ids)
        
        if not panels_data:
            return {"summary": "분석할 데이터가 없습니다.", "used_fields": []}

        df = pd.DataFrame(panels_data)

        # 2. 관련 컬럼 찾기
        target_columns = find_target_columns_dynamic(question)
        
        # 3. 통계 텍스트 생성
        if not target_columns:
            stats_context = self._calculate_column_stats(df, ['gender', 'birth_year', 'region_major'])
            target_columns = ['기본 인구통계']
        else:
            stats_context = self._calculate_column_stats(df, target_columns)

        # 4. LLM 요약
        summary_text = await self.llm_service.generate_insight_summary(question, stats_context)

        return {
            "summary": summary_text,
            "used_fields": target_columns
        }

    async def analyze_search_results(self, query: str, classification: Dict, panel_ids: List[str]) -> Tuple[Dict, str]:
        """
        [Pro 모드] 검색 결과 심층 분석 및 차트 생성
        반환값: (분석결과Dict, 요약Text)
        """
        if not panel_ids:
            return {"main_summary": "검색 결과가 없습니다.", "charts": []}, "검색 결과 없음"

        # 1. 데이터 로드 (비동기 병렬 처리 가능 포인트)
        panels_data = self.panel_repo.get_panels_by_ids(panel_ids[:5000])
        if not panels_data:
            return {"main_summary": "데이터 없음", "charts": []}, "데이터 없음"

        # 2. 차트 생성 로직 수행
        charts, used_fields = await self._generate_charts(query, classification, panels_data)

        # 3. 검색 결과 텍스트 요약 생성 (별도 함수 분리)
        summary_text = await self._generate_result_overview(query, panel_ids, classification, panels_data)

        return {
            "query": query,
            "total_count": len(panels_data),
            "charts": charts
        }, summary_text

    # --- Internal Logic Helper Methods ---

    async def _generate_charts(self, query: str, classification: Dict, panels_data: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """분석 우선순위에 따라 차트 데이터를 생성합니다."""
        charts = []
        used_fields = []
        chart_tasks = []

        # (1) 타겟 필드 분석
        target_field = classification.get('target_field')
        if target_field and target_field != 'unknown':
            if target_field in QPOLL_FIELD_TO_TEXT:
                chart_tasks.append({"type": "qpoll", "field": target_field, "priority": 0})
            else:
                chart_tasks.append({"type": "filter", "field": target_field, "priority": 0})
            used_fields.append(target_field)

        # (2) 인구통계 필터 분석
        filters = classification.get('demographic_filters', {})
        for key in filters:
            if key not in used_fields and key != 'age_range': # age_range는 birth_year로 처리
                chart_tasks.append({"type": "filter", "field": key, "priority": 1})
                used_fields.append(key)
                
        if 'age_range' in filters and 'birth_year' not in used_fields:
            chart_tasks.append({"type": "filter", "field": "birth_year", "priority": 1})
            used_fields.append('birth_year')

        # (3) 병렬로 차트 데이터 생성
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for task in chart_tasks:
                field = task['field']
                if task['type'] == 'qpoll':
                    futures.append(executor.submit(self._create_qpoll_chart, field))
                else:
                    korean_name = FIELD_NAME_MAP.get(field, field)
                    futures.append(executor.submit(self._create_basic_chart, field, korean_name, panels_data))
            
            for future in as_completed(futures):
                try:
                    chart = future.result()
                    if chart and chart.get('chart_data'):
                        charts.append(chart)
                except Exception as e:
                    logging.error(f"차트 생성 실패: {e}")

        return charts, used_fields

    def _create_basic_chart(self, field_name: str, korean_name: str, panels_data: List[Dict]) -> Dict:
        """일반 필드(DB) 차트 데이터 생성"""
        # 전체 모수 DB 집계가 필요한 경우 (Repository 위임)
        if field_name in ["children_count", "birth_year"]: # 예시 조건
            distribution = self.panel_repo.get_field_distribution(field_name)
        else:
            # 메모리 내 집계
            values = []
            for item in panels_data:
                val = item.get(field_name)
                if val:
                    if isinstance(val, list):
                        values.extend([clean_label(v) for v in val])
                    else:
                        values.append(clean_label(val))
            distribution = calculate_distribution(values)

        # 상위 N개 추출 및 기타 처리 로직 (간소화)
        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:10]
        final_dist = dict(sorted_items)
        
        if not final_dist: return {}

        top_k, top_v = list(final_dist.items())[0]
        
        return {
            "topic": f"{korean_name} 분포",
            "description": f"가장 많은 응답은 '{top_k}'({top_v}%) 입니다.",
            "ratio": f"{top_v}%",
            "chart_data": [{"label": korean_name, "values": final_dist}],
            "field": field_name
        }

    def _create_qpoll_chart(self, field_name: str) -> Dict:
        """Q-Poll(Qdrant) 차트 데이터 생성"""
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
        """Pandas DataFrame 통계 텍스트 변환"""
        report = []
        for col in columns:
            if col not in df.columns: continue
            
            try:
                valid = df[col].dropna()
                if valid.empty: continue
                
                # 리스트 타입 처리
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
        """검색 결과에 대한 종합 텍스트 요약 생성"""
        df = pd.DataFrame(panels_data[:1000]) # 샘플링
        
        stats_context = []
        target_field = classification.get('target_field')
        
        # 1. 타겟 필드 통계
        if target_field and target_field in df.columns:
            counts = df[target_field].value_counts(normalize=True).head(3)
            items = [f"{k}({v*100:.1f}%)" for k, v in counts.items()]
            kname = FIELD_NAME_MAP.get(target_field, target_field)
            stats_context.append(f"[{kname}]: {', '.join(items)}")

        # 2. 주요 인구통계
        for field in ['gender', 'region_major']:
            if field in df.columns:
                top = df[field].value_counts(normalize=True).head(1)
                if not top.empty:
                    k, v = top.index[0], top.values[0]
                    stats_context.append(f"[{FIELD_NAME_MAP.get(field)}]: {k}({v*100:.1f}%)")

        full_stats = "\n".join(stats_context)
        return await self.llm_service.generate_analysis_summary(query, full_stats, len(panel_ids))