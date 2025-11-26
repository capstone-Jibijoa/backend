import logging
from app.services.llm_summarizer import generate_stats_summary, generate_demographic_summary

class LLMService:
    """
    LLM 관련 로직을 담당하는 서비스
    """
    
    async def generate_insight_summary(self, question: str, stats_context: str) -> str:
        """Lite 모드용 요약 생성"""
        try:
            return generate_stats_summary(question, stats_context)
        except Exception as e:
            logging.error(f"LLM 요약 생성 실패: {e}")
            return "요약 생성 중 오류가 발생했습니다."

    async def generate_analysis_summary(self, query: str, stats_text: str, total_count: int) -> str:
        """Pro 모드용 분석 요약 생성"""
        try:
            return generate_demographic_summary(query, stats_text, total_count)
        except Exception as e:
            logging.error(f"LLM 분석 요약 실패: {e}")
            return "데이터 분석 중 오류가 발생했습니다."