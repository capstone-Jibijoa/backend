import json
import logging
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.llm_client import CLAUDE_CLIENT 

def extract_relevant_columns_via_llm(question: str, all_columns_info: str) -> List[str]:
    """
    질문을 분석하여 통계 분석에 필요한 DB 컬럼명들을 추출합니다.
    """
    if not CLAUDE_CLIENT: 
        logging.error("Claude Client is not initialized.")
        return []

    system_prompt = f"""
    You are a Data Analyst. Select the most relevant database columns from the [Column List] to answer the user's [Question].
    
    [Column List]
    {all_columns_info}
    
    [Rules]
    1. Return ONLY a JSON object with a key "columns" containing a list of strings.
    2. If no column is relevant, return "columns": [].
    3. Select strictly from the provided list.
    """
    
    user_prompt = f"Question: {question}"

    try:
        response = CLAUDE_CLIENT.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # JSON 파싱 
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
            
        data = json.loads(content)
        return data.get("columns", [])
    except Exception as e:
        logging.error(f"컬럼 추출 실패: {e}")
        return []

def generate_demographic_summary(query: str, stats_text: str, total_count: int) -> str:
    """
    [Pro 모드] 통계 데이터를 바탕으로 '심층 인사이트'가 담긴 요약 문장을 생성합니다.
    """
    if not CLAUDE_CLIENT: return ""

    system_prompt = """
    당신은 데이터 인사이트 전문가입니다. 
    제공된 [데이터 통계]를 근거로 [사용자 질문]에 대한 핵심 요약을 작성하세요.
    
    [작성 원칙]
    1. 막연한 표현 대신 **제공된 수치(명, %)**를 반드시 인용하여 근거를 제시하세요. 
    2. 가장 두드러진 특징(최댓값, 과반수 등)을 강조하세요.
    3. 질문과 관련 없는 통계는 언급하지 마세요.
    4. "~하는 것이 특징입니다"와 같은 분석적인 어조를 사용하세요.
    6. 마케팅을 위한 인사이트를 추가로 제안하세요.
    5. 한국어로 간결하게 답변하세요 (5문장 내외).
    """
    
    user_prompt = f"""
    [사용자 질문]
    {query}

    [분석 대상]
    총 {total_count}명의 패널

    [통계 데이터]
    {stats_text}

    위 데이터를 바탕으로 핵심 분석 요약 마케팅 인사이트를 작성해주세요.
    """
    
    try:
        response = CLAUDE_CLIENT.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return response.content.strip()
    except Exception as e:
        logging.error(f"통계 요약 생성 실패: {e}")
        return "요약 생성 중 오류가 발생했습니다."