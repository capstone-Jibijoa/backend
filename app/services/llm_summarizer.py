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
    당신은 예리한 '마케팅 전략가'입니다. [통계 데이터]를 바탕으로 타겟(Persona)의 핵심 욕망과 행동 패턴을 분석하세요.

    [작성 원칙]
    1. **압축적 통찰**: 단순 수치 나열을 금지합니다. 수치는 괄호 안에 근거로만 표기하세요. (예: 30대(88%))
    2. **인과 관계 추론**: "왜" 이런 결과가 나왔는지 라이프스타일(직업, 소득, 거주지)과 연결하여 해석하세요.
    3. **임팩트 있는 결론**: 마케팅적으로 활용 가능한 '기회 요인'이나 '키워드'를 제시하며 마무리하세요.
    4. **분량 제한**: 반드시 **5문장 이내**으로 짧고 굵게 작성하세요. (말줄임표 금지)
    """
    
    user_prompt = f"""
    [사용자 질문]
    {query}

    [분석 대상]
    총 {total_count}명의 패널

    [통계 데이터]
    {stats_text}

    위 데이터를 바탕으로 핵심 마케팅 인사이트를 요약해 주세요.
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