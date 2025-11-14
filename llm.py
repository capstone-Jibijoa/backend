import os
import json
import re
import hashlib
import logging
from dotenv import load_dotenv
from datetime import datetime
from functools import lru_cache
from fastapi import HTTPException
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Claude 클라이언트 초기화
try:
    CLAUDE_CLIENT = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.1)
except Exception as e:
    CLAUDE_CLIENT = None
    logging.error(f"Anthropic 클라이언트 생성 실패: {e}")

@lru_cache(maxsize=128)
def classify_query_keywords(query: str) -> dict:
    """
    쿼리를 키워드로 분류 (LLM 직접 호출)
    """
    if CLAUDE_CLIENT is None:
        raise HTTPException(status_code=500, detail="Claude 클라이언트가 초기화되지 않았습니다.")

    system_prompt = """
당신은 사용자 쿼리를 분석하고 DB 검색 키워드로 분류하는 전문가입니다.

분류 기준
1. objective (구조화 필터 / 1순위)
PostgreSQL로 필터링 가능한 명확한 카테고리.
예: 지역 (서울, 경기), 연령대 (20대), 성별 (남성), 직업군 (직장인, 학생)

2. qpoll_keywords (설문 벡터 / 2순위)
1번에 해당하지 않지만, 사용자의 경험/행동/구독/의견과 관련된 키워드.
예: "OTT", "넷플릭스", "가성비", "영상 구독"

3. subjective (주관식 벡터 / 3순위)
1, 2번에 해당하지 않는 모든 세부 키워드. (속도 튜닝됨)
예: "IT", "아이폰", "창의적인", "예술가", "환경을 생각하는"

## 판단 로직
"10개 이상 큰 그룹으로 나눌 수 있는가?"
→ YES: objective (예: 직장인, 30대, 서울)
→ NO: subjective (예: 삼성, 커피, BMW)

출력 (순수 JSON만)
{
  "welcome_keywords": {
    "objective": ["카테고리1", "카테고리2"],
    "subjective": ["특징1", "특징2"],
    "subjective_expansion": ["연관키워드1", "연관키워드2"]
  },
  "qpoll_keywords": {
    "survey_type": "주제 또는 null",
    "keywords": ["키워드1", "키워드2"]
  },
  "ranked_keywords_raw": ["키워드1", "키워드2", "키워드3"]
}

예시 
쿼리: "서울 30대 삼성폰 사용자 중 가성비를 중요하게 생각하는 마케팅 직무 100명"
{
  "welcome_keywords": {
    "objective": ["서울", "30대"],
    "subjective": ["삼성폰 사용자", "마케팅 직무"],
    "subjective_expansion": ["갤럭시", "마케터", "광고", "홍보"]
  },
  "qpoll_keywords": {
    "survey_type": "가치관/경제",
    "keywords": ["가성비", "가심비", "가격 민감도", "비용 효율"]
  },
  "ranked_keywords_raw": ["서울", "30대", "삼성폰", "가성비", "마케팅 직무"],
  "limit": 100
}

사용자 쿼리:
<query>
{{QUERY}}
</query>
"""

    logging.info(f"🔄 LLM 호출 중... (쿼리: {query})")

    limit_match = re.search(r'(\d+)\s*명', query)
    limit_value = None

    if limit_match:
        try:
            limit_value = int(limit_match.group(1))
            logging.info(f"💡 인원 수 감지: {limit_value}명")
        except ValueError:
            pass

    user_prompt = f"다음 질의를 분석하세요:\n\n{query}"

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = CLAUDE_CLIENT.invoke(messages)
    
        text_output = response.content.strip()
    
        code_block_pattern = r'^```(?:json)?\s*\n(.*?)\n```$'
        match = re.search(code_block_pattern, text_output, re.DOTALL | re.MULTILINE)
   
        if match:
            text_output = match.group(1).strip()
   
        text_output = text_output.strip('`').strip()
    
        try:
            parsed = json.loads(text_output)
            parsed['limit'] = limit_value
            parsed_result = parsed
            return parsed_result 
       
        except json.JSONDecodeError as je:
            logging.error(f"❌ JSON 파싱 실패: {je.msg}. 원본 응답: {text_output}")
            json_match = re.search(code_block_pattern, text_output, re.DOTALL)
        if json_match:
            parsed_fallback = json.loads(json_match.group(0))
            parsed_fallback['limit'] = limit_value
            parsed_result = parsed_fallback
        else:
            raise HTTPException(status_code=500, detail=f"Claude 응답 파싱 실패: {je.msg}")

        return parsed_result
       
    except Exception as e:
        logging.error(f"Claude 호출 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Claude 호출 실패: {e}") from e
 