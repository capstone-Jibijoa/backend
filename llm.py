import os
import json
import re
import hashlib
import logging
from dotenv import load_dotenv
from datetime import datetime
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


def classify_query_keywords(query: str) -> dict:
    """
    쿼리를 키워드로 분류 (LLM 직접 호출)
    """
    if CLAUDE_CLIENT is None:
        raise HTTPException(status_code=500, detail="Claude 클라이언트가 초기화되지 않았습니다.")

    system_prompt = """
사용자 쿼리를 분석하여 핵심 키워드를 추출하고 분류하는 전문가입니다.

## 분류 기준
**objective (구조화 필터)**: 넓은 그룹 분류 (예: 지역, 연령대, 성별, 직업군)
**subjective (벡터 검색)**: 구체적 특성 (예: 브랜드명, 세부 직무, 기술, 구체적 취향)
**qpoll_keywords (설문 응답 검색)**: 3단계 구조 (카테고리, 브랜드, 행동)

## 판단 로직
"10개 이상 큰 그룹으로 나눌 수 있는가?"
→ YES: objective (예: 직장인, 30대, 서울)
→ NO: subjective (예: 삼성, 커피, BMW)

## 출력 (순수 JSON만)
```json
{
 "welcome_keywords": {
  "objective": ["카테고리1", "카테고리2"],
  "subjective": ["특징1", "특징2"]
 },
 "qpoll_keywords": {
  "survey_type": "주제 또는 null",
  "keywords": ["키워드1", "키워드2"]
 },
 "ranked_keywords_raw": ["키워드1", "키워드2", "키워드3"]
}

## 예시

쿼리: "서울 30대 IT 직장인 100명"
```json
{
  "welcome_keywords": {
    "objective": ["서울", "30대", "직장인"],
    "subjective": ["IT"]
  },
  "qpoll_keywords": {
    "survey_type": null,
    "keywords": []
  },
  "ranked_keywords_raw": ["서울", "30대", "IT"]
}
```

쿼리: "부산 40대 삼성폰 쓰는 고소득자 50명"
```json
{
  "welcome_keywords": {
    "objective": ["부산", "40대", "고소득자"],
    "subjective": ["삼성폰"]
  },
  "qpoll_keywords": {
    "survey_type": "전자기기",
    "keywords": ["스마트폰", "핸드폰", "삼성", "갤럭시", "사용"]
  },
  "ranked_keywords_raw": ["부산", "40대", "고소득자"]
}
```

쿼리: "서울 OTT 사용하는 40~50대 남성" 
```json
{
  "welcome_keywords": {
    "objective": ["서울", "40~50대", "남성"],
    "subjective": ["OTT"]
  },
  "qpoll_keywords": {
    "survey_type": "엔터테인먼트",
    "keywords": ["OTT", "스트리밍", "영상", "넷플릭스", "티빙", "구독"]
  },
  "ranked_keywords_raw": ["서울", "40~50대", "남성"]
}
```
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
 