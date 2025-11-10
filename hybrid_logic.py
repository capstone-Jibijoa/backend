import os
import json
import re
from dotenv import load_dotenv
from datetime import datetime
from fastapi import HTTPException
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Claude 모델 초기화
try:
    CLAUDE_CLIENT = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.1)
except Exception as e:
    CLAUDE_CLIENT = None
    print(f"Anthropic 클라이언트 생성 실패: {e}")

def classify_query_keywords(query: str) -> dict:
   
    if CLAUDE_CLIENT is None:
        raise HTTPException(status_code=500, detail="Claude 클라이언트가 초기화되지 않았습니다.")

    system_prompt = system_prompt ="""
사용자 쿼리를 분석하고 데이터베이스 검색 키워드로 분류하는 전문가입니다.

## 분류 기준

**objective (구조화 필터)**: 넓은 그룹 분류 - 체크박스로 검색 가능
- 인구통계: 지역, 연령대, 성별, 직업군
- 경제: 소득수준, 차량보유
- 라이프스타일: 흡연/음주 여부

**subjective (벡터 검색)**: 구체적 특성 - 의미 유사도 검색
- 브랜드/제품명, 세부 직무/전공, 기술/도구, 구체적 취향

**qpoll_keywords (설문 응답 검색)**: 3단계 구조
1. 일반 카테고리 (필수)
2. 대표 브랜드/제품
3. 관련 행동/경험

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
  }
}
```

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
  }
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
  }
}
```

쿼리: "서울 경기 OTT 이용 젊은층 30명"
```json
{
  "welcome_keywords": {
    "objective": ["서울", "경기", "젊은층"],
    "subjective": ["OTT"]
  },
  "qpoll_keywords": {
    "survey_type": "엔터테인먼트",
    "keywords": ["OTT", "스트리밍", "영상", "넷플릭스", "티빙", "구독"]
  }
}
```

사용자 쿼리:
<query>
{{QUERY}}
</query>
"""


    # 인원 수(limit) 추출 로직 
    limit_match = re.search(r'(\d+)\s*명', query)
    limit_value = None
    
    if limit_match:
        try:
            limit_value = int(limit_match.group(1))
            print(f"💡 인원 수 감지: {limit_value}명")
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
        print(f"🔍 Claude 원본 응답:\n{text_output}\n{'='*50}")
        
        code_block_pattern = r'^```(?:json)?\s*\n(.*?)\n```$'
        match = re.search(code_block_pattern, text_output, re.DOTALL | re.MULTILINE)
       
        if match:
            text_output = match.group(1).strip()
       
        text_output = text_output.strip('`').strip()
        
        try:
            parsed = json.loads(text_output)

            # 추출한 limit 값을 최종 JSON에 추가
            parsed['limit'] = limit_value
            return parsed
           
        except json.JSONDecodeError as je:
            print(f"❌ JSON 파싱 실패: {je}")
            json_match = re.search(r'\{.*\}', text_output, re.DOTALL)
            if json_match:
                parsed_fallback = json.loads(json_match.group(0))
                parsed_fallback['limit'] = limit_value
                return parsed_fallback
            raise HTTPException(status_code=500, detail=f"Claude 응답 파싱 실패: {je.msg}")
           
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ API 호출 오류: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"API 오류: {str(e)}")

# 테스트 코드
if __name__ == "__main__":
    test_queries = [
        "서울 20대 남자 100명",
        "경기 30~40대 남자 술을 먹은 사람 50명",
        "서울, 경기 OTT 이용하는 젊은층 30명"
    ]
   
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"테스트 쿼리: '{query}'")
        print('='*60)
        try:
            result = classify_query_keywords(query)
            print("\n✅ [성공]")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"\n❌ [실패]: {e}")
