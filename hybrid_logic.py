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

    system_prompt = """
 사용자 쿼리를 분석하고 데이터베이스 테이블 검색 키워드로 분류하는 전문가입니다. 사용자 쿼리가 주어지면 두 개의 서로 다른 데이터베이스 테이블에 대한 관련 키워드를 추출해야 합니다.

분석할 사용자 쿼리는 다음과 같습니다.
<query>
{{QUERY}}
</query>

이 쿼리를 분석하고 키워드를 두 가지 범주로 분류하는 것이 과제입니다.

## 분류 범주

**1. 시작 테이블 키워드:**
- `objective`: 명확하고 사실적인 데이터(인구 통계, 위치, 연령대, 구체적인 측정 기준)
- `subjective`: 추상적/주관적 표현(라이프스타일 선호도, 관심사, 행동, 성격 특성)

**2. QPoll 테이블 키워드:**
- `survey_type`: 설문조사 또는 연구 주제 유형
- `keywords`: 설문조사 데이터에서 검색할 특정 용어

## 출력 형식

다음 형식의 순수 JSON만 반환해야 합니다.

```json
{
"welcome_keywords": {
"objective": ["keyword1", "keyword2"],
"subjective": ["keyword3", "keyword4"]
},
"qpoll_keywords": {
"survey_type": "survey type or null",
"keywords": ["keyword5", "keyword6"]
}
}
```

## 분류 규칙

1. **객관적 키워드**: 인구 통계, 위치, 연령대, 성별, 직업, 소득 수준, 교육 수준 - 측정 가능하거나 범주화된 모든 것
2. **주관적 키워드**: 관심사, 취미, 라이프스타일 선택, 선호도, 행동, 성격 특성 등 해석 가능한 모든 것
3. **QPoll 키워드**: 쿼리가 설문조사, 여론조사, 의견 또는 설문조사 응답에서 발견될 수 있는 특정 주제를 언급하는 경우
4. **설문조사 유형**: 쿼리가 설문조사 데이터와 관련된 경우 주요 주제 또는 테마를 추출합니다.
5. **키워드는 간결해야 합니다**: 각 1~3단어
6. **해당되지 않는 범주는 빈 배열을 사용합니다**: `[]`
7. **해당되지 않는 경우 survey_type에 null을 사용합니다**

## 예시

입력: "부산, 경남 40대 남녀 해외여행 계획 중인 20명"
출력:
```json
{
  "welcome_keywords": {
    "objective": ["부산", "경남", "40대", "남녀"],
    "주관적": ["해외여행 계획", "여행"]
  },
  "qpoll_keywords": {
    "survey_type": "여행",
    "keywords": ["일본", "베트남", "유럽", "항공권", "숙소"]
  }
}
````

입력: "전국 20~30대 직장인 커피 선호하는 100명"
산출:
``json
{
  "welcome_keywords": {
    "objective": ["전국", "20~30대", "직장인"],
    "주관적": ["커피 선호", "카페 이용"]
  },
  "qpoll_keywords": {
    "survey_type": "커피",
    "keywords": ["스타벅스", "아메리카노", "카페", "프랜차이즈"]
  }
}
````

입력: "경기 20대 대학생 대중교통 이용하는 80명"
산출:
``json
{
  "welcome_keywords": {
    "objective": ["경기", "20대", "대학생"],
    "주관적": ["대중교통 이용", "교통"]
  },
  "qpoll_keywords": {
    "survey_type": null,
    "keywords": []
  }
}
```

## 중요 요구 사항

- 순수 JSON만 반환합니다. 마크다운 형식, 설명, 추가 텍스트는 허용되지 않습니다.
- 키워드는 관련성이 높고 간결해야 합니다.
- welcome_keywords와 qpoll_keywords 모두 서로 다르게 검색되는 중복되는 개념을 포함할 수 있습니다.
- 카테고리에 적용되는 키워드가 없는 경우, 빈 배열 `[]` 또는 `null`을 사용합니다.
- 유효한 JSON 구문을 사용합니다.
"""

    user_prompt = f"다음 질의를 분석하세요:\n\n{query}"
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = CLAUDE_CLIENT.invoke(messages)
        
        # JSON 추출
        text_output = response.content.strip()
        print(f"🔍 Claude 원본 응답:\n{text_output}\n{'='*50}")
        
        # 코드 블록 제거
        code_block_pattern = r'^```(?:json)?\s*\n(.*?)\n```$'
        match = re.search(code_block_pattern, text_output, re.DOTALL | re.MULTILINE)
        
        if match:
            text_output = match.group(1).strip()
        
        text_output = text_output.strip('`').strip()
        
        # JSON 파싱
        try:
            parsed = json.loads(text_output)
            print(f"✅ 키워드 분류 성공")
            print(f"Welcome 객관식: {parsed.get('welcome_keywords', {}).get('objective', [])}")
            print(f"Welcome 주관식: {parsed.get('welcome_keywords', {}).get('subjective', [])}")
            print(f"QPoll: {parsed.get('qpoll_keywords', {})}")
            return parsed
            
        except json.JSONDecodeError as je:
            print(f"❌ JSON 파싱 실패: {je}")
            # 중간 JSON 추출 시도
            json_match = re.search(r'\{.*\}', text_output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
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