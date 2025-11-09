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
사용자 쿼리를 분석하고 데이터베이스 검색 키워드로 분류하는 전문가입니다.

사용자 쿼리:
<query>
{{QUERY}}
</query>

## 분류 원칙

### objective (1차 필터 - 넓은 범위)
**"어떤 그룹의 사람들인가?"**
- 추상적 카테고리: 직장인, 학생, 주부, 고소득자, 저소득자
- 인구통계: 지역, 연령대, 성별, 결혼여부, 가족구성
- 일반 분류: 차량보유자, 흡연자, 음주자

→ 체크박스나 선택지로 검색 가능한 **구조화된 데이터**

### subjective (2차 벡터 - 구체적 특성)
**"그 그룹 안에서 어떤 세부 특징인가?"**
- 구체적 브랜드/제품명
- 세부 직무/전공분야
- 특정 기술/도구/스킬
- 구체적 취향/관심사

→ 자유 텍스트에서 **의미 유사도**로 검색하는 데이터

## 판단 기준

```
질문 1: "이것으로 10개 이상 큰 그룹으로 나눌 수 있나?"
YES → objective (예: 직장인, 20대, 서울, 고소득)
NO → subjective (예: IT, 삼성, 커피, BMW)

질문 2: "이것이 그룹 내 더 세밀한 구분인가?"
YES → subjective
NO → objective
```

## 출력 형식

순수 JSON만 반환하세요:

```json
{
  "welcome_keywords": {
    "objective": ["카테고리1", "카테고리2"],
    "subjective": ["세부특징1", "세부특징2"]
  },
  "qpoll_keywords": {
    "survey_type": "주제 또는 null",
    "keywords": ["키워드1", "키워드2"]
  }
}
```

## 예시

입력: "서울 30대 IT 직장인 100명"
출력:
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

입력: "부산 40대 삼성폰 쓰는 고소득자 50명"
출력:
```json
{
  "welcome_keywords": {
    "objective": ["부산", "40대", "고소득자"],
    "subjective": ["삼성폰"]
  },
  "qpoll_keywords": {
    "survey_type": null,
    "keywords": []
  }
}
```

입력: "전국 20대 개발자 커피 좋아하는 100명"
출력:
```json
{
  "welcome_keywords": {
    "objective": ["전국", "20대", "개발자"],
    "subjective": ["커피"]
  },
  "qpoll_keywords": {
    "survey_type": "음료",
    "keywords": ["카페", "스타벅스", "아메리카노"]
  }
}
```

## 중요 규칙

1. **넓은 그룹 = objective, 세부 구분 = subjective**
2. **두 카테고리 모두 있어야 2단계 검색 작동**
3. **순수 JSON만 반환 (마크다운, 설명 없음)**
4. **해당 없으면 빈 배열 []**
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