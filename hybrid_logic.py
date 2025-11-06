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
    CLAUDE_CLIENT = ChatAnthropic(model="claude-opus-4-1", temperature=0.1)
except Exception as e:
    CLAUDE_CLIENT = None
    print(f"Anthropic 클라이언트 생성 실패: {e}")

def classify_query_keywords(query: str) -> dict:
    """
    LLM을 사용하여 질의를 분석하고 Welcome/QPoll 키워드로 분류합니다.
    
    반환 형식:
    {
        "welcome_keywords": {
            "objective": ["키워드1", "키워드2"],  # 객관식 (PostgreSQL)
            "subjective": ["키워드3"]              # 주관식 (Qdrant)
        },
        "qpoll_keywords": {
            "survey_type": "설문종류",  # 예: "lifestyle", "preference" 등
            "keywords": ["키워드4", "키워드5"]
        }
    }
    """
    if CLAUDE_CLIENT is None:
        raise HTTPException(status_code=500, detail="Claude 클라이언트가 초기화되지 않았습니다.")

    system_prompt = """당신은 사용자 질의를 분석하여 데이터베이스 테이블별 검색 키워드로 분류하는 전문가입니다.

# 작업 목표
사용자 질의를 분석하여 다음과 같이 분류하세요:

1. **Welcome 테이블 관련 키워드**
   - objective: 명확한 속성 기반 조건 (성별, 나이, 지역, 소득 등)
   - subjective: 추상적/주관적 표현 (라이프스타일, 관심사, 성향 등)

2. **QPoll 테이블 관련 키워드**
   - survey_type: 설문 유형 분류
   - keywords: 해당 설문에서 검색할 키워드

# Welcome 테이블 필드 (objective용)
- 인구통계: gender(성별), birth_year(출생연도), region(지역), marital_status(결혼상태)
- 경제: income_personal_monthly(개인소득), income_household_monthly(가구소득), job_title_raw(직업)
- 가족: children_count(자녀수), family_size(가족구성원수)
- 소유물: owned_electronics(가전제품), phone_brand(휴대폰), car_ownership(자동차)
- 생활습관: smoking_experience(흡연), drinking_experience(음주)

# Welcome 테이블 - 주관식 키워드 (subjective용)
- 라이프스타일, 취미, 관심사, 가치관, 소비패턴, 성향 등 추상적 표현

# QPoll 설문 유형
- lifestyle: 라이프스타일/일상생활 관련
- consumption: 소비행태/구매패턴
- media: 미디어 이용/콘텐츠 선호
- health: 건강/운동/식습관
- technology: 기술/디지털 기기 사용
- travel: 여행/레저 활동
- finance: 금융/투자 관련

# 출력 형식
반드시 순수 JSON만 반환하세요:

{
  "welcome_keywords": {
    "objective": ["키워드1", "키워드2"],
    "subjective": ["키워드3"]
  },
  "qpoll_keywords": {
    "survey_type": "설문종류 또는 null",
    "keywords": ["키워드4"]
  }
}

# 분류 규칙
1. 명확한 수치/범주형 조건 → welcome_keywords.objective
2. 추상적/감성적 표현 → welcome_keywords.subjective
3. 설문 응답 관련 → qpoll_keywords
4. 매칭되지 않으면 빈 배열 또는 null

# 예시

입력: "경기 30대 남자 중 럭셔리 소비에 관심있는 사람"
출력:
{
  "welcome_keywords": {
    "objective": ["경기", "30대", "남자"],
    "subjective": ["럭셔리", "소비"]
  },
  "qpoll_keywords": {
    "survey_type": "consumption",
    "keywords": ["럭셔리", "고가", "프리미엄"]
  }
}

입력: "서울 미혼 여성 중 요가 하는 사람"
출력:
{
  "welcome_keywords": {
    "objective": ["서울", "미혼", "여성"],
    "subjective": ["운동", "건강"]
  },
  "qpoll_keywords": {
    "survey_type": "health",
    "keywords": ["요가", "운동", "헬스"]
  }
}

입력: "20대 남성 게임 유저"
출력:
{
  "welcome_keywords": {
    "objective": ["20대", "남성"],
    "subjective": ["게임"]
  },
  "qpoll_keywords": {
    "survey_type": "media",
    "keywords": ["게임", "게이머", "플레이"]
  }
}

# 중요 규칙
- 순수 JSON만 반환 (마크다운, 설명 금지)
- 키워드는 간결하게 (1-3단어)
- 중복 키워드 허용 (테이블마다 다른 방식으로 검색)
- 매칭 안 되면 빈 배열/null"""

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
        "경기 30대 남자 중 럭셔리 소비에 관심있는 사람",
        "서울 미혼 여성 중 요가 하는 사람",
        "20대 남성 게임 유저"
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