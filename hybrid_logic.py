import os
import json
from dotenv import load_dotenv
from datetime import datetime # 👈 현재 연도를 가져오기 위해 추가
from fastapi import HTTPException
from langchain_anthropic import ChatAnthropic # 👈 LangChain의 Anthropic 래퍼 사용
from langchain_core.messages import SystemMessage, HumanMessage # 👈 메시지 객체 임포트

load_dotenv()

# =======================================================
# 1. claude 모델을 모듈 수준에서 한 번만 초기화한다.
# =======================================================
try:
    # 💡 최적화: 질의 분리는 비교적 간단한 작업이므로 더 빠르고 저렴한 Sonnet 모델로 변경합니다.
    # model="claude-4-sonnet-20250514"
    CLAUDE_CLIENT = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.1)
except Exception as e:
    CLAUDE_CLIENT = None
    print(f"Anthropic 클라이언트 생성 실패: {e}") # 모델과 동일한 패턴으로 유지

# 하이브리드 검색을 위한 질의 분리 함수
def split_query_for_hybrid_search(query: str) -> dict:
    """
    Claude API를 이용해 질의를 정형(Structured Filter)과 비정형(Semantic Keyword)으로 분리합니다.
    """
    if CLAUDE_CLIENT is None:
        raise HTTPException(status_code=500, detail="Anthropic Claude 클라이언트가 초기화되지 않았습니다.")

    current_year = datetime.now().year # 👈 동적으로 현재 연도 가져오기

    # =======================================================
    # 프롬프트 디테일 강화 (오차 최소화) 적용 부분
    # =======================================================
    system_prompt = """
    당신은 사용자 질의를 하이브리드 검색에 사용될 JSON 객체로 변환하는 최고 전문가입니다.
**응답은 오직 하나의 완벽한 JSON 객체** 형태로만 반환해야 하며, 어떤 추가 설명이나 문장도 포함해서는 안 됩니다.
**현재 데이터 샘플은 최대 150개**이므로, target_count는 150을 초과하지 않도록 엄격하게 제한해야 합니다.

[핵심 규칙]
1. **정형 조건 (filters)**: '지역', '성별', '나이', '소득', '직무' 등 명확한 속성 필터는 'filters' 배열에 객체 형태로 변환하세요.
   - **컬럼 목록 확정**: 다음 확정된 컬럼 목록만 사용하세요. 질문에 직접 관련된 정보만 필터링하고, 나머지 컬럼은 무시하세요:
     **region_major, gender, birth_year, marital_status, education_level, job_duty, income_personal_monthly, car_ownership, drinking_experience, smoking_experience**
   - **연산자**: EQ(동일), BETWEEN(범위), GT(초과), LT(미만)만 사용하세요.
   - **값 표준화**: 
     a. **나이 변환**: 나이(예: 30~40대)는 **현재 연도({current_year}년)**를 기준으로 출생 연도(birth_year)의 **BETWEEN** 범위(예: [{current_year}-49, {current_year}-30])로 변환하세요.
     b. **성별 변환**: '남자', '여자'만 사용하세요.
     c. **누락 처리**: 정형 조건에 해당하는 내용이 없으면 'filters' 배열은 빈 리스트(`[]`)로 반환하세요.
     
2. **비정형 조건 (semantic_query)**: 의미론적 검색어는 'semantic_query' 필드에 담으세요.
   - **[매우 중요] 핵심 키워드 추출**: 'semantic_query'는 **KURE 임베딩에 바로 사용할 핵심 명사/구문**만 남기고 불필요한 관형어나 문장 성분(예: '추천해줘', '찾아줘' 등)은 **모두 제거**해야 합니다.
   
3. **목표 수량 (target_count)**: 쿼리에 'n명', 'top k'와 같은 목표 수량이 있으면 숫자로 반환하세요. 없으면 `null`로 반환하세요. **150을 초과하지 않도록** 하세요.

[출력 스키마 예시]
// 입력 쿼리 예시: '경기 30~40대 남자 술을 먹은 사람 50명'
{{
  "target_count": 50,
  "filters": [
    {{ "key": "region_major", "operator": "EQ", "value": "경기" }},
    {{ "key": "birth_year", "operator": "BETWEEN", "value": [1985, 1995] }}, 
    {{ "key": "gender", "operator": "EQ", "value": "남자" }},
    {{ "key": "drinking_experience", "operator": "EQ", "value": "경험 있음" }}
  ],
  "semantic_query": "" // 이 쿼리에는 비정형 조건이 없으므로 빈 문자열 반환
}}
""".format(current_year=current_year) # 👈 프롬프트에 현재 연도 포맷팅
    
    user_prompt = f"""
    다음 쿼리를 분석하여 JSON 형식으로 반환하세요.
    쿼리: '{query}'
    """
    
    try:
      # 🌟 Anthropic SDK messages.create 호출
      # 💡 최종 수정: invoke에 문자열 대신, SystemMessage와 HumanMessage 객체 리스트를 전달합니다.
      # 이 방식이 모델의 역할을 명확히 하여 안정성을 높입니다.
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = CLAUDE_CLIENT.invoke(messages)
        
        text_output = response.content.strip()
        parsed = json.loads(text_output)
        
        # 반환 구조는 이전 단계에서 정의된 대로 유지 (JSON 문자열로 반환)
        filters = parsed.get("filters", []) 
        semantic = parsed.get("semantic_query", "").strip()
        
        return {
            "structured_condition": json.dumps(filters), # DB 로직에서 파싱할 JSON 필터 배열
            "semantic_condition": semantic # KURE 임베딩에 사용할 핵심 키워드
        }

    except Exception as e: # LangChain 래퍼는 일반 Exception으로 처리
        print("Anthropic API 호출 에러:", e)
        raise HTTPException(status_code=500, detail=f"Anthropic API 호출 에러: {e}")
    
if __name__ == "__main__":
    # 이 파일을 직접 실행할 때만 아래 코드가 동작합니다.
    # 테스트하고 싶은 쿼리를 여기에 입력하세요.
    test_query = "최신 기술에 관심 많은 20대 남성"
    print(f"테스트 쿼리: '{test_query}'")
    print("-" * 50)
    try:
        # 함수를 직접 호출하여 결과를 확인합니다.
        result = split_query_for_hybrid_search(test_query)
        print("\n[질의 분리 성공]")
        # 보기 좋게 출력
        print("정형 조건 (structured_condition):")
        print(json.dumps(json.loads(result["structured_condition"]), indent=2, ensure_ascii=False))
        print("\n비정형 검색어 (semantic_condition):")
        print(f"'{result['semantic_condition']}'")
    except Exception as e:
        print(f"\n[질의 분리 실패]: {e}")
    
    