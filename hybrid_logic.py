import os
import json
import re
from dotenv import load_dotenv
from datetime import datetime
from fastapi import HTTPException
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# =======================================================
# 1. claude 모델을 모듈 수준에서 한 번만 초기화한다.
# =======================================================
try:
    CLAUDE_CLIENT = ChatAnthropic(model="claude-opus-4-1", temperature=0.0)  # 👈 0으로 변경
except Exception as e:
    CLAUDE_CLIENT = None
    print(f"Anthropic 클라이언트 생성 실패: {e}")

# 하이브리드 검색을 위한 질의 분리 함수
def split_query_for_hybrid_search(query: str) -> dict:
    """
    Claude API를 이용해 질의를 정형(Structured Filter)과 비정형(Semantic Keyword)으로 분리합니다.
    """
    if CLAUDE_CLIENT is None:
        raise HTTPException(status_code=500, detail="Anthropic Claude 클라이언트가 초기화되지 않았습니다.")

    current_year = datetime.now().year

    system_prompt = f"""당신은 사용자의 자연어 질의를 구조화된 검색 조건으로 변환하는 전문가입니다.

# 작업 목표
사용자 질의를 분석하여 다음 3가지 정보를 추출하고 JSON 형식으로 반환하세요:
1. filters: 명시적 조건들의 배열 (성별, 나이, 지역 등)
2. semantic_query: 추상적 관심사/라이프스타일 키워드
3. target_count: 요청된 결과 개수 (없으면 null)

# 출력 형식
반드시 아래 형식의 순수 JSON만 반환하세요. 설명이나 마크다운은 절대 포함하지 마세요.

{{
  "filters": [
    {{"key": "필드명", "operator": "연산자", "value": "값"}}
  ],
  "semantic_query": "검색 키워드",
  "target_count": null
}}

# 사용 가능한 필드
- gender: 성별 (예: 'M', 'F')
- birth_year: 출생연도
- region_minor / region: 거주 지역 (예: '경기', '서울', '인천')
- marital_status: 결혼 여부
- children_count: 자녀 수
- family_size: 가족 구성 인원
- education_level: 최종 학력
- job_title_raw / job_duty: 직종 및 직무
- income_personal_monthly / income_household_monthly: 개인 및 가구 월소득
- owned_electronics: 보유 가전제품 리스트
- phone_brand / phone_model_raw: 휴대폰 제조사 및 모델
- car_ownership / car_manufacturer: 자동차 보유 여부 및 제조사
- smoking_experience / drinking_experience: 흡연 및 음주 경험

# 연산자
- EQ: 일치
- BETWEEN: 범위 (value는 [최소, 최대] 배열)
- GT/LT: 초과/미만
- GTE/LTE: 이상/이하
- CONTAINS: 배열 포함 (owned_electronics 전용)

# 나이 변환 규칙 (현재 {current_year}년)
- 30대 → birth_year BETWEEN [1986, 1995]
- 35세 → birth_year EQ {current_year - 35}
- 30~40대 → birth_year BETWEEN [1976, 1995]

# 값 매핑
- 성별: 남자/남성/남 → M, 여자/여성/여 → F
- 결혼: 미혼/싱글 → 미혼, 결혼/기혼 → 기혼, 돌싱/이혼 → 이혼
- 음주: 술먹는/음주 → 경험 있음, 술안먹는/금주 → 경험 없음
- 차량: 차있음/자가용 → 보유, 차없음 → 미보유

# 예시

입력: "경기 30대 남자 중 술먹는 사람 50명"
출력:
{{
  "filters": [
    {{"key": "region", "operator": "EQ", "value": "경기"}},  
    {{"key": "birth_year", "operator": "BETWEEN", "value": [1986, 1995]}},
    {{"key": "gender", "operator": "EQ", "value": "M"}},
    {{"key": "drinking_experience", "operator": "EQ", "value": "경험 있음"}}
  ],
  "semantic_query": "",
  "target_count": 50
}}

입력: "20대 미혼 남성 럭셔리 소비 패턴"
출력:
{{
  "filters": [
    {{"key": "birth_year", "operator": "BETWEEN", "value": [1996, 2005]}},
    {{"key": "marital_status", "operator": "EQ", "value": "미혼"}},
    {{"key": "gender", "operator": "EQ", "value": "M"}}
  ],
  "semantic_query": "럭셔리 소비 패턴",
  "target_count": null
}}

# 중요 규칙
- 순수 JSON만 반환 (마크다운, 코드블록, 설명 금지)
- filters가 없으면 빈 배열 []
- semantic_query가 없으면 빈 문자열 ""
- target_count가 없으면 null"""

    user_prompt = f"다음 질의를 분석하세요:\n\n{query}"
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = CLAUDE_CLIENT.invoke(messages)
        
        # ✅ JSON 추출 로직
        text_output = response.content.strip()
        
        # 디버깅: 원본 응답 출력
        print(f"🔍 Claude 원본 응답:\n{text_output}\n{'='*50}")
        
        # 코드 블록 제거
        code_block_pattern = r'^```(?:json)?\s*\n(.*?)\n```$'
        match = re.search(code_block_pattern, text_output, re.DOTALL | re.MULTILINE)
        
        if match:
            text_output = match.group(1).strip()
            print(f"✅ 코드 블록 제거 완료")
        
        # 앞뒤 백틱 제거
        text_output = text_output.strip('`').strip()
        
        # JSON 파싱
        try:
            parsed = json.loads(text_output)
            print(f"✅ JSON 파싱 성공")
        except json.JSONDecodeError as je:
            print(f"❌ JSON 파싱 실패!")
            print(f"위치: line {je.lineno}, col {je.colno}")
            print(f"메시지: {je.msg}")
            print(f"파싱 시도 텍스트:\n{text_output}")
            
            # 혹시 JSON이 중간에 있는 경우를 위한 추가 시도
            json_match = re.search(r'\{.*\}', text_output, re.DOTALL)
            if json_match:
                print("⚠️  중간 JSON 추출 시도...")
                text_output = json_match.group(0)
                parsed = json.loads(text_output)
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Claude 응답 JSON 파싱 실패: {je.msg}"
                )
        
        # 결과 반환
        filters = parsed.get("filters", []) 
        semantic = parsed.get("semantic_query", "").strip()
        
        print(f"✅ 파싱 완료 - filters: {len(filters)}개, semantic: '{semantic}'")
        
        return {
            "structured_condition": json.dumps(filters, ensure_ascii=False),
            "semantic_condition": semantic
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ API 호출 오류: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"API 오류: {str(e)}")

if __name__ == "__main__":
    test_queries = [
        "최신 기술에 관심 많은 20대 남성",
        "경기 30대 남자 중 술먹는 사람 50명",
        "서울 기혼 여성 자동차 보유"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"테스트 쿼리: '{query}'")
        print('='*60)
        try:
            result = split_query_for_hybrid_search(query)
            print("\n✅ [성공]")
            print(f"정형 조건:\n{json.dumps(json.loads(result['structured_condition']), indent=2, ensure_ascii=False)}")
            print(f"\n비정형 검색어: '{result['semantic_condition']}'")
        except Exception as e:
            print(f"\n❌ [실패]: {e}")