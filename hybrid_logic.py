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

**ranked_keywords (우선순위 키워드)** ✅ 신규 추가
- 주요 검색 조건 3개를 우선순위순으로 나열
- 각 키워드에 대응하는 DB 필드명 포함
- 프론트엔드 테이블 컬럼 표시 순서 결정용

## 필드 매핑 규칙
- 서울/경기/부산 등 → region_major (거주 지역)
- 안양시/시흥시/금정구/완주군 등 → region_minor (시/구/군 등 세부 거주 지역)
- 20대/30대/40대 등 → birth_year (연령대)
- 남자/여자/남성/여성 → gender (성별)
- 직장인/학생 등 → job_title_raw (직업)
- 고소득/저소득 → income_personal_monthly (소득)
- 미혼/기혼 → marital_status (결혼 여부)
- 흡연/비흡연 → smoking_experience (흡연 경험)
- 음주/금주 → drinking_experience (음주 경험)
- 차량보유/차없음 → car_ownership (차량 보유)
- 직장인/학생/주부 등 구체적인 직업 분류 → job_title_raw
- IT/마케팅 등 구체적 직무 → job_duty_raw (직무)
- 삼성/갤럭시/아이폰/애플 등 휴대전화 브랜드 → phone_brand_raw
- 아이폰 15/갤럭시 S23 등 휴대전화 모델 → phone_model_raw
- 현대차/기아/BMW/테슬라 등 차량 제조사 → car_manufacturer_raw
- 소나타/K5/Model Y 등 차량 모델명 → car_model_raw
- 말보로/에쎄/담배/전자담배 등 흡연 브랜드/종류 → smoking_brand_etc_raw
- 기타 담배 종류/흡연 세부 사항 → smoking_brand_other_details_raw
- 주류 종류/음주 세부 사항 → drinking_experience_other_details_raw
- 기타 브랜드/제품명 → 해당 필드 또는 null

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
  "ranked_keywords": [
    {"keyword": "키워드1", "field": "필드명", "description": "한글 설명", "priority": 1},
    {"keyword": "키워드2", "field": "필드명", "description": "한글 설명", "priority": 2},
    {"keyword": "키워드3", "field": "필드명", "description": "한글 설명", "priority": 3}
  ]
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
  },
  "ranked_keywords": [
    {"keyword": "서울", "field": "region_major", "description": "거주 지역", "priority": 1},
    {"keyword": "30대", "field": "birth_year", "description": "연령대", "priority": 2},
    {"keyword": "IT", "field": "job_duty_raw", "description": "직무", "priority": 3}
  ]
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
  "ranked_keywords": [
    {"keyword": "부산", "field": "region_major", "description": "거주 지역", "priority": 1},
    {"keyword": "40대", "field": "birth_year", "description": "연령대", "priority": 2},
    {"keyword": "고소득", "field": "income_personal_monthly", "description": "소득", "priority": 3}
  ]
}
```

쿼리: "서울 OTT 사용하는 40~50대 남성" ✅ 연령대 통합 예시
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
  "ranked_keywords": [
    {"keyword": "서울", "field": "region_major", "description": "거주 지역", "priority": 1},
    {"keyword": "40~50대", "field": "birth_year", "description": "연령대", "priority": 2},
    {"keyword": "남성", "field": "gender", "description": "성별", "priority": 3}
  ]
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