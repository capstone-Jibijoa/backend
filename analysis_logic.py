import json
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, conlist 
from typing import List, Dict, Any

# 1. JSON 출력 구조 정의 (Pydantic Schema)
# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 기존 코드와 동일
# 시각화 데이터 항목의 Pydantic 모델
class ChartDataEntry(BaseModel):
    label: str = Field(description="차트 데이터의 레이블 (예: '결혼 여부')")
    values: dict = Field(description="각 카테고리별 값의 딕셔너리 (예: {'기혼': 100, '미혼': 0})")

# 개별 분석 차트 모델
class AnalysisChart(BaseModel):
    topic: str = Field(description="분석 주제 (한글 문장)")
    description: str = Field(description="차트 분석 요약 설명 (1~2줄)")
    ratio: str = Field(description="주요 비율 (예: '100.0%')")
    chart_data: conlist(ChartDataEntry, min_length=1, max_length=1)

# 최종 JSON 출력 구조 모델
class FinalAnalysisOutput(BaseModel):
    main_summary: str = Field(description="검색 결과에 대한 포괄적인 요약 (2~3줄)")
    query_focused_chart: AnalysisChart = Field(description="검색 질의 특징 분석 차트")
    related_topic_chart: AnalysisChart = Field(description="검색 결과 연관 주제 분석 차트")
    high_ratio_charts: conlist(AnalysisChart, min_length=3, max_length=3) # 차트 3개

# 2. 초기화 및 설정 (Claude API 사용)

# LangChain LLM 객체와 JSON 파서 초기화
try:
    # 환경 변수 ANTHROPIC_API_KEY를 자동으로 사용합니다. 
    # 모델명은 Claude 3 Sonnet의 API 식별자인 "claude-opus-4-1"를 사용합니다.
    llm = ChatAnthropic(
        model="claude-opus-4-1",
        temperature=0.4,
        # api_key=os.environ.get("ANTHROPIC_API_KEY") # 환경 변수가 아닌 경우 직접 전달 가능
    )
    
    # JSON 출력 파서 초기화 (Pydantic 모델 기반)
    parser = JsonOutputParser(pydantic_object=FinalAnalysisOutput)

except Exception as e:
    # Anthropic API 키 누락 등으로 초기화 실패 시 처리
    print("LangChain Claude API LLM 초기화 실패: ANTHROPIC_API_KEY를 확인하세요.", e)
    llm = None
    parser = None

PROMPT_TEMPLATE = """
당신은 데이터 분석가이자 통계 시각화 전문가입니다.
다음은 사용자의 자연어 질의와 그에 해당하는 설문 데이터입니다.

[사용자 질의]
"{user_query}"

[데이터 샘플] (최대 150명)
아래 JSON 배열은 특정 주제에 맞는 응답자들의 설문 결과입니다.
각 항목은 개인 응답자 하나를 의미하며, 필드는 다음과 같습니다:
- gender: 성별 (예: 'M', 'F')
- birth_year: 출생연도
- region_major / region_minor: 거주 지역 (예: '경기', '화성시')
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

분석 목표
아래의 데이터를 분석하여 총 5개의 차트 데이터를 포함하는 JSON 형식으로 출력하세요.

1. query_focused_chart (검색 질의 특징 분석 - 차트 1개)
- 목적: 사용자의 질의에 포함된 가장 대표적인 인구통계학적 속성 1개 (예: 성별, 연령대, 결혼 여부 등)의 분포를 분석하고 시각화용 데이터(`chart_data`)를 생성하세요.

2. related_topic_chart (검색 결과 연관 주제 분석 - 차트 1개)
- 목적: 검색 질의와 의미적으로 가장 연관된 주제 1개를 도출하고, 그 비율과 함께 시각화용 데이터(`chart_data`)를 생성하세요. (예: 40대 기혼 남성은 자녀 수와 관련이 깊음)

3. high_ratio_charts (우연히 높은 비율을 보이는 주제 분석 - 차트 3개)
- 목적: 데이터 내에서 검색 질의에 명시되지 않았지만 높은 비율을 차지하거나 뚜렷한 패턴이 있는 속성 **3개**를 선정하고, 그 비율과 함께 시각화용 데이터(`chart_data`)를 생성하세요.

최종 JSON 출력 구조
반드시 다음 구조를 따르세요.

```json
{{
  "main_summary": "검색 결과에 대한 포괄적인 요약입니다. 2~3줄로 작성합니다.",
  "query_focused_chart": {{
    "topic": "결혼 여부",
    "description": "응답자의 100%가 기혼입니다.",
    "ratio": "100.0%",
    "chart_data": [ {{ "label": "결혼 여부", "values": {{ "기혼": 100, "미혼": 0 }} }} ]
  }},
  "related_topic_chart": {{
    "topic": "평균 가족 구성원 수",
    "description": "응답자의 80%가 3인 가족입니다.",
    "ratio": "80.0%",
    "chart_data": [ {{ "label": "가족 크기", "values": {{ "3명": 80, "4명 이상": 20 }} }} ]
  }},
  "high_ratio_charts": [
    {{
      "topic": "가장 많이 사용하는 휴대폰 브랜드",
      "description": "응답자의 95.5%가 삼성전자 휴대폰을 사용합니다.",
      "ratio": "95.5%",
      "chart_data": [ {{ "label": "휴대폰 브랜드", "values": {{ "삼성전자": 95.5, "Apple": 4.5 }} }} ]
    }},
    {{
      "topic": "가구 월소득 분포",
      "description": "응답자의 75%가 월 700만원 이상 가구 소득입니다.",
      "ratio": "75.0%",
      "chart_data": [ {{ "label": "가구 소득", "values": {{ "700만원 이상": 75, "700만원 미만": 25 }} }} ]
    }},
    {{
      "topic": "선호하는 주거 형태",
      "description": "응답자의 60%가 아파트에 거주합니다.",
      "ratio": "60.0%",
      "chart_data": [ {{ "label": "주거 형태", "values": {{ "아파트": 60, "빌라/단독": 40 }} }} ]
    }}
  ]
}}
```
### 작성 규칙
- 반드시 JSON 포맷만 출력하세요.
- ratio는 소수점 한 자리까지 표시 (예: "64.3%")
- summary는 분석 리포트처럼 자연스럽게 작성.
- chart_data는 프론트엔드에서 시각화 가능한 구조로 유지.
- 주제명(topic)은 설문 항목명을 그대로 사용하지 말고, 의미를 가진 한글 문장으로 표현.
"""

prompt_template = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["user_query", "search_results_json"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# --- 4. LangChain 분석 체인 함수 (이전 코드와 동일) ---

def analyze_search_results_chain(user_query: str, search_results: List[Dict[str, Any]]):
    """
    LangChain 체인을 사용하여 Claude 3 Sonnet 모델을 호출하고 결과를 파싱합니다.
    """
    if llm is None or parser is None:
        return {"error": "LLM/Parser가 초기화되지 않았습니다. 환경 설정을 확인하세요."}, 500

    # 💡 최적화: 검색 결과가 없으면 LLM을 호출하지 않고, Pydantic 모델 형식에 맞는 빈 객체를 즉시 반환합니다.
    if not search_results:
        empty_chart = AnalysisChart(topic="", description="", ratio="", chart_data=[ChartDataEntry(label="", values={})])
        empty_output = FinalAnalysisOutput(
            main_summary="검색 결과가 없습니다. 다른 검색어를 시도해 보세요.",
            query_focused_chart=empty_chart,
            related_topic_chart=empty_chart,
            high_ratio_charts=[empty_chart, empty_chart, empty_chart]
        )
        return empty_output.model_dump(), 200 # 💡 .dict() 또는 .model_dump()로 딕셔너리 변환

    # 데이터 샘플 JSON 문자열 준비 (프롬프트 주입용)
    search_results_json = json.dumps(search_results[:150], ensure_ascii=False, indent=2)

    from langchain_core.exceptions import OutputParserException
    try:
        # 💡 중요: chain 객체 생성을 try 블록 안으로 이동합니다.
        # 이렇게 해야 테스트에서 mocker.patch가 올바르게 동작합니다.
        chain = prompt_template | llm | parser

        # 체인 실행
        analysis_result = chain.invoke({
            "user_query": user_query,
            "search_results_json": search_results_json
        })
        
        # Pydantic 모델을 통과한 유효한 JSON 객체 반환
        return analysis_result, 200

    except OutputParserException as e:
        print(f"LLM 응답 JSON 파싱 오류: {e}")
        # LLM의 원본 출력을 포함하여 에러를 반환하면 디버깅에 용이합니다.
        return {"error": "LLM 응답 JSON 파싱 오류", "raw_output": e.llm_output}, 500
    except Exception as e:
        print("LangChain 체인 실행 또는 JSON 파싱 오류:", e)
        # LLM이 JSON이 아닌 다른 응답을 반환했거나 호출에 실패한 경우
        return {"error": f"분석 실패 또는 LLM 응답 JSON 파싱 오류: {str(e)}"}, 500
    

if __name__ == "__main__":
    # 테스트용 사용자 질의
    test_query = "서울에 거주하는 30대 여성의 가전제품 구매 의향에 대한 분석을 요청합니다."
    
    # 테스트용 Mock 검색 결과 데이터 (db_logic.py에서 넘어왔다고 가정)
    test_search_results = [
        {"gender": "F", "birth_year": 1993, "region_major": "서울", "marital_status": "기혼", "owned_electronics": ["TV", "건조기"]},
        {"gender": "F", "birth_year": 1996, "region_major": "경기", "marital_status": "미혼", "owned_electronics": ["TV", "공기청정기"]},
        {"gender": "M", "birth_year": 1990, "region_major": "서울", "marital_status": "기혼", "owned_electronics": ["냉장고", "에어컨"]},
        # LLM 분석을 테스트하기 위해 충분한 수의 데이터를 넣어주세요.
    ] * 5  # 데이터를 복제하여 15개로 늘림 (분석에 용이)

    print("="*50)
    print(f"** LLM 분석 함수 단독 테스트 시작 **")
    print(f"** 테스트 쿼리: {test_query} **")
    print("="*50)
    
    # 함수 직접 호출
    analysis_result, status_code = analyze_search_results_chain(test_query, test_search_results)

    if status_code == 200:
        print("\nLLM 분석 및 JSON 파싱 성공 (Status 200)")
        # 보기 쉽게 JSON 포맷으로 출력
        print(json.dumps(analysis_result, indent=2, ensure_ascii=False))
    else:
        print(f"\nLLM 분석 또는 파싱 실패 (Status {status_code})")
        print(analysis_result)