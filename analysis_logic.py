import json
import boto3
from botocore.exceptions import ClientError

# Bedrock 클라이언트 생성
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# 프롬프트 생성 함수
def build_opus_prompt(user_query: str, search_results: list) -> str:
    """
    Claude 3 Opus 4.1 모델용 프롬프트 생성
    - 설문데이터의 구조적 특징을 반영한 최적화 버전
    """
    prompt = f"""
당신은 데이터 분석가이자 통계 시각화 전문가입니다.
다음은 사용자의 자연어 질의와 그에 해당하는 설문 데이터입니다.

### [사용자 질의]
"{user_query}"

### [데이터 샘플] (최대 150명)
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

### 분석 목표
아래의 데이터를 분석하여 다음 세 가지를 JSON 형식으로 출력하세요.

#### ① main_summary
- 전체 데이터의 대표적인 특징을 요약한 3~5줄 텍스트.
- 인구통계학적 분포, 생활 패턴, 소비 특성 등을 포함하세요.

#### ② related_topics
- 검색 질의와 의미적으로 연관된 주제 2개를 도출하세요.
- 각 주제별 설명과 관련 비율(%)을 함께 작성하세요.
- 예시:
[
  {{ "topic": "음주 빈도", "description": "흡연자 중 60%가 주 1회 이상 음주", "ratio": "60%" }},
  {{ "topic": "자동차 보유", "description": "흡연자 중 70%가 자가 차량을 소유", "ratio": "70%" }}
]

#### ③ high_ratio_topics
- 데이터 내에서 높은 비율을 차지하거나 뚜렷한 패턴이 있는 속성 3개를 선정하세요.
- 각 항목에 대한 요약 설명과 시각화용 데이터(`chart_data`)를 포함하세요.
- chart_data는 원형 다이어그램 시각화에 사용할 JSON 구조로 출력하세요.

출력 예시:
```json
{{
  "summary": "이 데이터는 40대 이상 남성 흡연자 중심으로 구성되며...",
  "related_topics": [
    {{ "topic": "음주 빈도", "description": "흡연자의 60%는 주 1회 이상 음주", "ratio": "60%" }},
    {{ "topic": "자동차 보유", "description": "흡연자의 70%는 자동차를 소유", "ratio": "70%" }}
  ],
  "high_ratio_topics": [
    {{
      "topic": "가구 소득 수준",
      "description": "응답자의 45%가 월 700만원 이상 가구 소득",
      "ratio": "45%",
      "chart_data": [
        {{ "label": "가구 소득", "values": {{ "700만원 이상": 45, "700만원 미만": 55 }} }}
      ]
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

### [데이터 샘플]
{json.dumps(search_results[:150], ensure_ascii=False, indent=2)}
"""
    return prompt

# Bedrock Opus 4.1 모델 호출 함수
def analyze_search_results(user_query: str, search_results: list):
    """
    Claude 3 Opus 4.1 모델을 호출하여
    검색 결과를 요약 및 시각화용 데이터로 구조화.
    """
    model_id = "anthropic.claude-3-opus-20240806-v1:0"
    prompt = build_opus_prompt(user_query, search_results)

    # Bedrock API 요청 바디 구성
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1800,
        "temperature": 0.4,  # 일관된 구조 출력
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    }

    try:
        # Bedrock API 호출
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        # 응답 파싱
        result = json.loads(response["body"].read())
        output_text = result.get("content", [])[0].get("text", "").strip()

        # Claude가 JSON 포맷을 반환하도록 요청했으므로 변환
        return json.loads(output_text)

    except ClientError as e:
        print("Bedrock 호출 실패:", e)
        return {"error": str(e)}

    except json.JSONDecodeError:
        print("JSON 파싱 실패. 원문 반환:")
        return {"raw_output": output_text}
