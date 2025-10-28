import json
import boto3
from botocore.exceptions import ClientError
from flask import Flask, request, jsonify 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Bedrock 클라이언트 생성
try:
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
except Exception as e:
    # 환경 변수 누락 등 초기 에러 발생 시 처리
    print("Boto3 클라이언트 초기화 실패: 환경 변수를 확인하세요.", e)
    # 테스트 종료를 위해 임시로 설정
    bedrock = None

# 프롬프트 생성 함수
def build_opus_prompt(user_query: str, search_results: list) -> str:
    """
    Claude 3 Sonnet 모델용 프롬프트 생성
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
아래의 데이터를 분석하여 총 5개의 차트 데이터를 포함하는 JSON 형식으로 출력하세요.

#### query_focused_chart (검색 질의 특징 분석 - 차트 1개)
- 목적: 사용자의 질의에 포함된 가장 대표적인 인구통계학적 속성 1개 (예: 성별, 연령대, 결혼 여부 등)의 분포를 분석하고 시각화용 데이터(`chart_data`)를 생성하세요.

#### ② related_topic_chart (검색 결과 연관 주제 분석 - 차트 1개)
- 목적: 검색 질의와 의미적으로 가장 연관된 주제 1개를 도출하고, 그 비율과 함께 시각화용 데이터(`chart_data`)를 생성하세요. (예: 40대 기혼 남성은 자녀 수와 관련이 깊음)

#### ③ high_ratio_charts (우연히 높은 비율을 보이는 주제 분석 - 차트 3개)
- 목적: 데이터 내에서 검색 질의에 명시되지 않았지만 높은 비율을 차지하거나 뚜렷한 패턴이 있는 속성 **3개**를 선정하고, 그 비율과 함께 시각화용 데이터(`chart_data`)를 생성하세요.

### 최종 JSON 출력 구조
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

### [데이터 샘플]
{json.dumps(search_results[:150], ensure_ascii=False, indent=2)}
"""
    return prompt

# Bedrock Sonnet 3 모델 호출 함수
def analyze_search_results(user_query: str, search_results: list):
    """
    Claude 3 Sonnet 모델을 호출하여
    검색 결과를 요약 및 시각화용 데이터로 구조화.
    """
    global bedrock
    if bedrock is None:
        return {"error": "Bedrock 클라이언트가 초기화되지 않았습니다. 환경 변수를 확인하세요."}
    
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
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
        # 2. JSON 파싱 실패 시, 마크다운 코드를 제거하고 재시도
        try:
            # ```json과 ```을 제거하고 문자열 정리
            clean_text = output_text.strip().lstrip('```json').rstrip('```').strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            print("JSON 파싱 실패. 원문 반환:")
            return {"raw_output": output_text}
        
    except Exception as e:
        print("예상치 못한 오류 발생:", e)
        return {"error": str(e)}
    
@app.route('/api/analyze', methods=['POST'])
def analyze_data_endpoint():
    """
    프론트엔드로부터 user_query를 받아 분석 결과를 반환하는 API 엔드포인트
    """
    # 1. 요청 본문(body)에서 사용자 질의 추출
    try:
        data = request.get_json()
        user_query = data.get('user_query', '')
        if not user_query:
            return jsonify({"error": "user_query is missing"}), 400
    except Exception:
        return jsonify({"error": "Invalid JSON format"}), 400

    # 2. 실제 검색 로직 (임시로 Mock 데이터 사용)
    MOCK_DATA_COUNT = 50
    mock_search_results = [
        {
            "gender": "M", "birth_year": 1983, "region_major": "서울", "region_minor": "강남구",
            "marital_status": "기혼" if i < 45 else "미혼", # 기혼 90%
            "family_size": "3명" if i < 40 else "2명", # 3명 80%
            "income_household_monthly": "월 700~799만원" if i < 35 else "월 400~499만원", # 700만원 이상 70%
            "car_ownership": "있다",
            "car_manufacturer": "Mercedes-Benz" if i < 30 else "BMW", # 벤츠 60%
            # ... (분석에 필요한 다른 필드도 포함)
        } for i in range(MOCK_DATA_COUNT)
    ]
    
    # 3. Bedrock LLM 분석 함수 호출
    analysis_result = analyze_search_results(user_query, mock_search_results)

    # 4. 결과 반환
    if analysis_result.get("error"):
        return jsonify(analysis_result), 500
    if analysis_result.get("raw_output"):
        # JSON 파싱 실패 시, 원시 텍스트를 담아 프론트에 전달
        return jsonify(analysis_result), 500
        
    return jsonify(analysis_result), 200

if __name__ == "__main__":
    # Flask 서버 실행
    print("Starting Flask Server...")
    # debug=True는 개발용이며, production 환경에서는 반드시 False로 설정해야 합니다.
    # use_reloader=False를 사용하면 VS Code에서 디버깅 시 재로딩 문제를 방지할 수 있습니다.
    app.run(port=8000, debug=True, use_reloader=False)
