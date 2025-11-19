import os
import json
import re
import logging
from dotenv import load_dotenv
from functools import lru_cache
from fastapi import HTTPException
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Optional
#from settings import settings

from mapping_rules import QPOLL_FIELD_TO_TEXT

load_dotenv()

# Claude 클라이언트 초기화
try:
    CLAUDE_CLIENT = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.1)
except Exception as e:
    CLAUDE_CLIENT = None
    logging.error(f"Anthropic 클라이언트 생성 실패: {e}")

@lru_cache(maxsize=128)
def classify_query_keywords(query: str) -> dict:
    """
    쿼리를 3개 카테고리(must_have, preference, negative)와
    1개의 구조화된 필터(structured_filters)로 분류합니다.
    """
    if CLAUDE_CLIENT is None:
        raise HTTPException(status_code=500, detail="Claude 클라이언트가 초기화되지 않았습니다.")

    system_prompt = """
당신은 사용자 쿼리를 분석하여, PostgreSQL 필터링을 위한 `structured_filters`와 벡터 검색을 위한 3가지 키워드 카테고리로 정확하게 분류하는 전문가입니다.

## 1. `structured_filters` (PostgreSQL 필터링용)
- 사용자의 인구통계학적 또는 명확한 사실 기반 요청을 구조화된 JSON 객체로 변환합니다.
- **객체 구조**: `{"field": "필드명", "operator": "연산자", "value": "값"}`
- **지원 연산자**: `eq`, `in`, `between`, `like`, `gte`, `lte`
- **필드명 매핑**:
  - 나이/연령대 -> `age`로 생성하세요. (예: "20대" -> `{"field": "age", "operator": "between", "value": [20, 29]}`) **중요: 실제 DB 필드는 `birth_year`이지만, 시스템이 `age`를 보고 변환하므로 항상 `age`로 만드세요.**
  - 성별 -> `gender` (예: "여성" -> `{"field": "gender", "operator": "eq", "value": "F"}`)
  - 지역 -> `region_major` (예: "서울" -> `{"field": "region_major", "operator": "eq", "value": "서울"}`)
  - 직업 -> `job_title_raw`
  - 직무 -> `job_duty_raw`
  - 결혼 -> `marital_status` (예: "기혼" -> `{"field": "marital_status", "operator": "eq", "value": "기혼"}`)
  - 차량보유 -> `car_ownership` (예: "차량 없는" -> `{"field": "car_ownership", "operator": "eq", "value": "없다"}`)
- **중요**: '젊은층'은 20-30대, 'MZ세대'는 20-30대, '중장년층'은 40-50대로 해석하여 `age` `between`으로 변환하세요.

## 2. 키워드 카테고리 (벡터 검색용)
- **must_have_keywords**: 사용자가 명시적으로 요구한 **행동, 경험, 태도**. (예: "OTT 이용", "헬스장 다니는")
- **preference_keywords**: 있으면 좋은 **추상적 개념, 가치관, 성향**. (예: "가성비", "워라밸", "트렌디한")
- **negative_keywords**: 명시적으로 **제외하길 원하는 조건**. (예: "비흡연자", "운동 안하는")

## 출력 규칙
- **출력 형식**: 순수 JSON만, 코드 블록 없이 출력합니다.
- **`must_have_keywords`**: 동의어 생성 없이 대표 키워드 1개만 사용합니다.
- **추론 금지**: 사용자가 명시적으로 언급한 주제만 키워드로 포함하세요. 인구통계 정보만으로 주제를 추론하지 마세요.

## 최종 JSON 출력 형식
{
  "structured_filters": [
    {"field": "필드명", "operator": "연산자", "value": "값"}
  ],
  "must_have_keywords": ["필수조건1"],
  "preference_keywords": ["선호1", "선호2"],
  "negative_keywords": ["제외1"],
  "limit": <숫자>
}
---
## 예시 1
쿼리: "서울, 경기 지역에 사는 OTT를 이용하는 젊은층 30명"
{
  "structured_filters": [
    {"field": "region_major", "operator": "in", "value": ["서울", "경기"]},
    {"field": "age", "operator": "between", "value": [20, 39]}
  ],
  "must_have_keywords": ["OTT 이용"],
  "preference_keywords": [],
  "negative_keywords": ["OTT 미이용", "OTT 안보는", "스트리밍 서비스 미사용"],
  "limit": 30
}

## 예시 2
쿼리: "30대 여성 중 헬스장 다니고 가성비 중시하는 사람 50명"
{
  "structured_filters": [
    {"field": "age", "operator": "between", "value": [30, 39]},
    {"field": "gender", "operator": "eq", "value": "F"}
  ],
  "must_have_keywords": ["헬스장 다니는"],
  "preference_keywords": ["가성비", "비용 효율", "가격 민감도"],
  "negative_keywords": ["운동 안하는", "헬스장 안가는", "비활동적인"],
  "limit": 50
}

## 예시 3
쿼리: "차량 없는 40대 기혼 남성"
{
  "structured_filters": [
    {"field": "car_ownership", "operator": "eq", "value": "없다"},
    {"field": "age", "operator": "between", "value": [40, 49]},
    {"field": "marital_status", "operator": "eq", "value": "기혼"},
    {"field": "gender", "operator": "eq", "value": "M"}
  ],
  "must_have_keywords": [],
  "preference_keywords": [],
  "negative_keywords": [],
  "limit": 100
}
---
사용자 쿼리:
<query>
{{QUERY}}
</query>
"""

    logging.info(f"🔄 LLM 호출 중...")

    limit_value = None
    all_limit_matches = re.findall(r'(\d+)\s*명', query)
    if all_limit_matches:
        try:
            limit_value = int(all_limit_matches[-1])
            logging.info(f"💡 인원 수 감지: {limit_value}명")
        except ValueError:
            pass

    try:
        messages = [
            SystemMessage(content=system_prompt.replace("{{QUERY}}", query)),
            HumanMessage(content="Analyze the query and provide JSON output.")
        ]
        response = CLAUDE_CLIENT.invoke(messages)
        text_output = response.content.strip()

        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'({.*})', text_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = text_output

        parsed = json.loads(json_str)

        # 'objective_keywords'를 'structured_filters'로 변경하여 파싱
        result = {
            'structured_filters': parsed.get('structured_filters', []),
            'must_have_keywords': list(set(parsed.get('must_have_keywords', []))),
            'preference_keywords': list(set(parsed.get('preference_keywords', []))),
            'negative_keywords': list(set(parsed.get('negative_keywords', []))),
            'limit': limit_value or parsed.get('limit')
        }

        logging.info(f"✅ LLM 분류 완료")
        return result

    except json.JSONDecodeError as je:
        logging.error(f"❌ JSON 파싱 실패: {je.msg}. 원본: {json_str}")
        raise HTTPException(status_code=500, detail=f"Claude 응답 파싱 실패: {je.msg}")
    except Exception as e:
        logging.error(f"Claude 호출 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Claude 호출 실패: {e}") from e

@lru_cache(maxsize=128)
def classify_keyword_to_qpoll_topic(keyword: str) -> Optional[str]:
    """
    주어진 키워드를 Q-Poll 주제 목록 중 가장 적합한 필드명으로 분류합니다.
    """
    if CLAUDE_CLIENT is None:
        logging.error("Claude 클라이언트가 초기화되지 않았습니다.")
        return None

    # Q-Poll 주제 목록을 LLM 프롬프트에 포함하기 위해 포맷팅
    qpoll_topics_formatted = "\n".join(
        [f"- {field}: {desc}" for field, desc in QPOLL_FIELD_TO_TEXT.items()]
    )

    system_prompt = f"""
당신은 키워드를 미리 정의된 Q-Poll 주제로 분류하는 전문가입니다.
사용자 키워드와 Q-Poll 주제 목록(설명 포함)이 주어졌을 때,
가장 관련성이 높은 Q-Poll 주제의 필드명(FIELD_NAME)을 하나 선택해야 합니다.

어떤 주제도 관련이 없다면, "None"을 반환하세요.

사용 가능한 Q-Poll 주제 (FIELD_NAME: DESCRIPTION):
{qpoll_topics_formatted}

지침:
1. 제공된 키워드를 가장 잘 설명하는 Q-Poll 주제를 식별하세요.
2. 선택된 주제의 필드명만 반환하세요.
3. 관련 주제가 없다면, "None"을 반환하세요.
"""

    logging.info(f"🔄 LLM을 사용하여 키워드 '{keyword}'를 Q-Poll 주제로 분류 시도 중...")

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"키워드: {keyword}")
        ]
        response = CLAUDE_CLIENT.invoke(messages)
        classification_result = response.content.strip()

        if classification_result in QPOLL_FIELD_TO_TEXT:
            logging.info(f"✅ 키워드 '{keyword}' -> Q-Poll 주제: '{classification_result}'")
            return classification_result
        elif classification_result == "None":
            logging.info(f"⚠️ 키워드 '{keyword}'에 대한 관련 Q-Poll 주제를 찾을 수 없습니다.")
            return None
        else:
            logging.warning(f"🤔 LLM이 예상치 못한 응답을 반환했습니다: '{classification_result}'")
            return None

    except Exception as e:
        logging.error(f"❌ 키워드 '{keyword}' Q-Poll 주제 분류 중 오류 발생: {e}", exc_info=True)
        return None