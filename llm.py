import os
import json
import re
import logging
from dotenv import load_dotenv
from functools import lru_cache
from fastapi import HTTPException
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

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
    [v2 개선] 쿼리를 4개 카테고리로 정확하게 분류
    - objective_keywords: PostgreSQL demographic 필터링
    - must_have_keywords: 벡터 검색 엄격 검증 (높은 threshold 0.55+)
    - preference_keywords: 벡터 검색 선호도 (낮은 threshold 0.38)
    - negative_keywords: 제외 조건
    """
    if CLAUDE_CLIENT is None:
        raise HTTPException(status_code=500, detail="Claude 클라이언트가 초기화되지 않았습니다.")

    system_prompt = """
당신은 사용자 쿼리를 분석하여 4가지 카테고리로 정확하게 분류하는 전문가입니다.

## 분류 카테고리 (우선순위 순서로 판단)

1. **objective_keywords** (PostgreSQL 필터링용 - 명확한 demographic 조건)
   - 인구통계: 성별, 연령대(10대/20대/30대/40대/50대/60대 이상), 지역(시도 단위)
   - 명확한 속성: 결혼여부, 학력, 직업, 직무, 소득, 가족수, 자녀수
   - 소유 여부: 휴대폰 브랜드, 차량 보유 여부, 차량 제조사
   - 경험 여부: 흡연, 음주, 전자담배
   - **중요**: '젊은층'(20~30대), 'MZ세대'(20~30대), '중장년층'(40~50대) 등도 여기 포함
   - 예시: "서울", "20대", "여성", "기혼", "대졸", "사무직", "아이폰"

2. **must_have_keywords** (벡터 검색 - 엄격 검증용, 반드시 충족해야 함)
   - 사용자가 **명시적으로 요구한 행동, 경험, 태도, 라이프스타일**
   - 키워드 패턴: "~하는 사람", "~을/를 이용하는", "~을/를 하는", "~이 있는"
   - **정확히 일치해야 하는 주관적 조건**
   - **중요 제외 패턴**: "선호하는", "좋아하는", "관심있는", "원하는"은 **Preference**로 분류
   - **중요**: 동의어를 3~5개 생성하여 검색 정확도 향상 (다양한 표현 방식 포함)
   - 예시: 
     * "OTT 이용" → ["OTT 구독", "넷플릭스 시청", "스트리밍 서비스 이용", "온라인 영상 플랫폼"]
     * "헬스장 다니는" → ["운동 정기적으로", "피트니스 등록", "체력 단련", "헬스 트레이닝"]
     * "해외여행 좋아하는" → ["여행 자주 가는", "외국 여행", "해외 방문", "국제 여행 경험"]

3. **preference_keywords** (벡터 검색 - 선호도용, 있으면 좋은 조건)
   - 명시적이지 않지만 **선호하면 좋은 추상적 개념, 가치관, 성향**
   - **"선호하는", "좋아하는", "관심있는", "원하는" 등의 표현 포함**
   - 쿼리에서 암묵적으로 드러나는 특성
   - must_have와 명확히 구분: 필수가 아닌 선호 조건
   - 예시: "가성비", "워라밸", "환경보호", "자기계발", "트렌디한", "가전제품", "패션"

4. **negative_keywords** (제외할 조건 - 명확한 부정 표현)
   - 사용자가 **명시적으로 제외하길 원하는 조건**
   - 키워드 패턴: "~하지 않는", "~을/를 안 하는", "~이 없는", "~제외", "~빼고"
   - 예시: "OTT 미이용", "비흡연자", "차량 없는", "결혼 안 한"

## 분류 원칙
- **정확성 우선**: 애매하면 must_have보다 preference로 분류
- **동의어 확장 필수**: must_have는 반드시 3개 이상의 다양한 표현 생성
- **중복 제거**: 같은 의미는 한 카테고리에 한 번만 포함
- **부정 명확화**: "~하지 않는"은 반드시 negative_keywords로 분류

출력 (순수 JSON만, 코드 블록 없이)
{
  "objective_keywords": ["필터링1", "필터링2"],
  "must_have_keywords": ["필수조건1", "필수조건1_동의어1", "필수조건1_동의어2"],
  "preference_keywords": ["선호1", "선호2"],
  "negative_keywords": ["제외1"],
  "limit": <숫자>
}

## 예시 1
쿼리: "서울, 경기 지역에 사는 OTT를 이용하는 젊은층 30명"
{
  "objective_keywords": ["서울", "경기", "젊은층"],
  "must_have_keywords": ["OTT 이용", "OTT 구독", "넷플릭스 시청", "스트리밍 서비스 이용", "온라인 영상 플랫폼"],
  "preference_keywords": [],
  "negative_keywords": [],
  "limit": 30
}

## 예시 2
쿼리: "30대 여성 중 헬스장 다니고 가성비 중시하는 사람 50명"
{
  "objective_keywords": ["30대", "여성"],
  "must_have_keywords": ["헬스장 다니는", "피트니스 운동", "체력 단련", "정기적인 운동", "헬스 트레이닝"],
  "preference_keywords": ["가성비", "비용 효율", "가격 민감도"],
  "negative_keywords": [],
  "limit": 50
}

## 예시 3
쿼리: "서울 거주 20대 남성 중 차량 없고 대중교통 이용하는 사람 100명"
{
  "objective_keywords": ["서울", "20대", "남성"],
  "must_have_keywords": ["대중교통 이용", "지하철 버스 통근", "교통카드 사용", "공공교통 활용"],
  "preference_keywords": [],
  "negative_keywords": ["차량 보유", "자가용 운전", "자동차 소유"],
  "limit": 100
}

## 예시 4
쿼리: "경기 사는 40대 기혼자 중 해외여행 좋아하고 환경보호에 관심있는 사람 80명"
{
  "objective_keywords": ["경기", "40대", "기혼"],
  "must_have_keywords": ["해외여행 좋아하는", "여행 자주 가는", "외국 여행", "해외 방문 경험"],
  "preference_keywords": ["환경보호", "친환경", "지속가능성"],
  "negative_keywords": [],
  "limit": 80
}

## 예시 5 (중요)
쿼리: "30대 여성이 선호하는 가전제품"
{
  "objective_keywords": ["30대", "여성"],
  "must_have_keywords": [],
  "preference_keywords": ["가전제품 선호", "가전제품 관심", "가전 제품", "전자제품", "생활 가전"],
  "negative_keywords": [],
  "limit": 100
}

## 예시 6 
쿼리: "서울 거주 20대 중 패션에 관심있는 사람 50명"
{
  "objective_keywords": ["서울", "20대"],
  "must_have_keywords": [],
  "preference_keywords": ["패션 관심", "패션 선호", "옷 쇼핑", "스타일링", "트렌디한"],
  "negative_keywords": [],
  "limit": 50
}
## 예시 7 
쿼리: "20대 남성이 많이 타는 차종"
{
  "objective_keywords": ["20대", "남성"],
  "must_have_keywords": ["차종", "챠랑 종류", "자동차 모델"],
  "preference_keywords": ["자동차", "차량 모델", "인기 차종"],
  "negative_keywords": [],
  "limit": 100
}

사용자 쿼리:
<query>
{{QUERY}}
</query>
"""

    logging.info(f"🔄 LLM v2 호출 중... (쿼리: {query})")

    # 쿼리에서 인원 수 추출
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

        # JSON 추출
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

        # 중복 제거 및 키 이름 변환 (backward compatibility)
        result = {
            'objective_keywords': list(set(parsed.get('objective_keywords', []))),
            'must_have_keywords': list(set(parsed.get('must_have_keywords', []))),
            'preference_keywords': list(set(parsed.get('preference_keywords', []))),
            'negative_keywords': list(set(parsed.get('negative_keywords', []))),
            'limit': limit_value or parsed.get('limit')
        }

        logging.info(f"✅ LLM v2 분류 완료:")
        logging.info(f"  - Objective: {result['objective_keywords']}")
        logging.info(f"  - Must-have: {result['must_have_keywords']}")
        logging.info(f"  - Preference: {result['preference_keywords']}")
        logging.info(f"  - Negative: {result['negative_keywords']}")

        return result

    except json.JSONDecodeError as je:
        logging.error(f"❌ JSON 파싱 실패: {je.msg}. 원본: {json_str}")
        raise HTTPException(status_code=500, detail=f"Claude 응답 파싱 실패: {je.msg}")
    except Exception as e:
        logging.error(f"Claude 호출 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Claude 호출 실패: {e}") from e