import json
import re
import os
import logging
from typing import Dict, List, Optional, Any
from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# 1. Claude 클라이언트 설정
try:
    # API Key는 환경변수(.env)에서 자동으로 로드됩니다.
    CLAUDE_CLIENT = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.1)
except Exception as e:
    CLAUDE_CLIENT = None
    logging.error(f"Anthropic 클라이언트 생성 실패: {e}")


# 2. DB 스키마 정보
DB_SCHEMA_INFO = """
## PostgreSQL (인구통계): gender, birth_year, region_major, marital_status, education_level, job_title_raw, income_household_monthly, car_ownership, smoking_experience, drinking_experience

## Qdrant (벡터 검색):
- welcome_subjective_vectors: 주관식 답변 전체
- qpoll_vectors_v2: 라이프스타일 설문 (ott_count, physical_activity, skincare_spending, ai_chatbot_used, stress_relief_method, travel_planning_style 등 40+ 카테고리)
"""

# 3. 시스템 프롬프트 (수정됨: {{QUERY}} 위치 명시 및 JSON 포맷 최적화)
SYSTEM_PROMPT_V2 = """
당신은 자연어 쿼리를 분석하여 **"정형 필터(SQL)"**와 **"의미 검색 조건(Vector Search)"**으로 완벽하게 분리하는 **Search Query Analyzer**입니다.

## ⚠️ 절대 주의사항
1. 위 예시(Examples)의 데이터(나이, 성별, 키워드)를 그대로 베끼지 마십시오.
2. 반드시 아래 제공되는 **[사용자 쿼리]**의 내용만 분석하십시오.
3. 쿼리에 언급되지 않은 조건(성별, 나이 등)을 임의로 생성하지 마십시오.

## 🎯 목표
사용자의 질문에서 **'누구(Who)'**에 해당하는 인구통계학적 조건과 **'무엇(What)'**에 해당하는 행동/성향/경험 조건을 명확히 분리하여 구조화된 JSON으로 반환합니다.

## 🛠️ 수행 작업 정의

### 1. Demographic Filters (SQL 필터)
- **대상**: 나이, 성별, 거주지역, 결혼여부, 자녀수, 직업, 소득, 휴대폰 기종, 차량 보유 여부 등 **객관적이고 명확한 프로필 정보**.
- **규칙**: 쿼리에 명시된 내용만 추출합니다. (추론 금지)
- **예시**: "20대", "서울 거주", "아이폰 유저", "미혼"

### 2. Semantic Conditions (의미 검색 - 핵심!)
- **대상**: 취미, 습관, 선호도, 라이프스타일, 경험, 가치관, 고민 등 **주관적이거나 행동에 관련된 모든 표현**.
- **규칙**: 인구통계가 아닌 모든 명사/동사 구문은 이곳으로 분류해야 합니다.
- **중요**: "OTT를 보는", "운동을 즐기는", "야식을 먹는", "스트레스 받는" 등은 절대 필터가 아닌 **Semantic Condition**입니다.
- **속성 정의**:
  - `original_keyword`: 사용자 쿼리 그대로의 표현 (예: "OTT 이용")
  - `expanded_queries`: 라우터 매칭을 돕기 위한 3~4개의 구체적인 문장형 동의어. (예: "넷플릭스나 유튜브를 자주 시청한다", "동영상 스트리밍 서비스를 구독 중이다")
  - `importance`: 0.9(필수/핵심주제), 0.7(중요조건), 0.5(단순선호)

---
## 📋 DB 스키마 정보 (참고용)
{schema}
---

## 💡 Few-Shot 예시

### 예시 1: 복합 조건 (필터 + 의미)
**쿼리**: "서울 경기 사는 20대 남성 중 OTT를 즐겨 보고 주말에 배달음식 시켜먹는 사람 30명"
**분석 결과**:
{
  "demographic_filters": {
    "region_major": ["서울", "경기"],
    "age_range": [20, 29],
    "gender": ["남성"]
  },
  "semantic_conditions": [
    {
      "id": "cond_1",
      "original_keyword": "OTT를 즐겨 보고",
      "importance": 0.9,
      "expanded_queries": ["넷플릭스, 왓챠 등 OTT 서비스를 자주 이용한다", "주말에 동영상 스트리밍을 몰아본다", "OTT 구독료를 지출한다"],
      "search_strategy": "category_specific"
    },
    {
      "id": "cond_2",
      "original_keyword": "주말에 배달음식 시켜먹는",
      "importance": 0.7,
      "expanded_queries": ["배달 앱을 자주 사용한다", "주말 식사를 주로 배달 음식으로 해결한다", "배달의민족이나 요기요를 이용한다"],
      "search_strategy": "category_specific"
    }
  ],
  "logic_structure": {"operator": "AND", "children": [{"operator": "LEAF", "condition_id": "cond_1"}, {"operator": "LEAF", "condition_id": "cond_2"}]},
  "search_strategy_recommendation": {"strategy": "balanced"},
  "limit": 30
}

### 예시 2: 의미 조건만 있는 경우
**쿼리**: "여름 휴가 계획이 있는 사람 찾아줘"
**분석 결과**:
{
  "demographic_filters": {},
  "semantic_conditions": [
    {
      "id": "cond_1",
      "original_keyword": "여름 휴가 계획",
      "importance": 0.9,
      "expanded_queries": ["올해 여름 휴가를 떠날 예정이다", "해외 여행이나 국내 여행 계획이 있다", "휴가철 여행지를 알아보고 있다"],
      "search_strategy": "category_specific"
    }
  ],
  "logic_structure": {"operator": "LEAF", "condition_id": "cond_1"},
  "search_strategy_recommendation": {"strategy": "semantic_first"},
  "limit": 50
}

---

## 📤 출력 형식 (JSON Only)
```json
{
  "demographic_filters": { ... },
  "semantic_conditions": [ ... ],
  "logic_structure": { ... },
  "exclude_conditions": [],
  "search_strategy_recommendation": { ... },
  "limit": <number>
}

*** 실제 분석 대상 *** 
사용자 쿼리: 
<query>
{{QUERY}}
</query> 
"""


@lru_cache(maxsize=256)
def parse_query_intelligent(query: str) -> Dict[str, Any]:
   """ 쿼리를 지능적으로 파싱하여 구조화된 검색 조건 생성 """ 
   if CLAUDE_CLIENT is None:
       raise RuntimeError("Claude 클라이언트가 초기화되지 않았습니다.")
   
   logging.info(f"🔄 LLM Parser v2 호출 중: {query}")

   # 프롬프트 생성 (schema는 단순 문자열 치환, QUERY는 사용자 입력 치환)
   prompt = SYSTEM_PROMPT_V2.replace("{{QUERY}}", query).replace("{schema}", DB_SCHEMA_INFO)

   try:
       messages = [
           SystemMessage(content=prompt),
           HumanMessage(content="Analyze the query and provide structured search conditions in JSON.")
       ]
       
       response = CLAUDE_CLIENT.invoke(messages)
       text_output = response.content.strip()
       logging.info(f"🤖 Claude LLM 원본 응답:\n---\n{text_output}\n---")
       
       # JSON 추출 (마크다운 코드 블록 제거)
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
       
       # 기본값 설정 및 반환 구조 생성
       result = {
           'demographic_filters': parsed.get('demographic_filters', {}),
           'semantic_conditions': parsed.get('semantic_conditions', []),
           'logic_structure': parsed.get('logic_structure', {'operator': 'AND', 'children': []}),
           'exclude_conditions': parsed.get('exclude_conditions', []),
           'search_strategy_recommendation': parsed.get('search_strategy_recommendation', {
               'strategy': 'balanced',
               'use_collections': ['welcome_subjective_vectors', 'qpoll_vectors_v2']
           }),
           'limit': parsed.get('limit', 100),
           'query_intent': parsed.get('query_intent', {})
       }
       
       logging.debug(f"✅ LLM Parser v2 완료")
       logging.info(f"  - Demographic filters: {result['demographic_filters']}")
       intent_keywords = [c.get('original_keyword', '') for c in result['semantic_conditions']]
       logging.info(f"  - Semantic conditions: {intent_keywords}")

       # 🔍 디버깅: Semantic Conditions 상세 정보 출력 (DEBUG 레벨)
       if result['semantic_conditions']:
           logging.debug("="*60)
           logging.debug("🔍 [디버깅] Semantic Conditions 상세 정보:")
           for idx, cond in enumerate(result['semantic_conditions'], 1):
               logging.debug(f"  [{idx}] original_keyword: {cond.get('original_keyword')}")
               logging.debug(f"      importance: {cond.get('importance')}")
               logging.debug(f"      search_strategy: {cond.get('search_strategy')}")
               expanded = cond.get('expanded_queries', [])
               if expanded:
                   logging.debug(f"      expanded_queries:")
                   for exp_idx, exp_q in enumerate(expanded, 1):
                       logging.debug(f"        {exp_idx}. {exp_q}")
           logging.debug("="*60)
       
       return result
       
   except json.JSONDecodeError as je:
       logging.error(f"❌ JSON 파싱 실패: {je.msg}. 원본: {text_output}")
       raise RuntimeError(f"Claude 응답 파싱 실패: {je.msg}")
   except Exception as e:
       logging.error(f"❌ Claude 호출 실패: {e}", exc_info=True)
       raise RuntimeError(f"Claude 호출 실패: {e}")


def extract_limit_from_query(query: str) -> Optional[int]:
    """쿼리에서 인원 수 추출"""
    all_limit_matches = re.findall(r'(\d+)\s*명', query)
    if all_limit_matches:
        try:
            return int(all_limit_matches[-1])
        except ValueError:
            pass
    return None