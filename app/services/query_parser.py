import json
import re
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.llm_client import CLAUDE_CLIENT  # [1]번 파일에서 import

# --- DB Schema & Prompts ---
DB_SCHEMA_INFO = """
## PostgreSQL (인구통계): gender, birth_year, region_major, marital_status, education_level, job_title_raw, income_household_monthly, car_ownership, smoking_experience, drinking_experience
## Qdrant (벡터 검색): welcome_subjective_vectors, qpoll_vectors_v2
"""
SYSTEM_PROMPT_V2 = """
You are a Search Query Analyzer. Parse the query into "Demographic Filters (SQL)" and "Semantic Conditions (Vector)".

## 📋 DB Schema
{schema}

## 🛠️ Extraction Rules

### 1. Demographic Filters (Strict SQL)
Extract ONLY explicit matches. **ALL VALUES MUST BE TRANSLATED TO KOREAN.**
Extract ONLY explicit matches for these fields:
- **Basic**: `age` (convert to range), `gender`, `region` (e.g., 서울, 경기).
- **Social**: `marital_status`, `family_size`, `children_count`.
- **Status**: `job`, `education_level`, `income_personal`, `income_household`.
- **Asset**: `car_ownership` (Only 'have car' or 'no car').
*Note: Do NOT infer missing data. Exclude smoking/drinking/appliances here.*

### 2. Semantic Conditions (Vector Search)
Extract all other subjective intents, hobbies, habits, and specific item ownerships (e.g., specific car model, phone type).
- **Negative Handling**: Mark "don't", "no", "hate" (e.g., "안 하는", "없는") as **`is_negative: true`**.
- **Expansion**: Generate 3 positive synonyms in `expanded_queries` even for negative conditions.
- **Importance**: 0.9 (Core), 0.7 (Important), 0.5 (Optional).

## 💡 Few-Shot Examples

**Query**: "서울 경기 사는 20대 남성 중 OTT 즐겨 보고 주말에 배달음식 시켜먹는 사람 30명"
**Output**:
{
  "demographic_filters": { "region_major": ["서울", "경기"], "age_range": [20, 29], "gender": ["남성"] },
  "semantic_conditions": [
    { "original_keyword": "OTT를 즐겨 보고", "is_negative": false, "importance": 0.9, "expanded_queries": ["넷플릭스나 유튜브를 자주 본다", "동영상 스트리밍 구독 중이다", "주말에 드라마 정주행한다"] },
    { "original_keyword": "주말에 배달음식 시켜먹는", "importance": 0.7, "expanded_queries": ["배달 앱을 자주 쓴다", "요기요나 배민을 이용한다", "배달 음식을 선호한다"] }
  ],
  "limit": 30
}

**Query**: "경기도 사는 30대 중 고양이를 안 키우는 사람"
**Output**:
{
  "demographic_filters": { "region_major": ["경기"], "age_range": [30, 39] },
  "semantic_conditions": [
    { 
      "original_keyword": "고양이를 안 키우는", 
      "is_negative": true, 
      "importance": 0.9, 
      "expanded_queries": ["고양이를 키운다", "반려묘가 있다", "고양이 집사다"],
      "note": "Filter out people similar to expanded_queries"
    }
  ],
  "limit": 100
}

## 📤 Output Format (JSON Only)
Return ONLY the raw JSON.
{
  "demographic_filters": { ... },
  "semantic_conditions": [ ... ],
  "limit": <number>
}

*** Target Query ***
<query>
{{QUERY}}
</query> 
"""

def extract_limit_from_query(query: str) -> Optional[int]:
    """쿼리에서 인원 수 추출"""
    all_limit_matches = re.findall(r'(\d+)\s*명', query)
    if all_limit_matches:
        try:
            return int(all_limit_matches[-1])
        except ValueError:
            pass
    return None

@lru_cache(maxsize=256)
def parse_query_intelligent(query: str) -> Dict[str, Any]:
    """쿼리를 지능적으로 파싱하여 구조화된 검색 조건 생성"""
    if CLAUDE_CLIENT is None:
        raise RuntimeError("Claude 클라이언트가 초기화되지 않았습니다.")
    
    logging.info(f"🔄 LLM Parser 호출: {query}")
    prompt = SYSTEM_PROMPT_V2.replace("{{QUERY}}", query).replace("{schema}", DB_SCHEMA_INFO)

    try:
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Analyze the query and provide structured search conditions in JSON.")
        ]
        
        response = CLAUDE_CLIENT.invoke(messages)
        text_output = response.content.strip()
        
        # JSON 추출 로직
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text_output, re.DOTALL)
        if not json_match:
            json_match = re.search(r'({.*})', text_output, re.DOTALL)
        
        json_str = json_match.group(1) if json_match else text_output
        parsed = json.loads(json_str)
        
        # 결과 구조화
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
        
    except Exception as e:
        logging.error(f"❌ Query Parsing 실패: {e}", exc_info=True)
        raise RuntimeError(f"Claude 호출 실패: {e}")