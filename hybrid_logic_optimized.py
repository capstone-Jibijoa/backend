"""
LLM 응답 캐싱을 적용한 hybrid_logic.py 최적화 버전
- Redis 캐시로 동일 쿼리 반복 시 LLM 호출 생략
- 예상 개선: 0.5~2초 → 0.01초 (캐시 히트 시)
"""
import os
import json
import re
import hashlib
from dotenv import load_dotenv
from datetime import datetime
from fastapi import HTTPException
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

# Redis 캐싱 추가
try:
    import redis
    REDIS_CLIENT = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
        socket_connect_timeout=1
    )
    # 연결 테스트
    REDIS_CLIENT.ping()
    print("✅ Redis 캐시 연결 성공")
    REDIS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Redis 연결 실패 (캐싱 비활성화): {e}")
    REDIS_CLIENT = None
    REDIS_AVAILABLE = False

load_dotenv()

# Claude 모델 초기화
try:
    CLAUDE_CLIENT = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.1)
except Exception as e:
    CLAUDE_CLIENT = None
    print(f"Anthropic 클라이언트 생성 실패: {e}")

# 캐시 TTL (초)
CACHE_TTL = 86400  # 24시간


def get_cache_key(query: str) -> str:
    """쿼리에 대한 캐시 키 생성"""
    # 쿼리를 정규화 (공백, 대소문자 등)
    normalized = query.strip().lower()
    # MD5 해시 생성
    hash_value = hashlib.md5(normalized.encode()).hexdigest()
    return f"llm_classify:{hash_value}"


def get_cached_classification(query: str) -> dict:
    """Redis에서 캐시된 분류 결과 조회"""
    if not REDIS_AVAILABLE:
        return None
    
    try:
        cache_key = get_cache_key(query)
        cached = REDIS_CLIENT.get(cache_key)
        
        if cached:
            print(f"✅ LLM 캐시 히트! (키: {cache_key[:20]}...)")
            return json.loads(cached)
        
        return None
    except Exception as e:
        print(f"⚠️  캐시 조회 실패: {e}")
        return None


def set_cached_classification(query: str, result: dict):
    """Redis에 분류 결과 캐싱"""
    if not REDIS_AVAILABLE:
        return
    
    try:
        cache_key = get_cache_key(query)
        REDIS_CLIENT.setex(
            cache_key,
            CACHE_TTL,
            json.dumps(result, ensure_ascii=False)
        )
        print(f"💾 LLM 결과 캐싱 완료 (TTL: {CACHE_TTL}초)")
    except Exception as e:
        print(f"⚠️  캐싱 실패: {e}")


def classify_query_keywords(query: str) -> dict:
    """
    쿼리를 키워드로 분류 (캐싱 적용)
    
    개선점:
    1. 동일 쿼리는 Redis 캐시에서 즉시 반환
    2. LLM 호출 비용 절감
    3. 응답 속도 대폭 향상 (2초 → 0.01초)
    """
    # 1. 캐시 확인
    cached_result = get_cached_classification(query)
    if cached_result:
        return cached_result
    
    # 2. LLM 호출 (캐시 미스)
    print(f"🔄 LLM 호출 중... (캐시 미스)")
    result = _classify_query_keywords_uncached(query)
    
    # 3. 결과 캐싱
    set_cached_classification(query, result)
    
    return result


def _classify_query_keywords_uncached(query: str) -> dict:
    """
    실제 LLM 호출 함수 (캐싱 미적용)
    """
    if CLAUDE_CLIENT is None:
        raise HTTPException(status_code=500, detail="Claude 클라이언트가 초기화되지 않았습니다.")

    system_prompt = """
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
  "welcome_keywords": { ... },
  "qpoll_keywords": { ... },
  "ranked_keywords": [ ... ],
  "query_propensity": "objective_heavy | subjective_heavy | balanced"
}

```

## 예시

쿼리: "IT 기술에 관심 많고 재테크도 잘하는 서울 30대 IT 직장인 100명"
```json
{
  "welcome_keywords": {
    "objective": ["서울", "30대", "직장인"],
    "subjective": [["IT", "기술"], ["재테크", "자산관리"]]
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


def clear_cache(query: str = None):
    """
    캐시 삭제 함수
    
    Args:
        query: 특정 쿼리의 캐시만 삭제 (None이면 전체 삭제)
    """
    if not REDIS_AVAILABLE:
        print("⚠️  Redis 사용 불가")
        return
    
    try:
        if query:
            # 특정 쿼리 캐시 삭제
            cache_key = get_cache_key(query)
            result = REDIS_CLIENT.delete(cache_key)
            if result:
                print(f"✅ 캐시 삭제 완료: {query}")
            else:
                print(f"⚠️  캐시 없음: {query}")
        else:
            # 전체 LLM 캐시 삭제
            pattern = "llm_classify:*"
            keys = REDIS_CLIENT.keys(pattern)
            if keys:
                REDIS_CLIENT.delete(*keys)
                print(f"✅ 전체 캐시 삭제 완료: {len(keys)}개")
            else:
                print("⚠️  삭제할 캐시 없음")
    except Exception as e:
        print(f"❌ 캐시 삭제 실패: {e}")


# 캐시 통계 조회
def get_cache_stats():
    """캐시 통계 반환"""
    if not REDIS_AVAILABLE:
        return {"status": "disabled"}
    
    try:
        pattern = "llm_classify:*"
        keys = REDIS_CLIENT.keys(pattern)
        
        total_size = 0
        for key in keys[:100]:  # 샘플링
            try:
                size = len(REDIS_CLIENT.get(key) or "")
                total_size += size
            except:
                pass
        
        avg_size = total_size / min(len(keys), 100) if keys else 0
        
        return {
            "status": "enabled",
            "total_keys": len(keys),
            "estimated_total_size_mb": (avg_size * len(keys)) / (1024 * 1024),
            "avg_entry_size_kb": avg_size / 1024
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}