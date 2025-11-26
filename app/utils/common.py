import re
import logging
from typing import List, Dict, Optional, Any
from functools import lru_cache

# 분리된 상수 파일에서 데이터 import
from app.constants.mapping import (
    COMMON_NEGATIVE_PATTERNS,
    SPECIFIC_NEGATIVE_PATTERNS,
    KEYWORD_MAPPINGS,
    FIELD_NAME_MAP,
    QPOLL_FIELD_TO_TEXT
)

# LLM 관련 함수 import (llm.py 파일에서 가져옴)
try:
    from llm import extract_relevant_columns_via_llm
except ImportError:
    # llm 모듈이 없을 경우를 대비한 가짜 구현 또는 에러 처리
    logging.warning("llm 모듈을 찾을 수 없습니다. 동적 컬럼 탐색이 제한됩니다.")
    def extract_relevant_columns_via_llm(q, c): return []

def get_negative_patterns(field_name: str) -> List[str]:
    """
    특정 필드에 대한 부정 표현 패턴 목록을 반환합니다.
    공통 부정 패턴과 필드별 특수 부정 패턴을 합칩니다.
    """
    patterns = COMMON_NEGATIVE_PATTERNS.copy()
    if field_name in SPECIFIC_NEGATIVE_PATTERNS:
        patterns.extend(SPECIFIC_NEGATIVE_PATTERNS[field_name])
    return patterns

@lru_cache(maxsize=512)
def get_field_mapping(keyword: str) -> Optional[Dict[str, Any]]:
    """
    키워드를 기반으로 매핑된 필드 정보를 찾습니다. (캐싱 적용)
    """
    search_keyword = keyword.lower().strip()
    for pattern, mapping_info in KEYWORD_MAPPINGS:
        result_info = mapping_info.copy()
        if isinstance(pattern, re.Pattern):
            if pattern.search(search_keyword):
                return result_info
        elif isinstance(pattern, str):
            if pattern.lower() in search_keyword:
                return result_info
                
    return {
        "field": "unknown", 
        "description": keyword, 
        "type": "unknown"
    }

def find_related_fields(query: str) -> List[str]:
    
    related_fields = set()
    
    # 1. 필드 설명(FIELD_NAME_MAP) 전체 스캔
    for field_key, field_desc in FIELD_NAME_MAP.items():
        # 필드 설명이 쿼리의 일부 단어를 포함하는지 확인 (간단한 키워드 매칭)
        # 쿼리를 단어 단위로 쪼개서 확인
        query_words = query.split()
        for word in query_words:
            if len(word) >= 2 and word in field_desc: # 2글자 이상만 매칭
                related_fields.add(field_key)
    
    # 2. 비즈니스 로직상 강제 연결이 필요한 경우만 최소한으로 정의
    IMPLICIT_RELATIONS = {
        '여행': ['income_household_monthly'],
        '차': ['car_model_raw', 'car_manufacturer_raw'],
        '자동차': ['car_model_raw', 'car_manufacturer_raw'],
        '자녀': ['children_count', 'family_size'],
        '결혼': ['marital_status'],
        '소득': ['job_title_raw', 'education_level']
    }
    
    for keyword, fields in IMPLICIT_RELATIONS.items():
        if keyword in query:
            related_fields.update(fields)
            
    return list(related_fields)

def find_target_columns_dynamic(question: str) -> List[str]:
    """
    질문 의도를 파악해 분석할 타겟 컬럼들을 LLM을 통해 동적으로 선정합니다.
    """
    # 1. LLM에게 제공할 컬럼 메타데이터 생성 (필드명: 한글설명)
    all_fields_str = ""
    valid_columns = []
    
    # 기본 인구통계 + Q-Poll 필드 병합
    for eng, kor in FIELD_NAME_MAP.items():
        all_fields_str += f"- {eng}: {kor}\n"
        valid_columns.append(eng)
        
    # Q-Poll 질문 텍스트도 매핑 정보로 활용
    for eng, text in QPOLL_FIELD_TO_TEXT.items():
        if eng not in FIELD_NAME_MAP: # 중복 제외
            all_fields_str += f"- {eng}: {text}\n"
            valid_columns.append(eng)

    # 2. LLM 호출
    logging.info(f"🔍 동적 컬럼 탐색 시작: '{question}'")
    found_columns = extract_relevant_columns_via_llm(question, all_fields_str)
    
    # 3. 유효성 검사 (실제 존재하는 컬럼만 필터링)
    final_columns = [col for col in found_columns if col in valid_columns]
    
    # 4. 보정 로직 (예: 질문에 '소득'이 있으면 관련 필드 강제 추가)
    if '소득' in question and 'income_personal_monthly' not in final_columns:
        final_columns.append('income_personal_monthly')

    logging.info(f"✅ 매핑 완료: {final_columns}")
    return final_columns