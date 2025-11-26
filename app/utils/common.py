import re
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Union, Tuple
from collections import Counter
from functools import lru_cache

# 외부 모듈 의존성 (app 패키지)
from app.core.llm_client import get_claude_client
from app.constants.mapping import (
    QPOLL_ANSWER_TEMPLATES,
    COMMON_NEGATIVE_PATTERNS,
    SPECIFIC_NEGATIVE_PATTERNS,
    KEYWORD_MAPPINGS,
    FIELD_NAME_MAP,
    QPOLL_FIELD_TO_TEXT
)

try:
    from app.services.llm_summarizer import extract_relevant_columns_via_llm
except ImportError:
    logging.warning("llm 모듈을 찾을 수 없습니다. 동적 컬럼 탐색이 제한됩니다.")
    def extract_relevant_columns_via_llm(q, c): return []

def clean_label(text: Any, max_length: int = 25) -> str:
    """라벨 정제 함수: 특수문자 및 괄호 내용 제거"""
    if not text: return ""
    text_str = str(text)
    cleaned = re.sub(r'\([^)]*\)', '', text_str).strip()
    cleaned = " ".join(cleaned.split())
    
    if len(cleaned) > max_length:
        return cleaned[:max_length] + ".."
    return cleaned

def truncate_text(value: Any, max_length: int = 30) -> str:
    """긴 텍스트 말줄임"""
    if value is None: return ""
    if isinstance(value, list):
        text = ", ".join(map(str, value))
    else:
        text = str(value)
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def calculate_age_from_birth_year(birth_year: Any, current_year: int = None) -> int:
    """출생연도로부터 나이 계산 (만 나이/연 나이 기준)"""
    if current_year is None:
        current_year = datetime.now().year 
    try:
        b_year = int(str(birth_year).split('.')[0])
        return current_year - b_year
    except:
        return 0
    
def extract_birth_year_from_raw(age_raw: str) -> int:
    """
    '1971년 03월 07일 (만 54세)' 형태의 문자열에서 연도(1971)만 추출
    """
    if not age_raw or not isinstance(age_raw, str):
        return 0
    try:
        return int(age_raw[:4])
    except ValueError:
        return 0

def get_age_group(birth_year: Any) -> str:
    """생년 -> 연령대 변환"""
    if not birth_year: return "알 수 없음"
    try:
        b_year = int(str(birth_year).split('.')[0])
        current_year = datetime.now().year
        age = current_year - b_year + 1
        if age < 20: return "10대"
        elif age < 30: return "20대"
        elif age < 40: return "30대"
        elif age < 50: return "40대"
        elif age < 60: return "50대"
        else: return "60대 이상"
    except: return "알 수 없음"

def calculate_distribution(values: List[Any]) -> Dict[str, float]:
    """
    리스트 값들의 분포(%) 계산
    (text_utils.py와 common.py의 중복 함수 통합됨)
    """
    if not values: return {}
    total = len(values)
    counts = Counter(values)
    return {k: round((v / total) * 100, 1) for k, v in counts.items()}

def extract_answer_from_template(field_name: str, sentence: str) -> str:
    """
    문장형 데이터에서 '핵심 답변'만 추출
    예: "제 월 소득은 300만원 입니다" -> "300만원"
    """
    if not sentence: return ""
    
    # 1. 특수 필드 하드코딩 처리 (정규식)
    if field_name == "ott_count":
        match = re.search(r'(\d+개|이용 안 함|없음)', sentence)
        if match: return match.group(1)
    elif field_name == "skincare_spending":
        match = re.search(r'(\d+만\s*원|\d+~\d+만\s*원|\d+원)', sentence)
        if match: return match.group(1)

    # 2. 템플릿 매칭 처리 (QPOLL_ANSWER_TEMPLATES 활용)
    template = QPOLL_ANSWER_TEMPLATES.get(field_name)
    if template:
        try:
            # 템플릿을 정규식 패턴으로 변환
            pattern_str = re.escape(template)
            pattern_str = pattern_str.replace(re.escape("{answer_str}"), r"(.*?)")
            
            # 한국어 조사 처리 유연화
            pattern_str = pattern_str.replace(r"\(이\)다", r"(?:이)?다")
            pattern_str = pattern_str.replace(r"\(으\)로", r"(?:으)?로")
            pattern_str = pattern_str.replace(r"\(가\)", r"(?:가)?")
            pattern_str = pattern_str.replace(r"\ ", r"\s*") # 띄어쓰기 유연화

            match = re.search(pattern_str, sentence)
            if match:
                extracted = match.group(1)
                return clean_label(extracted, 20) # 괄호 제거 및 길이 제한
        except Exception:
            pass

    # 3. 매칭 실패 시 기본 정제 후 반환
    return clean_label(sentence, 30)


# ==========================================
# 2. 비즈니스 로직 및 매핑 유틸리티 (from common.py)
# ==========================================

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
    """쿼리와 관련된 필드를 찾습니다."""
    related_fields = set()
    
    # 1. 필드 설명(FIELD_NAME_MAP) 전체 스캔
    for field_key, field_desc in FIELD_NAME_MAP.items():
        # 필드 설명이 쿼리의 일부 단어를 포함하는지 확인 (간단한 키워드 매칭)
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

def filter_merged_panels(panels_data: List[Dict], filters: Dict[str, Union[str, List[str]]]) -> List[Dict]:
    """
    [기능]
    get_panels_data_from_db()로 병합된 패널 데이터 리스트를 입력받아,
    메모리 상에서 지역(region) 및 성별(gender) 등의 조건으로 필터링합니다.

    [특징]
    - 'region' 필터 시: qpoll_meta의 'region'이 있으면 그것을, 없으면 welcome_meta2의 'region_major'를 확인합니다.
    - 'gender' 필터 시: 병합된 'gender' 값을 확인합니다.
    - 리스트 입력 지원: filters={'region': ['서울', '경기']} 와 같이 다중 선택이 가능합니다.

    [사용 예시]
    filtered = filter_merged_panels(all_panels, {'region': ['서울'], 'gender': '여성'})
    """
    if not panels_data or not filters:
        return panels_data

    filtered_list = []

    for panel in panels_data:
        is_match = True

        for key, condition in filters.items():
            # 1. 비교할 패널의 값 추출 (우선순위 로직 적용)
            panel_value = None
            
            if key == 'region':
                # region(Qpoll) 우선 확인 -> 없으면 region_major(Welcome) 확인
                panel_value = panel.get('region') or panel.get('region_major')
            elif key == 'gender':
                panel_value = panel.get('gender')
            else:
                # 그 외 필드는 키 그대로 확인
                panel_value = panel.get(key)

            # 데이터 정제 (공백 제거 등)
            if isinstance(panel_value, str):
                panel_value = panel_value.strip()

            # 2. 조건 비교
            if condition:
                if isinstance(condition, list):
                    # 필터가 리스트인 경우 (OR 조건): ['서울', '경기'] 중 하나라도 포함되면 통과
                    # (부분 일치 허용: "서울특별시" == "서울")
                    match_found = False
                    for cond_item in condition:
                        if panel_value and str(cond_item) in str(panel_value):
                            match_found = True
                            break
                    if not match_found:
                        is_match = False
                else:
                    # 필터가 단일 값인 경우
                    if not panel_value or str(condition) not in str(panel_value):
                        is_match = False
            
            if not is_match:
                break  # 하나라도 조건 불일치 시 해당 패널 제외
        
        if is_match:
            filtered_list.append(panel)

    return filtered_list