import re
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Union, Tuple
from collections import Counter
from functools import lru_cache

from app.core.llm_client import get_claude_client

from app.constants.mapping import (
    QPOLL_ANSWER_TEMPLATES,
    COMMON_NEGATIVE_PATTERNS,
    SPECIFIC_NEGATIVE_PATTERNS,
    KEYWORD_MAPPINGS,
    FIELD_NAME_MAP,
    QPOLL_FIELD_TO_TEXT,
    VALUE_TRANSLATION_MAP  
)

try:
    from app.services.llm_summarizer import extract_relevant_columns_via_llm
except ImportError:
    logging.warning("llm 모듈을 찾을 수 없습니다. 동적 컬럼 탐색이 제한됩니다.")
    def extract_relevant_columns_via_llm(q, c): return []

def clean_label(text: Any, max_length: int = 25) -> str:
    """라벨 정제 함수"""
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
    """출생연도로부터 나이 계산"""
    if current_year is None:
        current_year = datetime.now().year 
    try:
        b_year = int(str(birth_year).split('.')[0])
        return current_year - b_year
    except:
        return 0
    
def extract_birth_year_from_raw(age_raw: str) -> int:
    """ '1971년...' 또는 '1971' 등에서 연도 추출 """
    if not age_raw: return 0
    if isinstance(age_raw, int): return age_raw
    if str(age_raw).isdigit(): return int(age_raw)
    try:
        match = re.search(r'(\d{4})년?', str(age_raw))
        if match: return int(match.group(1))
        cleaned = str(age_raw).strip()[:4]
        if cleaned.isdigit(): return int(cleaned)
    except ValueError:
        pass
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
    """리스트 값들의 분포(%) 계산"""
    if not values: return {}
    total = len(values)
    counts = Counter(values)
    return {k: round((v / total) * 100, 1) for k, v in counts.items()}

def extract_answer_from_template(field_name: str, sentence: str) -> str:
    """문장형 데이터에서 '핵심 답변'만 추출"""
    if not sentence: return ""
    
    if field_name == "ott_count":
        match = re.search(r'(\d+개|이용 안 함|없음)', sentence)
        if match: return match.group(1)
    elif field_name == "skincare_spending":
        match = re.search(r'(\d+만\s*원|\d+~\d+만\s*원|\d+원)', sentence)
        if match: return match.group(1)

    template = QPOLL_ANSWER_TEMPLATES.get(field_name)
    if template:
        try:
            pattern_str = re.escape(template)
            pattern_str = pattern_str.replace(re.escape("{answer_str}"), r"(.*?)")
            pattern_str = pattern_str.replace(r"\(이\)다", r"(?:이)?다")
            pattern_str = pattern_str.replace(r"\(으\)로", r"(?:으)?로")
            pattern_str = pattern_str.replace(r"\(가\)", r"(?:가)?")
            pattern_str = pattern_str.replace(r"\ ", r"\s*")

            match = re.search(pattern_str, sentence)
            if match:
                return clean_label(match.group(1), 20)
        except Exception:
            pass

    return clean_label(sentence, 30)

def get_negative_patterns(field_name: str) -> List[str]:
    patterns = COMMON_NEGATIVE_PATTERNS.copy()
    if field_name in SPECIFIC_NEGATIVE_PATTERNS:
        patterns.extend(SPECIFIC_NEGATIVE_PATTERNS[field_name])
    return patterns

@lru_cache(maxsize=512)
def get_field_mapping(keyword: str) -> Optional[Dict[str, Any]]:
    search_keyword = keyword.lower().strip()
    for pattern, mapping_info in KEYWORD_MAPPINGS:
        result_info = mapping_info.copy()
        if isinstance(pattern, re.Pattern):
            if pattern.search(search_keyword):
                return result_info
        elif isinstance(pattern, str):
            if pattern.lower() in search_keyword:
                return result_info
    return {"field": "unknown", "description": keyword, "type": "unknown"}

def find_related_fields(query: str) -> List[str]:
    related_fields = set()
    for field_key, field_desc in FIELD_NAME_MAP.items():
        query_words = query.split()
        for word in query_words:
            if len(word) >= 2 and word in field_desc:
                related_fields.add(field_key)
    
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
    all_fields_str = ""
    valid_columns = []
    for eng, kor in FIELD_NAME_MAP.items():
        all_fields_str += f"- {eng}: {kor}\n"
        valid_columns.append(eng)
    for eng, text in QPOLL_FIELD_TO_TEXT.items():
        if eng not in FIELD_NAME_MAP:
            all_fields_str += f"- {eng}: {text}\n"
            valid_columns.append(eng)

    logging.info(f"🔍 동적 컬럼 탐색 시작: '{question}'")
    found_columns = extract_relevant_columns_via_llm(question, all_fields_str)
    final_columns = [col for col in found_columns if col in valid_columns]
    if '소득' in question and 'income_personal_monthly' not in final_columns:
        final_columns.append('income_personal_monthly')
    logging.info(f"✅ 매핑 완료: {final_columns}")
    return final_columns

def filter_merged_panels(panels_data: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
    """
    병합 패널 필터링 (개선된 유연한 비교 로직)
    """
    if not panels_data or not filters:
        return panels_data

    filtered_list = []
    current_year = datetime.now().year
    
    # 1. 정규화 맵 구성 (VALUE_TRANSLATION_MAP 활용)
    VALUE_NORMALIZATION = {}
    for field, mapping in VALUE_TRANSLATION_MAP.items():
        for k, v in mapping.items():
            if not isinstance(v, list):
                VALUE_NORMALIZATION[k] = v

    # 기본 매핑 추가
    BASE_MAPPING = {
        'Female': '여성', 'F': '여성', 'Woman': '여성', '여': '여성',
        'Male': '남성', 'M': '남성', 'Man': '남성', '남': '남성',
        'Married': '기혼', 'Single': '미혼', 'Unmarried': '미혼'
    }
    VALUE_NORMALIZATION.update(BASE_MAPPING)

    logging.info(f"🔍 [Filter Debug] 필터링 시작 (대상: {len(panels_data)}명, 조건: {filters})")
    dropped_count = 0 

    for panel in panels_data:
        is_match = True
        panel_id = panel.get('panel_id', 'unknown')
        drop_reason = "" 

        for key, condition in filters.items():
            panel_value = None
            
            # [Case 1] 지역 (우선순위 적용)
            if key == 'region':
                panel_value = panel.get('region') or panel.get('region_major')
            
            # [Case 2] 나이 범위
            elif key == 'age_range' and isinstance(condition, list) and len(condition) == 2:
                birth_year = panel.get('birth_year')
                if birth_year:
                    try:
                        b_year = int(str(birth_year).split('.')[0])
                        age = current_year - b_year
                        if not (condition[0] <= age <= condition[1]):
                            is_match = False
                            drop_reason = f"나이 불일치 (생년:{birth_year}->나이:{age}, 조건:{condition})"
                            break
                        continue 
                    except:
                        pass
                # 나이 정보가 없으면 SQL 필터 결과를 신뢰하고 통과
                continue 

            # [Case 3] 일반 필드 값 가져오기
            else:
                panel_value = panel.get(key)

            # --- 값 비교 (유연한 로직) ---
            str_val = str(panel_value).strip() if panel_value is not None else ""
            
            # 1. 데이터 정규화 시도
            norm_val = VALUE_NORMALIZATION.get(str_val, str_val)
            if norm_val == str_val and str_val.capitalize() in VALUE_NORMALIZATION:
                norm_val = VALUE_NORMALIZATION[str_val.capitalize()]

            # 2. 숫자 추출 (숫자형 필드 비교용 - family_size, children_count 등)
            numeric_val = None
            # key가 family_size나 children_count인 경우에만 숫자 추출 시도
            if key in ['family_size', 'children_count'] or str_val.isdigit() or (str_val and str_val[:-1].isdigit()):
                 temp_numeric = re.sub(r'[^0-9]', '', str_val)
                 if temp_numeric:
                     numeric_val = int(temp_numeric)

            if condition:
                # 조건을 리스트로 통일하여 처리
                cond_list = condition if isinstance(condition, list) else [condition]
                match_found = False
                
                for cond_item in cond_list:
                    raw_cond = str(cond_item)
                    norm_cond = VALUE_NORMALIZATION.get(raw_cond, raw_cond)
                    
                    # 조건값에서도 숫자 추출
                    numeric_cond = None
                    temp_cond_numeric = re.sub(r'[^0-9]', '', raw_cond)
                    if temp_cond_numeric:
                        numeric_cond = int(temp_cond_numeric)

                    # 비교 로직 1: 문자열 포함 관계 (원본 및 정규화된 값 양방향 확인)
                    if (raw_cond in str_val) or (norm_cond in str_val) or \
                       (str_val in raw_cond) or (norm_val in raw_cond):
                        match_found = True
                        break
                    
                    # 비교 로직 2: 정규화된 값 간의 완전 일치
                    if str_val == raw_cond or str_val == norm_cond or norm_val == raw_cond or norm_val == norm_cond:
                        match_found = True
                        break
                    
                    # 비교 로직 3: 숫자값 비교 (가족 수, 자녀 수 등)
                    if numeric_val is not None and numeric_cond is not None and numeric_val == numeric_cond:
                        match_found = True
                        break
                
                if not match_found:
                    is_match = False
                    drop_reason = f"값 불일치 (키:{key}, 값:{panel_value}({norm_val}), 조건:{condition})"
                    break
        
        if is_match:
            filtered_list.append(panel)
        else:
            dropped_count += 1
            if dropped_count <= 5:
                 logging.info(f"❌ [Filter Debug] ID({panel_id}) 탈락: {drop_reason}")

    logging.info(f"✅ [Filter Debug] 최종 결과: {len(filtered_list)}명 (총 {len(panels_data)}명 중)")
    return filtered_list