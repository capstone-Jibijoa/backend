import re
from typing import Any, List, Dict
from collections import Counter
from datetime import datetime

def clean_label(text: Any, max_length: int = 25) -> str:
    """
    라벨 정제 함수: 특수문자 및 괄호 내용 제거
    """
    if not text: return ""
    text_str = str(text)
    # 괄호와 그 안의 내용 제거 (예: "사과(부사)" -> "사과")
    cleaned = re.sub(r'\([^)]*\)', '', text_str).strip()
    # 공백 정리
    cleaned = " ".join(cleaned.split())
    
    if len(cleaned) > max_length:
        return cleaned[:max_length] + ".."
    return cleaned

def truncate_text(value: Any, max_length: int = 30) -> str:
    """
    긴 텍스트를 잘라서 '...'을 붙여줍니다.
    """
    if value is None: return ""
    if isinstance(value, list):
        text = ", ".join(map(str, value))
    else:
        text = str(value)
        
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def get_age_group(birth_year: Any) -> str:
    """
    생년(birth_year)을 입력받아 한국식 연령대(예: '20대')를 반환합니다.
    """
    if not birth_year:
        return "알 수 없음"
    
    try:
        # 문자열이나 숫자가 들어올 수 있으므로 처리
        b_year = int(str(birth_year).split('.')[0]) # 1990.0 같은 경우 처리
        current_year = datetime.now().year
        age = current_year - b_year + 1 # 한국 나이 계산 (만 나이 필요 시 +1 제거)
        
        if age < 20: return "10대 이하"
        elif age < 30: return "20대"
        elif age < 40: return "30대"
        elif age < 50: return "40대"
        elif age < 60: return "50대"
        else: return "60대 이상"
    except (ValueError, TypeError):
        return "알 수 없음"

def extract_answer_from_template(field_name: str, sentence: str) -> str:
    """
    문장형 데이터에서 핵심 답변만 추출 (템플릿 매칭이 필요할 경우 확장 가능)
    """
    if not sentence: return ""
    
    # 간단한 정제 로직 (필요 시 정규식 패턴 추가 가능)
    cleaned = re.sub(r'\([^)]*\)', '', str(sentence)).strip()
    return truncate_text(cleaned, 30)