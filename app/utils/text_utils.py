import re
from typing import Any, Optional

try:
    from app.constants.mapping import QPOLL_ANSWER_TEMPLATES
except ImportError:
    from app.constants import QPOLL_ANSWER_TEMPLATES

def truncate_text(value: Any, max_length: int = 30) -> str:
    """
    텍스트가 너무 길 경우 잘라서 '...'을 붙여 반환합니다.
    리스트인 경우 쉼표로 연결합니다.
    """
    if value is None:
        return ""
    
    if isinstance(value, list):
        text = ", ".join(map(str, value))
    else:
        text = str(value)
        
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def clean_label(text: Any, max_length: int = 25) -> str:
    """
    라벨 정제 함수 (insights.py의 _clean_label 대체)
    괄호 안의 내용을 제거하고 공백을 정리합니다.
    """
    if not text: 
        return ""
    
    text_str = str(text)
    # 괄호와 괄호 안의 내용 제거
    cleaned = re.sub(r'\([^)]*\)', '', text_str).strip()
    # 중복 공백 제거
    cleaned = " ".join(cleaned.split())
    
    if len(cleaned) > max_length:
        return cleaned[:max_length] + ".."
    return cleaned

def extract_answer_from_template(field_name: str, sentence: str) -> str:
    """
    문장형 답변에서 핵심 키워드만 추출합니다.
    (기존 main.py의 함수와 insights.py의 _extract_core_value 통합)
    """
    if not sentence:
        return ""

    # 1. 특정 필드에 대한 하드코딩된 추출 규칙
    if field_name == "ott_count":
        match = re.search(r'(\d+개|이용 안 함|없음)', sentence)
        if match: return match.group(1)
    elif field_name == "skincare_spending":
        match = re.search(r'(\d+만\s*원|\d+~\d+만\s*원|\d+원)', sentence)
        if match: return match.group(1)

    # 2. 템플릿 기반 추출
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
            pattern_str = pattern_str.replace(r"\ ", r"\s*")

            match = re.search(pattern_str, sentence)
            if match:
                extracted = match.group(1)
                # 추출 후 괄호 제거 등 정제
                return clean_label(extracted, 20)
        except Exception:
            pass

    # 3. 매칭 실패 시 기본 정제 후 반환
    return clean_label(sentence, 30)