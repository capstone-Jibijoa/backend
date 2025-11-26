import re
from typing import Any, List, Dict
from collections import Counter
from datetime import datetime
import json
from app.constants.mapping import QPOLL_ANSWER_TEMPLATES

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
    """
    [복구된 핵심 로직] 문장형 데이터에서 '핵심 답변'만 추출
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