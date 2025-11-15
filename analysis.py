import os
import logging
import re 
from typing import List, Dict, Any, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client.models import Filter, FieldCondition, MatchValue

from utils import (
    extract_field_values,
    calculate_distribution,
    find_top_category,
    FIELD_NAME_MAP,
    WELCOME_OBJECTIVE_FIELDS,
    get_panels_data_from_db 
)
from db import get_db_connection_context, get_qdrant_client

# Q-Poll 질문 ID가 아닌, 질문 원문과 영어 키워드만 연결
QPOLL_FIELD_TO_TEXT = {
    "physical_activity": "여러분은 평소 체력 관리를 위해 어떤 활동을 하고 계신가요? 모두 선택해주세요.",
    "ott_count": "여러분이 현재 이용 중인 OTT 서비스는 몇 개인가요?",
    "traditional_market_freq": "여러분은 전통시장을 얼마나 자주 방문하시나요?",
    "lunar_new_year_gift_pref": "여러분이 가장 선호하는 설 선물 유형은 무엇인가요?",
    "elementary_winter_memories": "초등학생 시절 겨울방학 때 가장 기억에 남는 일은 무엇인가요?",
    "pet_experience": "여러분은 반려동물을 키우는 중이시거나 혹은 키워보신 적이 있으신가요?",
    "moving_stress_factor": "여러분은 이사할 때 가장 스트레스 받는 부분은 어떤걸까요?",
    "happiest_self_spending": "여러분은 본인을 위해 소비하는 것 중 가장 기분 좋아지는 소비는 무엇인가요?",
    "most_used_app": "여러분은 요즘 가장 많이 사용하는 앱은 무엇인가요?",
    "stress_situation": "다음 중 가장 스트레스를 많이 느끼는 상황은 무엇인가요?",
    "stress_relief_method": "스트레스를 해소하는 방법으로 주로 사용하는 것은 무엇인가요?",
    "skin_satisfaction": "현재 본인의 피부 상태에 얼마나 만족하시나요?",
    "skincare_spending": "한 달 기준으로 스킨케어 제품에 평균적으로 얼마나 소비하시나요?",
    "skincare_purchase_factor": "스킨케어 제품을 구매할 때 가장 중요하게 고려하는 요소는 무엇인가요?",
    "ai_chatbot_used": "여러분이 사용해 본 AI 챗봇 서비스는 무엇인가요? 모두 선택해주세요.",
    "ai_chatbot_main": "사용해 본 AI 챗봇 서비스 중 주로 사용하는 것은 무엇인가요?",
    "ai_chatbot_purpose": "AI 챗봇 서비스를 주로 어떤 용도로 활용하셨거나, 앞으로 활용하고 싶으신가요?",
    "ai_chatbot_sentiment": "다음 두 서비스 중, 어느 서비스에 더 호감이 가나요? 현재 사용 여부는 고려하지 않고 응답해 주세요.",
    "overseas_travel_pref": "여러분은 올해 해외여행을 간다면 어디로 가고 싶나요? 모두 선택해주세요",
    "fast_delivery_usage": "빠른 배송(당일·새벽·직진 배송) 서비스를 주로 어떤 제품을 구매할 때 이용하시나요?",
    "summer_worry": "여러분은 다가오는 여름철 가장 걱정되는 점이 무엇인가요?",
    "unused_item_disposal": "여러분은 버리기 아까운 물건이 있을 때, 주로 어떻게 하시나요?",
    "alarm_setting_style": "여러분은 아침에 기상하기 위해 어떤 방식으로 알람을 설정해두시나요?",
    "eating_alone_frequency": "여러분은 외부 식당에서 혼자 식사하는 빈도는 어느 정도인가요?",
    "happy_old_age_condition": "여러분이 가장 중요하다고 생각하는 행복한 노년의 조건은 무엇인가요?",
    "sweat_discomfort": "여름철 땀 때문에 겪는 불편함은 어떤 것이 있는지 모두 선택해주세요.",
    "most_effective_diet": "여러분이 지금까지 해본 다이어트 중 가장 효과 있었던 방법은 무엇인가요?",
    "late_night_snack_method": "여러분은 야식을 먹을 때 보통 어떤 방법으로 드시나요?",
    "favorite_summer_snack": "여러분의 여름철 최애 간식은 무엇인가요?",
    "recent_major_spending": "여러분은 최근 가장 지출을 많이 한 곳은 어디입니까?",
    "ai_service_usage_area": "여러분은 요즘 어떤 분야에서 AI 서비스를 활용하고 계신가요?",
    "minimalist_maximalist": "여러분은 본인을 미니멀리스트와 맥시멀리스트 중 어디에 더 가깝다고 생각하시나요?",
    "travel_planning_style": "어려분은 여행갈 때 어떤 스타일에 더 가까우신가요?",
    "plastic_bag_reduction_effort": "평소 일회용 비닐봉투 사용을 줄이기 위해 어떤 노력을 하고 계신가요?",
    "point_benefit_attention": "여러분은 할인, 캐시백, 멤버십 등 포인트 적립 혜택을 얼마나 신경 쓰시나요?",
    "chocolate_consumption_time": "여러분은 초콜릿을 주로 언제 드시나요?",
    "personal_info_protection_habit": "여러분은 평소 개인정보보호를 위해 어떤 습관이 있으신가요?",
    "summer_fashion_must_have": "여러분이 절대 포기할 수 없는 여름 패션 필수템은 무엇인가요?",
    "no_umbrella_reaction": "갑작스런 비로 우산이 없을 때 여러분은 어떻게 하시나요?",
    "most_saved_photo_type": "여러분의 휴대폰 갤러리에 가장 많이 저장되어져 있는 사진은 무엇인가요?",
    "favorite_summer_water_spot": "여러분이 여름철 물놀이 장소로 가장 선호하는 곳은 어디입니까?",
}

# 1. 정적 매핑 규칙 (Python 코드로 관리)
FIELD_MAPPING_RULES = [
    # --- type: "filter" (객관식 필터용 - Regex 패턴은 유지) ---
    (re.compile(r'^\d{2}대$'), 
     {"field": "birth_year", "description": "연령대", "type": "filter"}),
    (re.compile(r'^\d{2}~\d{2}대$'), 
     {"field": "birth_year", "description": "연령대", "type": "filter"}),
    (re.compile(r'젊은층|청년|MZ세대'), 
     {"field": "birth_year", "description": "연령대", "type": "filter"}),
    ("20대", {"field": "birth_year", "description": "연령대", "type": "filter"}),
    ("30대", {"field": "birth_year", "description": "연령대", "type": "filter"}),
    ("40대", {"field": "birth_year", "description": "연령대", "type": "filter"}),
    ("50대", {"field": "birth_year", "description": "연령대", "type": "filter"}),
    ("60대 이상", {"field": "birth_year", "description": "연령대", "type": "filter"}),
    
    (re.compile(r'^(서울|경기|부산|인천|대구|광주|대전|울산|세종|강원|충북|충남|전북|전남|경북|경남|제주)(특별)?(자?치)?(시|도|광역)?$', re.IGNORECASE), 
     {"field": "region_major", "description": "거주 지역", "type": "filter"}),
    
    (re.compile(r'.*(시|구|군)$'), 
     {"field": "region_minor", "description": "세부 거주 지역", "type": "filter"}),

    (re.compile(r'^(남|남자|남성)$', re.IGNORECASE), 
     {"field": "gender", "description": "성별", "type": "filter"}),
    (re.compile(r'^(여|여자|여성)$', re.IGNORECASE), 
     {"field": "gender", "description": "성별", "type": "filter"}),
    
    # [수정] 문자열 패턴을 소문자로 변경
    ("미혼", {"field": "marital_status", "description": "결혼 여부", "type": "filter"}),
    ("기혼", {"field": "marital_status", "description": "결혼 여부", "type": "filter"}),
    ("싱글", {"field": "marital_status", "description": "결혼 여부", "type": "filter"}),

    (re.compile(r'^(\d+인|가족\s*\d+명)$'), 
     {"field": "family_size", "description": "가족 수", "type": "filter"}),
    (re.compile(r'혼자|1인\s*가구|1인가구'), 
     {"field": "family_size", "description": "가족 수", "type": "filter"}),
    (re.compile(r'자녀\s*\d+명|다자녀'), 
     {"field": "children_count", "description": "자녀수", "type": "filter"}),
    (re.compile(r'.*가족\s*수.*|.*가구원\s*수.*|.*가구\s*\d+명.*|.*가족\s*\d+명.*', re.IGNORECASE), 
     {"field": "family_size", "description": "가족 수", "type": "filter"}),

    ("고졸", {"field": "education_level", "description": "최종학력", "type": "filter"}),
    ("대졸", {"field": "education_level", "description": "최종학력", "type": "filter"}),
    ("대학원", {"field": "education_level", "description": "최종학력", "type": "filter"}),
    
    # 7. job_duty_raw / job_title_raw (직무/직업) - String Pattern
    ("직장인", {"field": "job_duty_raw", "description": "직무", "type": "filter"}),
    ("학생", {"field": "job_title_raw", "description": "직업", "type": "filter"}),
    ("사무직", {"field": "job_duty_raw", "description": "직무", "type": "filter"}),
    ("마케팅", {"field": "job_duty_raw", "description": "직무", "type": "filter"}),
    ("IT", {"field": "job_duty_raw", "description": "직무", "type": "filter"}),

    # [수정] 개인 소득 관련 키워드는 income_personal_monthly에 매핑
    (re.compile(r'월소득|월\s*소득|개인소득|본인\s*소득'), 
    {"field": "income_personal_monthly", "description": "월소득(개인)", "type": "filter"}),
    (re.compile(r'고소득|저소득|중산층'), 
    {"field": "income_personal_monthly", "description": "월소득(개인)", "type": "filter"}),

    # [신규] 가구 소득 관련 키워드는 income_household_monthly에 매핑
    (re.compile(r'가구소득|가족\s*소득|가정\s*소득'), 
    {"field": "income_household_monthly", "description": "월소득(가구)", "type": "filter"}),

    ("아이폰", {"field": "phone_brand_raw", "description": "휴대폰 브랜드", "type": "filter"}),
    ("갤럭시", {"field": "phone_brand_raw", "description": "휴대폰 브랜드", "type": "filter"}),
    ("애플", {"field": "phone_brand_raw", "description": "휴대폰 브랜드", "type": "filter"}),
    ("삼성폰", {"field": "phone_brand_raw", "description": "휴대폰 브랜드", "type": "filter"}),
    ("LG", {"field": "phone_brand_raw", "description": "휴대폰 브랜드", "type": "filter"}),

    # --- 아이폰 시리즈 ---
    ("아이폰 15 pro 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("아이폰 15 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("아이폰 14 pro 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("아이폰 14 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("아이폰 13 pro 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("아이폰 13/13mini", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("아이폰 12 pro 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("아이폰 12/12mini", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("아이폰 11 pro 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("아이폰 11", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("아이폰 xs/xs max", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("아이폰 x", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("아이폰 se", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("기타 아이폰 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),

    # --- 갤럭시 시리즈 ---
    ("갤럭시 z fold 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("갤럭시 z filp 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("갤럭시 s23 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("갤럭시 s22 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("갤럭시 s21 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    # Note: S20 시리즈는 목록에 없지만, S21~S23 패턴을 고려하여 추가하지 않음 (목록에 S20이 있다면 추가 필요)
    ("갤럭시 s20 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}), 
    ("갤럭시 a 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("갤럭시 노트 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("갤럭시 m 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("기타 갤럭시 스마트폰", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),

    # --- LG 시리즈 ---
    ("lg 옵티머스 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("lg g pro", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("lg g flex", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("lg v 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("lg q 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    ("lg x 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    # Note: LG 기타 스마트폰은 목록에 있으므로 아래와 같이 처리
    ("lg 기타 스마트폰", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),

    # --- 포괄적인 키워드 (시리즈만 검색하는 경우) ---
    (re.compile(r'(아이폰|iphone)\s*(15|14|13|12|11|x|se)', re.IGNORECASE),
    {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    (re.compile(r'갤럭시\s*(s|z|a|m|노트)\s*\d+', re.IGNORECASE), 
    {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),

    # 10. car_ownership / car_manufacturer_raw (차량) - String Pattern
    ("차량보유", {"field": "car_ownership", "description": "차량 보유 여부", "type": "filter"}),
    ("차없음", {"field": "car_ownership", "description": "차량 보유 여부", "type": "filter"}),
    
    # --- 국내 브랜드 ---
    ("기아", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("르노삼성", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("쌍용", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("쉐보레", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("한국gm", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("현대", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("제네시스", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),

    # --- 해외/수입차 브랜드 ---
    ("아우디", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("벤틀리", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("bmw", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("포드", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("혼다", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("인피니티", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("재규어", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("지프", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("랜드로버", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("렉서스", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("링컨", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("메르세데스-벤츠", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("bmw 미니", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("닛산", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("포르쉐", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("롤스로이스", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("테슬라", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("토요타", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("볼보", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("폭스바겐", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),

    # --- 상용차 및 기타 ---
    ("만", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("스카니아", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("포톤", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("이스트", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("이베코", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),

    # --- 포괄적인 키워드 (이미지 목록에 없지만 일반적인 별칭) ---
    ("벤츠", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("미니", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("현대차", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("기아차", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),

    ("그랜저", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("쏘나타", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("아반떼", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("캐스퍼", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("코나", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("투싼", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("싼타페", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("팰리세이드", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("아이오닉 5", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("아이오닉 6", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("넥쏘", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("포터2", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("봉고3", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("레이", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("모닝", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("K3", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("K5", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("K8", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("스팅어", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("셀토스", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("스포티지", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("쏘렌토", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("카니발", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("모하비", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("EV6", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("니로", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("제네시스 G70", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("제네시스 G80", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("제네시스 G90", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("제네시스 GV70", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("제네시스 GV80", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),

    # --- 르노삼성 ---
    ("XM3", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("SM6", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),

    # --- KG모빌리티 (구 쌍용) ---
    ("토레스", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("티볼리", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("코란도", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("렉스턴", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),

    # --- 쉐보레 ---
    ("트레일블레이저", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("트랙스", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("말리부", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("트래버스", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("타호", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),

    # --- 수입차 (BMW, 벤츠, 아우디 등) ---
    ("BMW 3시리즈", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("BMW 5시리즈", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("BMW X3", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("BMW X5", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("벤츠 C-클래스", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("벤츠 E-클래스", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("아우디 A4", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("아우디 A6", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("테슬라 Model 3", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("테슬라 Model Y", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),

    # --- 기타 및 포괄적 ---
    ("기타 국산차", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("기타 수입차", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'SUV|세단|트럭|승합', re.IGNORECASE), 
    {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    
    ("흡연", {"field": "smoking_experience", "description": "흡연 여부", "type": "filter"}),
    ("비흡연", {"field": "smoking_experience", "description": "흡연 여부", "type": "filter"}),

    # 11.5. smoking_brand (담배 종류/브랜드) - String Pattern

    ("레종", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("에쎄", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("보헴", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("아프리카", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("더원", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("시즌", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("아이스볼트 gt", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("디스플러스", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("디스", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("한라산", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("라일락", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("심플", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("타임", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("88리턴즈", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("말보로", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("팔리아멘트", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("메비우스", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("던힐", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("라크", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("카멜", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("다비도프", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("하모니", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("럭키스트라이크", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("버지니아 s", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("블랙데빌", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("켄트", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("클라우드 나인", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("토니노 람보르기니", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("하비스트", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),

    # 11.7. e_cigarette_experience (전자 담배 이용 경험) - String Pattern

    ("아이코스", {"field": "e_cigarette_experience", "description": "전자 담배 이용 경험", "type": "filter"}),
    ("릴", {"field": "e_cigarette_experience", "description": "전자 담배 이용 경험", "type": "filter"}),
    ("글로", {"field": "e_cigarette_experience", "description": "전자 담배 이용 경험", "type": "filter"}),
    ("차이코스", {"field": "e_cigarette_experience", "description": "전자 담배 이용 경험", "type": "filter"}),
    ("차이코스 (cqs)", {"field": "e_cigarette_experience", "description": "전자 담배 이용 경험", "type": "filter"}),

    # --- 포괄적인 키워드 (이미 기존에 추가되었을 수 있지만 재확인) ---
    ("전자담배", {"field": "e_cigarette_experience", "description": "전자 담배 이용 경험", "type": "filter"}),
    
    ("음주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("술", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("금주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("소주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("맥주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("저도주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("막걸리", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("양주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("와인", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("과일칵테일주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("일본청주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("사케", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("최근 1년 이내 술을 마시지 않음", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),

    # --- 포괄적인 키워드 (별칭) ---
    ("위스키", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("보드카", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("데킬라", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("진", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    
    ("차량보유", {"field": "car_ownership", "description": "차량 보유", "type": "filter"}),
    ("차없음", {"field": "car_ownership", "description": "차량 보유", "type": "filter"}),
    
    # --- type: "qpoll" (Q-Poll 질문용, 전체 매핑) ---
    # [수정] 모든 Q-Poll 키워드 문자열 패턴을 소문자로 변경
    ("체력 관리", {"field": "physical_activity", "description": QPOLL_FIELD_TO_TEXT["physical_activity"], "type": "qpoll"}),
    ("운동 활동", {"field": "physical_activity", "description": QPOLL_FIELD_TO_TEXT["physical_activity"], "type": "qpoll"}),
    
    ("ott", {"field": "ott_count", "description": QPOLL_FIELD_TO_TEXT["ott_count"], "type": "qpoll"}),
    ("스트리밍 서비스", {"field": "ott_count", "description": QPOLL_FIELD_TO_TEXT["ott_count"], "type": "qpoll"}),
    
    ("전통시장", {"field": "traditional_market_freq", "description": QPOLL_FIELD_TO_TEXT["traditional_market_freq"], "type": "qpoll"}),
    
    ("설 선물", {"field": "lunar_new_year_gift_pref", "description": QPOLL_FIELD_TO_TEXT["lunar_new_year_gift_pref"], "type": "qpoll"}),
    ("선물 선호도", {"field": "lunar_new_year_gift_pref", "description": QPOLL_FIELD_TO_TEXT["lunar_new_year_gift_pref"], "type": "qpoll"}),
    
    ("겨울방학", {"field": "elementary_winter_memories", "description": QPOLL_FIELD_TO_TEXT["elementary_winter_memories"], "type": "qpoll"}),
    
    ("반려동물", {"field": "pet_experience", "description": QPOLL_FIELD_TO_TEXT["pet_experience"], "type": "qpoll"}),
    
    ("이사 스트레스", {"field": "moving_stress_factor", "description": QPOLL_FIELD_TO_TEXT["moving_stress_factor"], "type": "qpoll"}),
    
    ("가장 기분 좋아지는 소비", {"field": "happiest_self_spending", "description": QPOLL_FIELD_TO_TEXT["happiest_self_spending"], "type": "qpoll"}),
    ("나를 위한 소비", {"field": "happiest_self_spending", "description": QPOLL_FIELD_TO_TEXT["happiest_self_spending"], "type": "qpoll"}),
    
    ("사용하는 앱", {"field": "most_used_app", "description": QPOLL_FIELD_TO_TEXT["most_used_app"], "type": "qpoll"}),
    
    ("스트레스 상황", {"field": "stress_situation", "description": QPOLL_FIELD_TO_TEXT["stress_situation"], "type": "qpoll"}),
    ("스트레스 해소", {"field": "stress_relief_method", "description": QPOLL_FIELD_TO_TEXT["stress_relief_method"], "type": "qpoll"}),
    
    ("피부 만족도", {"field": "skin_satisfaction", "description": QPOLL_FIELD_TO_TEXT["skin_satisfaction"], "type": "qpoll"}),
    ("스킨케어 소비", {"field": "skincare_spending", "description": QPOLL_FIELD_TO_TEXT["skincare_spending"], "type": "qpoll"}),
    ("스킨케어 고려 요소", {"field": "skincare_purchase_factor", "description": QPOLL_FIELD_TO_TEXT["skincare_purchase_factor"], "type": "qpoll"}),
    
    ("ai 챗봇 사용 경험", {"field": "ai_chatbot_used", "description": QPOLL_FIELD_TO_TEXT["ai_chatbot_used"], "type": "qpoll"}),
    ("주로 사용하는 ai 챗봇", {"field": "ai_chatbot_main", "description": QPOLL_FIELD_TO_TEXT["ai_chatbot_main"], "type": "qpoll"}),
    ("ai 챗봇 활용 용도", {"field": "ai_chatbot_purpose", "description": QPOLL_FIELD_TO_TEXT["ai_chatbot_purpose"], "type": "qpoll"}),
    ("ai 챗봇 호감도", {"field": "ai_chatbot_sentiment", "description": QPOLL_FIELD_TO_TEXT["ai_chatbot_sentiment"], "type": "qpoll"}),
    
    ("해외여행 선호지", {"field": "overseas_travel_pref", "description": QPOLL_FIELD_TO_TEXT["overseas_travel_pref"], "type": "qpoll"}),
    
    ("빠른 배송", {"field": "fast_delivery_usage", "description": QPOLL_FIELD_TO_TEXT["fast_delivery_usage"], "type": "qpoll"}),
    
    ("여름철 걱정", {"field": "summer_worry", "description": QPOLL_FIELD_TO_TEXT["summer_worry"], "type": "qpoll"}),
    
    ("물건 처리", {"field": "unused_item_disposal", "description": QPOLL_FIELD_TO_TEXT["unused_item_disposal"], "type": "qpoll"}),
    ("업사이클링", {"field": "unused_item_disposal", "description": QPOLL_FIELD_TO_TEXT["unused_item_disposal"], "type": "qpoll"}),
    
    ("기상 알람", {"field": "alarm_setting_style", "description": QPOLL_FIELD_TO_TEXT["alarm_setting_style"], "type": "qpoll"}),
    
    ("혼밥", {"field": "eating_alone_frequency", "description": QPOLL_FIELD_TO_TEXT["eating_alone_frequency"], "type": "qpoll"}),
    
    ("행복한 노년", {"field": "happy_old_age_condition", "description": QPOLL_FIELD_TO_TEXT["happy_old_age_condition"], "type": "qpoll"}),
    
    ("여름 땀 불편함", {"field": "sweat_discomfort", "description": QPOLL_FIELD_TO_TEXT["sweat_discomfort"], "type": "qpoll"}),
    
    ("다이어트 방법", {"field": "most_effective_diet", "description": QPOLL_FIELD_TO_TEXT["most_effective_diet"], "type": "qpoll"}),
    
    ("야식", {"field": "late_night_snack_method", "description": QPOLL_FIELD_TO_TEXT["late_night_snack_method"], "type": "qpoll"}),
    
    ("여름철 간식", {"field": "favorite_summer_snack", "description": QPOLL_FIELD_TO_TEXT["favorite_summer_snack"], "type": "qpoll"}),
    
    ("최근 지출", {"field": "recent_major_spending", "description": QPOLL_FIELD_TO_TEXT["recent_major_spending"], "type": "qpoll"}),
    
    ("ai 서비스 활용 분야", {"field": "ai_service_usage_area", "description": QPOLL_FIELD_TO_TEXT["ai_service_usage_area"], "type": "qpoll"}),
    
    ("미니멀리스트", {"field": "minimalist_maximalist", "description": QPOLL_FIELD_TO_TEXT["minimalist_maximalist"], "type": "qpoll"}),
    ("맥시멀리스트", {"field": "minimalist_maximalist", "description": QPOLL_FIELD_TO_TEXT["minimalist_maximalist"], "type": "qpoll"}),
    
    ("여행 스타일", {"field": "travel_planning_style", "description": QPOLL_FIELD_TO_TEXT["travel_planning_style"], "type": "qpoll"}),
    
    ("비닐봉투 줄이기", {"field": "plastic_bag_reduction_effort", "description": QPOLL_FIELD_TO_TEXT["plastic_bag_reduction_effort"], "type": "qpoll"}),
    
    ("포인트 적립", {"field": "point_benefit_attention", "description": QPOLL_FIELD_TO_TEXT["point_benefit_attention"], "type": "qpoll"}),
    
    ("초콜릿", {"field": "chocolate_consumption_time", "description": QPOLL_FIELD_TO_TEXT["chocolate_consumption_time"], "type": "qpoll"}),
    
    ("개인정보보호", {"field": "personal_info_protection_habit", "description": QPOLL_FIELD_TO_TEXT["personal_info_protection_habit"], "type": "qpoll"}),
    
    ("여름 패션", {"field": "summer_fashion_must_have", "description": QPOLL_FIELD_TO_TEXT["summer_fashion_must_have"], "type": "qpoll"}),
    
    ("우산 없을 때", {"field": "no_umbrella_reaction", "description": QPOLL_FIELD_TO_TEXT["no_umbrella_reaction"], "type": "qpoll"}),
    
    ("갤러리 사진", {"field": "most_saved_photo_type", "description": QPOLL_FIELD_TO_TEXT["most_saved_photo_type"], "type": "qpoll"}),
    
    ("물놀이 장소", {"field": "favorite_summer_water_spot", "description": QPOLL_FIELD_TO_TEXT["favorite_summer_water_spot"], "type": "qpoll"}),
]

def get_field_mapping(keyword: str) -> Dict[str, str]:
    """
    [수정됨] 키워드를 받아 매핑되는 필드 정보와 "타입"을 반환합니다.
    필터 규칙에 없으면 'unknown'으로 반환합니다.
    """
    keyword_for_string_match = keyword.lower() 
    
    for pattern, mapping_info in FIELD_MAPPING_RULES:
        
        # [수정] 'type'을 명시적으로 확인 (기본값 'filter')
        rule_type = mapping_info.get("type", "filter")
            
        field = mapping_info["field"]
        description = mapping_info["description"] if rule_type == 'qpoll' else FIELD_NAME_MAP.get(field, mapping_info["description"])

        if isinstance(pattern, re.Pattern):
            if pattern.match(keyword): 
                return {"field": field, 
                        "description": description, 
                        "type": rule_type}
        elif isinstance(pattern, str):
            if pattern.lower() == keyword_for_string_match:
                return {"field": field, 
                        "description": description, 
                        "type": rule_type}
            
    # '간호직', 'OTT', 'it' 등 필터 규칙에 없는 모든 키워드는 'unknown'으로 처리
    # (주의: search.py의 파서가 'unknown' 타입을 'vector'로 해석해야 함)
    logging.warning(f" ⚠️  '{keyword}'에 대한 매핑 규칙 없음. 'unknown'(벡터)으로 처리.")
    return {"field": "unknown", "description": keyword, "type": "unknown"}


def get_field_distribution_from_db(field_name: str, limit: int = 10) -> Dict[str, float]:
    """
    PostgreSQL에서 직접 집계하여 필드 분포 조회 (전체 DB 대상)
    (캐시 롤백 버전)
    """
    try:
        with get_db_connection_context() as conn:
            if not conn:
                logging.error("DB 집계: 연결 실패")
                return {}
            
            cur = conn.cursor()
            
            if field_name == "birth_year":
                query = f"""
                    WITH age_groups AS (
                        SELECT 
                            CASE 
                                WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 20 THEN '10대'
                                WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 30 THEN '20대'
                                WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 40 THEN '30대'
                                WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 50 THEN '40대'
                                WHEN (date_part('year', CURRENT_DATE) - (structured_data->>'birth_year')::int) < 60 THEN '50대'
                                ELSE '60대 이상'
                            END as age_group
                        FROM welcome_meta2
                        WHERE structured_data->>'birth_year' IS NOT NULL
                            AND structured_data->>'birth_year' ~ '^[0-9]+$'
                    )
                    SELECT 
                        age_group,
                        COUNT(*) as count,
                        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
                    FROM age_groups
                    GROUP BY age_group
                    ORDER BY percentage DESC
                    LIMIT {limit}
                """
            else:
                query = f"""
                    SELECT 
                        structured_data->>'{field_name}' as value,
                        COUNT(*) as count,
                        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as percentage
                    FROM welcome_meta2
                    WHERE structured_data->>'{field_name}' IS NOT NULL
                        AND structured_data->>'{field_name}' != ''
                    GROUP BY structured_data->>'{field_name}'
                    ORDER BY percentage DESC
                    LIMIT {limit}
                """
            
            cur.execute(query)
            rows = cur.fetchall()
            
            distribution = {}
            for row in rows:
                value = row[0]
                percentage = float(row[2])
                if value and percentage > 0:
                    distribution[value] = percentage
            
            cur.close()
        
        logging.info(f"   📊 DB 집계 완료: {field_name} ({len(distribution)}개 카테고리)")
        return distribution
        
    except Exception as e:
        logging.error(f"   DB 집계 실패 ({field_name}): {e}", exc_info=True)
        return {}
    
def get_qpoll_distribution_from_db(qpoll_field: str, limit: int = 10) -> Dict[str, float]:
    """
    Qdrant Client를 사용하여 'qpoll_vectors_v2' 컬렉션에서 질문에 대한 응답 분포를 조회합니다.
    (PostgreSQL 대신 Qdrant API를 사용합니다.)
    """
    question_text = QPOLL_FIELD_TO_TEXT.get(qpoll_field)
    if not question_text:
        logging.error(f"Q-Poll DB 집계: '{qpoll_field}'에 해당하는 질문 원문을 찾을 수 없습니다.")
        return {}
    
    client = get_qdrant_client()
    if not client:
        logging.error("Q-Poll Qdrant 집계: Qdrant 클라이언트 연결 실패.")
        return {}
        
    try:
        COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_QPOLL_NAME", "qpoll_vectors_v2")
        
        # 1. 질문 텍스트로 필터 정의
        query_filter = Filter(
            must=[
                FieldCondition(key="question", match=MatchValue(value=question_text))
            ]
        )
        
        all_points = []
        next_offset = None
        
        # 2. 필터에 맞는 모든 포인트 스크롤 (전체 분포 집계를 위해 필요)
        while True:
            # 한 번에 1000개씩 스크롤
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=query_filter,
                limit=1000, 
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if next_offset is None:
                break
                
        total_count = len(all_points)
        
        if total_count == 0:
            return {}

        # 3. 'sentence' (응답) 필드 값의 분포 계산
        sentence_counts = Counter(p.payload.get("sentence") for p in all_points if p.payload and p.payload.get("sentence"))
        
        # 4. 백분율 계산 및 상위 N개 필터링
        distribution = {
            sentence: round((count / total_count) * 100, 1)
            for sentence, count in sentence_counts.most_common(limit)
        }
        
        logging.info(f"   📊 Q-Poll Qdrant 집계 완료: {qpoll_field} ({len(distribution)}개 카테고리)")
        return distribution
        
    except Exception as e:
        logging.error(f"   Q-Poll Qdrant 집계 실패: {e}", exc_info=True)
        return {}

def create_chart_data_optimized(
    keyword: str,
    field_name: str,
    korean_name: str,
    panels_data: List[Dict],
    use_full_db: bool = False,
    max_categories: int = 10
) -> Dict:
    """
    차트 데이터 생성 (최적화 버전)
    """
    # 전체 DB 기반 분석
    if use_full_db:
        logging.info(f"       → DB 집계로 '{field_name}' 분석 (최적화)")
        distribution = get_field_distribution_from_db(field_name, max_categories)
        
        if not distribution:
            return {
                "topic": korean_name,
                "description": f"'{keyword}' 관련 전체 데이터를 조회할 수 없습니다.",
                "ratio": "0.0%",
                "chart_data": []
            }
        
        top_category, top_ratio = find_top_category(distribution)
        description_prefix = f"전체 데이터 기준 '{korean_name}' 분석:"
        
        return {
            "topic": f"{korean_name} 분포",
            "description": f"{description_prefix} {top_ratio}%가 '{top_category}'입니다.",
            "ratio": f"{top_ratio}%",
            "chart_data": [{"label": korean_name, "values": distribution}]
        }
    
    # 검색 결과 기반 분석
    else:
        values = extract_field_values(panels_data, field_name)
        
        if not values:
            return { "topic": korean_name, "description": "데이터 부족", "ratio": "0.0%", "chart_data": [] }
        
        distribution = calculate_distribution(values)
        filtered_distribution = {k: v for k, v in distribution.items() if v > 0.0}
        
        if not filtered_distribution:
            return { "topic": korean_name, "description": "데이터 부족", "ratio": "0.0%", "chart_data": [] }
        
        # 상위 N개만 + 기타
        if len(filtered_distribution) > max_categories:
            sorted_items = sorted(filtered_distribution.items(), key=lambda x: x[1], reverse=True)
            top_items = dict(sorted_items[:max_categories - 1])
            other_sum = sum(v for k, v in sorted_items[max_categories - 1:])
            if other_sum > 0:
                top_items['기타'] = round(other_sum, 1)
            final_distribution = top_items
        else:
            final_distribution = filtered_distribution
        
        top_category, top_ratio = find_top_category(final_distribution)
        
        return {
            "topic": f"{korean_name} 분포",
            "description": f"'{keyword}' 검색 결과: {top_ratio}%가 '{top_category}'입니다.",
            "ratio": f"{top_ratio}%",
            "chart_data": [{
                "label": korean_name,
                "values": final_distribution
            }]
        }

# analysis.py (업데이트된 create_qpoll_chart_data 함수)

def create_qpoll_chart_data(
    qpoll_field: str,
    max_categories: int = 10
) -> Dict:
    """
    Q-Poll 데이터 기반으로 차트 데이터 생성 (Welcome과 동일한 구조 반환)
    """
    question_text = QPOLL_FIELD_TO_TEXT.get(qpoll_field, qpoll_field) 
    logging.info(f"       → Q-Poll Qdrant 집계로 '{qpoll_field}' 분석")
    
    # Qdrant 분포 조회
    distribution = get_qpoll_distribution_from_db(qpoll_field, max_categories)
    
    if not distribution:
        return {
            "topic": question_text,
            "description": f"'{question_text}' 관련 Q-Poll 데이터를 조회할 수 없습니다.",
            "ratio": "0.0%",
            "chart_data": []
        }
    
    final_distribution = distribution
    
    # 상위 N개만 + 기타로 필터링 (get_qpoll_distribution_from_db에서 이미 most_common(limit)로 처리됨)
    # 다만, Qdrant 스크롤/카운트가 정확한 전체 비율을 계산하지 못할 경우를 대비하여 여기서 다시 정규화하지 않음.
    
    top_category, top_ratio = find_top_category(final_distribution)
    
    # 복수 응답 여부를 질문 원문 텍스트의 키워드("모두 선택해주세요")로 확인
    is_array_type = "모두 선택해주세요" in question_text

    if is_array_type:
        description = f"Q-Poll 응답자 기준, 가장 많은 응답은 '{top_category}'로 {top_ratio}%입니다. (복수 응답 가능)"
    else:
        description = f"Q-Poll 응답자 기준, {top_ratio}%가 '{top_category}'입니다."
        
    return {
        "topic": question_text, 
        "description": description,
        "ratio": f"{top_ratio}%",
        "chart_data": [{
            "label": question_text,
            "values": final_distribution
        }]
    }

def create_crosstab_chart(
    panels_data: List[Dict],
    field1: str,  # 주축 (e.g., 'birth_year')
    field2: str,  # 세그먼트 (e.g., 'gender')
    field1_korean: str,
    field2_korean: str,
    max_categories: int = 5
) -> Dict:
    """
    교차 분석 차트 데이터를 생성합니다. (예: 연령대별 성별 분포)
    """
    logging.info(f"       → 교차 분석으로 '{field1}' vs '{field2}' 분석")
    from utils import get_age_group

    # 1. 두 필드에 대한 데이터 추출
    crosstab_data = {}
    for item in panels_data:
        val1 = item.get(field1)
        val2 = item.get(field2)

        if val1 is None or val2 is None:
            continue

        # 값 처리 (연령대 변환 등)
        key1 = get_age_group(val1) if field1 == 'birth_year' else str(val1)
        key2 = str(val2)

        if key1 not in crosstab_data:
            crosstab_data[key1] = []
        crosstab_data[key1].append(key2)

    if not crosstab_data:
        return {}

    # 2. 각 주축 카테고리별로 세그먼트 분포 계산
    chart_values = {}
    for key1, values2 in crosstab_data.items():
        distribution = calculate_distribution(values2)
        chart_values[key1] = distribution

    # 3. 주축 카테고리가 너무 많으면 상위 N개만 선택
    if len(chart_values) > max_categories:
        # 각 카테고리의 전체 개수를 기준으로 정렬
        sorted_keys = sorted(chart_values.keys(), key=lambda k: sum(crosstab_data[k].count(v) for v in set(crosstab_data[k])), reverse=True)
        chart_values = {k: chart_values[k] for k in sorted_keys[:max_categories]}

    if not chart_values:
        return {}

    return {
        "topic": f"{field1_korean}별 {field2_korean} 분포",
        "description": f"'{field1_korean}'에 따른 '{field2_korean}'의 상대적 분포를 보여줍니다.",
        "chart_type": "crosstab",
        "chart_data": [{"label": f"{field1_korean}별 {field2_korean}", "values": chart_values}]
    }


def _analyze_fields_in_parallel(panels_data: List[Dict], candidate_fields: List[Tuple[str, str]]) -> List[Dict]:
    """
    [리팩토링] panels_data를 한 번만 순회하여 모든 후보 필드의 값을 집계하고 분포를 계산합니다.
    """
    field_values = {field_name: [] for field_name, _ in candidate_fields}
    field_map = dict(candidate_fields)

    # 데이터를 한 번만 순회하여 모든 필드의 값을 추출
    for item in panels_data:
        for field_name in field_values.keys():
            value = item.get(field_name)
            if value is None:
                continue

            if field_name == "birth_year":
                from utils import get_age_group
                field_values[field_name].append(get_age_group(value))
            elif isinstance(value, list):
                field_values[field_name].extend(value)
            else:
                field_values[field_name].append(value)

    results = []
    for field_name, values in field_values.items():
        if not values:
            continue
        
        try:
            distribution = calculate_distribution(values)
            filtered_distribution = {k: v for k, v in distribution.items() if v > 0.0}
            if not filtered_distribution:
                continue

            results.append({
                "field": field_name,
                "korean_name": field_map[field_name],
                "distribution": filtered_distribution,
            })
        except Exception as e:
            logging.warning(f"   ⚠️  {field_name} 분포 계산 실패: {e}")

    return results


def find_high_ratio_fields_optimized(
    panels_data: List[Dict], 
    exclude_fields: List[str], 
    threshold: float = 50.0,
    max_charts: int = 3
) -> List[Dict]:
    """
    검색 결과(panels_data) 내에서 높은 비율을 차지하는 필드 찾기 (병렬 처리)
    """
    candidate_fields = []
    
    for field_name, korean_name in WELCOME_OBJECTIVE_FIELDS:
        if field_name not in exclude_fields:
            candidate_fields.append((field_name, korean_name))
    
    if not candidate_fields:
        return []
    
    logging.info(f"   🔍 {len(candidate_fields)}개 필드 병렬 분석 중... (제외 필드: {exclude_fields})")
    
    analysis_results = _analyze_fields_in_parallel(panels_data, candidate_fields)
    
    high_ratio_results = []
    for result in analysis_results:
        distribution = result['distribution']
        
        # 100% 단일 카테고리 스킵
        if len(distribution) == 1:
            top_category, top_ratio = find_top_category(distribution) 
            logging.info(f"   ⚠️  [{result['korean_name']}] 스킵: {top_category} {top_ratio}% (단일 카테고리 100%)")
            continue
        
        top_category, top_ratio = find_top_category(distribution)
        
        # 50% 이상이라는 1차 임계값 통과 시
        if top_ratio >= threshold:
        
            final_distribution = distribution
            if len(distribution) > 10:
                sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
                top_items = dict(sorted_items[:9])
                other_sum = sum(v for k, v in sorted_items[9:])
                if other_sum > 0:
                    top_items['기타'] = round(other_sum, 1)
                final_distribution = top_items
            
            high_ratio_results.append({
                "field": result['field'],
                "korean_name": result['korean_name'],
                "distribution": final_distribution,
                "top_category": top_category,
                "top_ratio": top_ratio
            })
    
    high_ratio_results.sort(key=lambda x: x["top_ratio"], reverse=True)
    
    return high_ratio_results[:max_charts]

def analyze_search_results_optimized(
    query: str,
    classified_keywords: dict,
    panel_id_list: List[str]
) -> Tuple[Dict, int]:
    """
    검색 결과 분석 (최적화 버전)
    """
    logging.info(f"📊 분석 시작 (최적화) - panel_id 수: {len(panel_id_list)}개")
    
    if not panel_id_list:
        return {"main_summary": "검색 결과가 없습니다.", "charts": []}, 200
    
    try:
        # 1단계: 패널 데이터 조회
        logging.info("   1단계: 패널 데이터 조회 (utils.py 사용)")
        panels_data = get_panels_data_from_db(panel_id_list)
        
        if not panels_data:
            return {"main_summary": "패널 데이터를 조회할 수 없습니다.", "charts": []}, 200
        
        logging.info(f"   ✅ {len(panels_data)}개 패널 데이터 조회 완료")
        
        # 2단계: ranked_keywords 추출 및 매핑
        # (주의: search.py에서 LLM 대신 규칙 기반 파서로 대체되었다면
        #       classified_keywords['ranked_keywords_raw']가 올바르게 전달되어야 함)
        raw_keywords = classified_keywords.get('ranked_keywords_raw', [])
        ranked_keywords = []
        search_used_fields = set()
        
        if raw_keywords:
            logging.info(f"   2a단계: (규칙 기반) 키워드 {raw_keywords} 매핑 시작")
            for i, keyword in enumerate(raw_keywords):
                
                mapping = get_field_mapping(keyword) 
                kw_type = mapping.get("type", "unknown")
                
                ranked_keywords.append({
                    "keyword": keyword, 
                    "field": mapping["field"],
                    "description": mapping["description"], 
                    "type": kw_type, # [수정] type 정보 저장
                    "priority": i + 1
                })
                
                # [수정] 'filter' 타입이고 'unknown'이 아닌 경우에만 '뻔한 필드'로 추가
                if kw_type == 'filter' and mapping["field"] != 'unknown':
                    search_used_fields.add(mapping["field"])
        
        if not ranked_keywords:
            logging.warning("   ⚠️  'ranked_keywords_raw' 없음. (Fallback 로직 실행)")
            # (Fallback 로직은 LLM을 사용하지 않으므로 수정이 필요할 수 있음)
            obj_keywords = classified_keywords.get('welcome_keywords', {}).get('objective', [])
            for i, kw in enumerate(obj_keywords[:5]):
                mapping = get_field_mapping(kw) # [수정] 수정된 함수 호출
                ranked_keywords.append({
                    'keyword': kw, 
                    'field': mapping["field"],
                    'description': mapping["description"], 
                    'type': mapping.get("type", "unknown"), # [수정] type 정보 저장
                    'priority': i + 1
                })
                if mapping["type"] == 'filter' and mapping["field"] != 'unknown': # [수정] type 체크
                    search_used_fields.add(mapping["field"])
            
        if not ranked_keywords:
            return { "main_summary": f"총 {len(panels_data)}명 조회, 분석할 키워드 없음.", "charts": [] }, 200
        
        ranked_keywords.sort(key=lambda x: x.get('priority', 999))
        logging.info(f"   ✅ 분석 키워드: {[k.get('keyword') for k in ranked_keywords]}")
        logging.info(f"   ✅ 검색 사용 필드 (뻔한 인사이트 제외용): {search_used_fields}")
        
        # 3단계: ranked_keywords 기반 차트 생성
        logging.info("   3단계: 주요 키워드 차트 생성 (DB 집계, 병렬)")
        charts = []
        used_fields = []
        objective_fields = set([f[0] for f in WELCOME_OBJECTIVE_FIELDS])
        
        # 1. 생성할 차트 작업 목록 정의
        chart_tasks = [] 
        chart_count = 0
        for kw_info in ranked_keywords:
            if chart_count >= 2: break
            
            field = kw_info.get('field', '')
            kw_type = kw_info.get('type', 'unknown') # [수정] type 가져오기
            
            if field in used_fields:
                continue

            if kw_type == 'filter':
                if field in objective_fields and field != 'unknown':
                    if panels_data: # Welcome 분석은 패널 데이터가 있을 때만 시도
                        chart_tasks.append({"type": "filter", "kw_info": kw_info})
                        used_fields.append(field)
                        chart_count += 1
            
            elif kw_type == 'qpoll':
                # Q-Poll 분석은 전체 DB 대상
                chart_tasks.append({"type": "qpoll", "kw_info": kw_info})
                used_fields.append(field)
                chart_count += 1

        # 2. ThreadPoolExecutor로 차트 생성 병렬 실행
        if chart_tasks:
            with ThreadPoolExecutor(max_workers=len(chart_tasks) or 1) as executor:
                
                def run_chart_creation(task):
                    kw_info = task["kw_info"]
                    field = kw_info.get('field', '')
                    korean_name = kw_info.get('description', field)
                    logging.info(f"   ⚡ [{korean_name}] 차트 DB 집계 스레드 시작 ({task['type']})...")
                    
                    if task["type"] == "filter":
                        return create_chart_data_optimized(
                            kw_info.get('keyword', ''), 
                            field, 
                            korean_name,
                            panels_data,
                            use_full_db=True
                        )
                    elif task["type"] == "qpoll":
                        return create_qpoll_chart_data( 
                            field
                        )
                    
                    return None

                futures = {executor.submit(run_chart_creation, task): task for task in chart_tasks}
                
                for future in as_completed(futures):
                    kw_info_original = futures[future]["kw_info"]
                    field_name = kw_info_original.get('field', 'unknown')
                    try:
                        chart = future.result() 
                        if chart.get('chart_data') and chart.get('ratio') != '0.0%':
                            chart['priority'] = kw_info_original.get('priority', 99)
                            charts.append(chart)
                            logging.info(f"   ✅ [{field_name}] 차트 생성 완료 (DB 집계)")
                        else:
                            logging.warning(f"   ⚠️  [{field_name}] 차트 데이터가 비어있음 (DB 집계)")
                    except Exception as e:
                        logging.error(f"   ❌ [{field_name}] 차트 생성 실패: {e}", exc_info=True)
            
            charts.sort(key=lambda x: x.get('priority', 99))
            
            for chart in charts:
                if 'priority' in chart:
                    del chart['priority']

        # 3.5단계: 교차 분석 차트 생성
        logging.info("   3.5단계: 교차 분석 차트 생성")
        if len(charts) < 5 and len(ranked_keywords) > 0:
            
            CROSSTAB_CANDIDATES = [
                ('gender', '성별'),
                ('birth_year', '연령대'),
                ('marital_status', '결혼 여부'),
                ('income_personal_monthly', '소득 수준'),
                ('job_duty_raw', '직무'), 
                ('job_title_raw', '직업'),
            ]
            
            primary_kw = None
            for kw in ranked_keywords:
                if kw.get("type") == "filter":
                    primary_kw = kw
                    break 
            
            primary_field = None
            primary_korean_name = None
            
            if primary_kw:
                primary_field = primary_kw.get('field')
                primary_korean_name = primary_kw.get('description')
            
            secondary_field = None
            secondary_korean_name = None
            
            if primary_field and primary_field != 'unknown' and primary_field in objective_fields:
                for field, korean in CROSSTAB_CANDIDATES:
                    if field == primary_field or field in search_used_fields:
                        continue
                    
                    secondary_field = field
                    secondary_korean_name = korean
                    logging.info(f"   ✨ 새 교차분석 축 발견: '{primary_korean_name}' vs '{secondary_korean_name}'")
                    break
            
            if secondary_field:
                crosstab_chart = create_crosstab_chart(
                    panels_data,
                    primary_field, secondary_field,
                    primary_korean_name, secondary_korean_name
                )
                if crosstab_chart:
                    charts.append(crosstab_chart)
                    if primary_field not in used_fields:
                         used_fields.append(primary_field) 
                    # [수정] 교차분석 보조축도 제외 목록에 추가
                    if secondary_field not in used_fields:
                        used_fields.append(secondary_field)
                    logging.info(f"   ✅ [{len(charts)}] 교차 분석 차트 생성 ({primary_korean_name} vs {secondary_korean_name})")
            else:
                logging.warning("   ⚠️  교차 분석 스킵: 1순위 필터 키워드가 없거나, 적절한 보조축 후보가 없음 (모두 검색어에 포함됨)")
                
        # 4단계: 높은 비율 필드 찾기
        logging.info("   4단계: 높은 비율 필드 차트 생성 (검색 결과 기반)")
        needed_charts = 5 - len(charts)
        # 3.5단계에서 보조축까지 제외됨
        exclude_fields_for_step4 = list(set(used_fields) | search_used_fields)
        
        if needed_charts > 0:
            high_ratio_fields = find_high_ratio_fields_optimized(
                panels_data, 
                # [롤백] 검색에 사용된 필드와 이미 차트로 만들어진 필드를 제외
                exclude_fields=exclude_fields_for_step4,
                threshold=50.0,
                max_charts=needed_charts
            )
            
            for field_info in high_ratio_fields:
                if len(charts) >= 5: break
                
                chart = {
                    "topic": f"{field_info['korean_name']} 분포",
                    "description": f"{field_info['top_ratio']:.1f}%가 '{field_info['top_category']}'로 뚜렷한 패턴을 보입니다.",
                    "ratio": f"{field_info['top_ratio']:.1f}%",
                    "chart_data": [{"label": field_info['korean_name'], "values": field_info['distribution']}]
                }
                charts.append(chart)
                logging.info(f"   ✅ [{len(charts)}] {field_info['korean_name']} ({field_info['top_ratio']:.1f}%) 차트 생성")
        
        # 5단계: 요약 생성
        main_summary = f"총 {len(panels_data)}명의 응답자 데이터를 분석했습니다. "
        if charts:
            top_chart = charts[0]
            summary_desc = top_chart.get('description', '')
            if '전체 데이터 기준' in summary_desc:
                summary_desc = summary_desc.split(':', 1)[-1].strip()
            elif ':' in summary_desc:
                 summary_desc = summary_desc.split(':', 1)[-1].strip()
            
            main_summary += f"주요 분석 결과: {top_chart.get('topic', '')}에서 {top_chart.get('ratio', '0%')}의 비율을 보입니다."
        
        result = {
            "query": query,
            "total_count": len(panels_data),
            "main_summary": main_summary,
            "charts": charts
        }
        
        logging.info(f"✅ 분석 완료: {len(charts)}개 차트 생성 (최적화)")
        return result, 200
        
    except Exception as e:
        logging.error(f"❌ 분석 실패: {e}", exc_info=True)
        return {"main_summary": f"분석 중 오류 발생: {str(e)}", "charts": []}, 500