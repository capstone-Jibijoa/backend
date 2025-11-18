import re
from typing import List, Dict, Union, Tuple, Optional, Any
import logging

# [수정] LLM이 생성할 수 있는 유사 키워드를 확장 규칙에 추가
# 키워드 확장용 매핑
CATEGORY_MAPPING = {
    '직장인': ['사무직', '전문직', '경영관리직', '생산노무직', '서비스직', '판매직', '기술직'],
    '고소득': ['월 500~599만원', '월 600~699만원', '월 700만원 이상'],
    '저소득': ['월 100~199만원', '월 200~299만원', '월 100만원 미만'],
    '중산층': ['월 300~399만원', '월 400~499만원'],
    '고학력': ['대학교 졸업', '대학원 재학 이상'],
    '저학력': ['고등학교 졸업 이하', '중학교 졸업 이하'],
    '젊은층': ['20대', '30대'],
    '청년': ['20대', '30대'],
    'MZ세대': ['20대', '30대'],
    '중장년층': ['40대', '50대'],
    'X세대': ['40대', '50대'],
    '장년층': ['50대', '60대'],
    '베이비부머': ['50대', '60대 이상'],
    '노년층': ['60대 이상'],
    '청소년': ['10대'],
    # 유사 키워드 추가
    '고소득자': ['월 500~599만원', '월 600~699만원', '월 700만원 이상'],
    '저소득자': ['월 100~199만원', '월 200~299만원', '월 100만원 미만'],
    '아이폰 사용자': ['Apple'],
}

VALID_REGIONS = [
    '서울', '경기', '인천', '부산', '대구', '대전', '광주', '울산', '세종',
    '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'
]

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

# 벡터 검색 결과의 category를 PostgreSQL 필드로 매핑
# [최종 수정] 사용자가 제안한 1:N 매핑 규칙 적용
VECTOR_CATEGORY_TO_FIELD = {
    "DEMO_BASIC": ["gender", "birth_year", "region_major", "region_minor"],
    "FAMILY_STATUS": ["marital_status", "family_size", "children_count"],
    "JOB_EDUCATION": ["job_title_raw", "job_duty_raw", "education_level"],
    "INCOME_LEVEL": ["income_personal_monthly", "income_household_monthly"],
    "TECH_OWNER": ["owned_electronics"], 
    "CAR_OWNER": ["car_ownership", "car_manufacturer_raw", "car_model_raw"],
    "DRINK_HABIT": ["drinking_experience", "drinking_experience_other_details_raw"],
    "SMOKE_HABIT": ["smoking_experience", "smoking_brand", "e_cigarette_experience"],
}


# 1. 정적 매핑 규칙 (Python 코드로 관리)
KEYWORD_MAPPINGS: List[Tuple[Union[re.Pattern, str], Dict[str, str]]] = [
    # --- 인구통계 ---
    (re.compile(r'^(여|여자|여성)$', re.IGNORECASE), {"field": "gender", "description": "성별", "type": "filter"}),
    (re.compile(r'^(남|남자|남성)$', re.IGNORECASE), {"field": "gender", "description": "성별", "type": "filter"}), # 남성 추가
    (re.compile(r'미혼|싱글'), {"field": "marital_status", "description": "결혼 여부", "type": "filter"}),
    (re.compile(r'기혼|결혼'), {"field": "marital_status", "description": "결혼 여부", "type": "filter"}),
    (re.compile(r'(\d+인|가족\s*\d+명|\d+인\s*가구)'), {"field": "family_size", "description": "가족 수", "type": "filter"}),
    ("대졸", {"field": "education_level", "description": "최종학력", "type": "filter"}),
    (re.compile(r'(\d{2}대\s*(초반|중반|후반)|\d{2}대(\s*이상)?)'), {"field": "birth_year", "description": "연령대", "type": "filter"}),
    ("대학원", {"field": "education_level", "description": "최종학력", "type": "filter"}),
    
    # --- 지역 ---
    (re.compile(r'|'.join(VALID_REGIONS) + r'(\s*에\s*사는|\s*거주자)?'), {"field": "region_major", "description": "거주 지역", "type": "filter"}),
    
    # --- 직업/직무 ---
    (re.compile(r'직장인|사무직|전문직|경영관리직|생산노무직|서비스직|판매직|기술직'), {"field": "job_duty_raw", "description": "직무", "type": "filter"}),
    (re.compile(r'학생|대학생|대학원생'), {"field": "job_title_raw", "description": "직업", "type": "filter"}),
    (re.compile(r'마케팅|마케터'), {"field": "job_duty_raw", "description": "직무", "type": "filter"}),
    (re.compile(r'IT|개발|개발자'), {"field": "job_duty_raw", "description": "직무", "type": "filter"}),

    # --- 소득 ---
    (re.compile(r'월소득|월\s*소득|개인소득|본인\s*소득'), {"field": "income_personal_monthly", "description": "월소득(개인)", "type": "filter"}),
    (re.compile(r'고소득|저소득|중산층'), {"field": "income_personal_monthly", "description": "월소득(개인)", "type": "filter"}),
    (re.compile(r'가구소득|가족\s*소득|가정\s*소득'), {"field": "income_household_monthly", "description": "월소득(가구)", "type": "filter"}),

    # --- 휴대폰 ---
    (re.compile(r'아이폰|애플', re.IGNORECASE), {"field": "phone_brand_raw", "description": "휴대폰 브랜드", "type": "filter"}),
    (re.compile(r'갤럭시|삼성', re.IGNORECASE), {"field": "phone_brand_raw", "description": "휴대폰 브랜드", "type": "filter"}),
    (re.compile(r'lg', re.IGNORECASE), {"field": "phone_brand_raw", "description": "휴대폰 브랜드", "type": "filter"}),
    
    # [수정] '가전제품' 키워드를 PostgreSQL의 owned_electronics 필드와 연결
    (re.compile(r'가전|전자제품'), {"field": "owned_electronics", "description": "보유 가전", "type": "filter"}),

    ("아이폰 15 pro 시리즈", {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    # (re.compile(r'아이폰\s*se\s*3'), {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}), # 누락된 정보 처리
    # 중복된 항목 제거: {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}
    (re.compile(r'(아이폰|iphone)\s*(15|14|13|12|11|x|se)', re.IGNORECASE), {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),
    (re.compile(r'갤럭시\s*(s|z|a|m|노트)\s*\d*', re.IGNORECASE), {"field": "phone_model_raw", "description": "휴대폰 모델", "type": "filter"}),

    # --- 차량 소유/모델 ---
    # [수정] '차량 보유' 관련 키워드도 car_model_raw로 매핑하여 차종이 표시되도록 유도
    (re.compile(r'차량\s*보유|차\s*있음||차\s*선호||자가용|차종|자동차\s*모델|차량\s*종류'), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    # [수정] '차 없음'은 car_ownership에 그대로 유지
    (re.compile(r'차\s*없음|무소유'), {"field": "car_ownership", "description": "차량 보유 여부", "type": "filter"}),
    
    # 국내 브랜드
    (re.compile(r'기아(차)?'), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    (re.compile(r'르노|르노삼성'), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    (re.compile(r'쌍용(차)?|kg\s*모빌리티', re.IGNORECASE), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    (re.compile(r'쉐보레|한국gm', re.IGNORECASE), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    (re.compile(r'현대(차)?'), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("제네시스", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),

    # 해외/수입차 브랜드
    ("아우디", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("벤틀리", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    (re.compile(r'bmw', re.IGNORECASE), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("포드", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("혼다", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("인피니티", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("재규어", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    (re.compile(r'지프|jeep', re.IGNORECASE), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("랜드로버", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("렉서스", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("링컨", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    (re.compile(r'벤츠|메르세데스-벤츠'), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    (re.compile(r'미니|bmw\s*미니', re.IGNORECASE), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("닛산", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    (re.compile(r'포르쉐', re.IGNORECASE), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("롤스로이스", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("테슬라", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    (re.compile(r'도요타|토요타'), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("볼보", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    (re.compile(r'폭스바겐', re.IGNORECASE), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),

    # 상용차 및 기타
    (re.compile(r'만|man', re.IGNORECASE), {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("스카니아", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("포톤", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("이스트", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    ("이베코", {"field": "car_manufacturer_raw", "description": "차량 제조사", "type": "filter"}),
    
    # 모델명
    (re.compile(r'그랜저|그랜져', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'소나타|쏘나타'), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'아반떼|아반테', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("캐스퍼", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("코나", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("투싼", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'싼타페|산타페'), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("팰리세이드", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'아이오닉\s*5', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'아이오닉\s*6', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("넥쏘", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'포터\s*2?', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'봉고\s*3?', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("레이", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("모닝", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'k3', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'k5', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'k8', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("스팅어", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("셀토스", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("스포티지", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("쏘렌토", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("카니발", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("모하비", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'ev6', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("니로", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'제네시스\s*g70', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'제네시스\s*g80', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'제네시스\s*g90', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'제네시스\s*gv70', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'제네시스\s*gv80', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),

    # 르노삼성
    (re.compile(r'xm3', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'sm6', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),

    # KG모빌리티 (구 쌍용)
    ("토레스", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("티볼리", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("코란도", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("렉스턴", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),

    # 쉐보레
    ("트레일블레이저", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("트랙스", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("말리부", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("트래버스", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("타호", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),

    # 수입차 (BMW, 벤츠, 아우디 등)
    (re.compile(r'bmw\s*3\s*시리즈', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'bmw\s*5\s*시리즈', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'bmw\s*x3', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'bmw\s*x5', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'벤츠\s*c\s*클래스', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'벤츠\s*e\s*클래스', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'아우디\s*a4', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'아우디\s*a6', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'테슬라\s*모델\s*3', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'테슬라\s*모델\s*y', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),

    # 기타 및 포괄적
    ("기타 국산차", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    ("기타 수입차", {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    (re.compile(r'suv|세단|트럭|승합', re.IGNORECASE), {"field": "car_model_raw", "description": "차량 모델명", "type": "filter"}),
    
    # --- 흡연/음주 ---
    (re.compile(r'흡연'), {"field": "smoking_experience", "description": "흡연 여부", "type": "filter"}),
    (re.compile(r'비흡연|금연'), {"field": "smoking_experience", "description": "흡연 여부", "type": "filter"}),

    # 담배 종류/브랜드
    ("레종", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("에쎄", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("보헴", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("아프리카", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("더원", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("시즌", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    ("아이스볼트 gt", {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
    (re.compile(r'디스플러스|디스'), {"field": "smoking_brand", "description": "담배 종류", "type": "filter"}),
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

    # 전자 담배 이용 경험
    (re.compile(r'아이코스|iqos', re.IGNORECASE), {"field": "e_cigarette_experience", "description": "전자 담배 이용 경험", "type": "filter"}),
    (re.compile(r'릴|lil', re.IGNORECASE), {"field": "e_cigarette_experience", "description": "전자 담배 이용 경험", "type": "filter"}),
    (re.compile(r'글로|glo', re.IGNORECASE), {"field": "e_cigarette_experience", "description": "전자 담배 이용 경험", "type": "filter"}),
    (re.compile(r'차이코스|cqs', re.IGNORECASE), {"field": "e_cigarette_experience", "description": "전자 담배 이용 경험", "type": "filter"}),
    (re.compile(r'전자담배|궐련형', re.IGNORECASE), {"field": "e_cigarette_experience", "description": "전자 담배 이용 경험", "type": "filter"}),
    
    # 음주 경험
    (re.compile(r'음주|술'), {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    (re.compile(r'금주|비음주'), {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("소주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("맥주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("저도주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("막걸리", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    (re.compile(r'양주|위스키|보드카|데킬라|진'), {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("와인", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("과일칵테일주", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    (re.compile(r'일본청주|사케'), {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),
    ("최근 1년 이내 술을 마시지 않음", {"field": "drinking_experience", "description": "음주 경험", "type": "filter"}),

    # --- Q-Poll 키워드 (type: "qpoll") ---
    (re.compile(r'체력\s*관리|운동'), {"field": "physical_activity", "description": QPOLL_FIELD_TO_TEXT["physical_activity"], "type": "qpoll"}),
    (re.compile(r'ott|스트리밍\s*서비스'), {"field": "ott_count", "description": QPOLL_FIELD_TO_TEXT["ott_count"], "type": "qpoll"}),
    ("전통시장", {"field": "traditional_market_freq", "description": QPOLL_FIELD_TO_TEXT["traditional_market_freq"], "type": "qpoll"}),
    (re.compile(r'설\s*선물|선물\s*선호'), {"field": "lunar_new_year_gift_pref", "description": QPOLL_FIELD_TO_TEXT["lunar_new_year_gift_pref"], "type": "qpoll"}),
    ("겨울방학", {"field": "elementary_winter_memories", "description": QPOLL_FIELD_TO_TEXT["elementary_winter_memories"], "type": "qpoll"}),
    ("반려동물", {"field": "pet_experience", "description": QPOLL_FIELD_TO_TEXT["pet_experience"], "type": "qpoll"}),
    (re.compile(r'이사\s*스트레스'), {"field": "moving_stress_factor", "description": QPOLL_FIELD_TO_TEXT["moving_stress_factor"], "type": "qpoll"}),
    (re.compile(r'기분\s*좋아지는\s*소비|나를\s*위한\s*소비'), {"field": "happiest_self_spending", "description": QPOLL_FIELD_TO_TEXT["happiest_self_spending"], "type": "qpoll"}),
    (re.compile(r'사용하는\s*앱|주요\s*앱'), {"field": "most_used_app", "description": QPOLL_FIELD_TO_TEXT["most_used_app"], "type": "qpoll"}),
    (re.compile(r'스트레스\s*상황'), {"field": "stress_situation", "description": QPOLL_FIELD_TO_TEXT["stress_situation"], "type": "qpoll"}),
    (re.compile(r'스트레스\s*해소'), {"field": "stress_relief_method", "description": QPOLL_FIELD_TO_TEXT["stress_relief_method"], "type": "qpoll"}),
    (re.compile(r'피부\s*만족도'), {"field": "skin_satisfaction", "description": QPOLL_FIELD_TO_TEXT["skin_satisfaction"], "type": "qpoll"}),
    (re.compile(r'스킨케어\s*소비|스킨케어\s*지출'), {"field": "skincare_spending", "description": QPOLL_FIELD_TO_TEXT["skincare_spending"], "type": "qpoll"}),
    (re.compile(r'스킨케어\s*구매\s*요소'), {"field": "skincare_purchase_factor", "description": QPOLL_FIELD_TO_TEXT["skincare_purchase_factor"], "type": "qpoll"}),
    (re.compile(r'ai\s*챗봇\s*사용|챗봇\s*경험', re.IGNORECASE), {"field": "ai_chatbot_used", "description": QPOLL_FIELD_TO_TEXT["ai_chatbot_used"], "type": "qpoll"}),
    (re.compile(r'주로\s*사용하는\s*ai\s*챗봇', re.IGNORECASE), {"field": "ai_chatbot_main", "description": QPOLL_FIELD_TO_TEXT["ai_chatbot_main"], "type": "qpoll"}),
    (re.compile(r'ai\s*챗봇\s*활용|챗봇\s*용도', re.IGNORECASE), {"field": "ai_chatbot_purpose", "description": QPOLL_FIELD_TO_TEXT["ai_chatbot_purpose"], "type": "qpoll"}),
    (re.compile(r'ai\s*챗봇\s*호감도', re.IGNORECASE), {"field": "ai_chatbot_sentiment", "description": QPOLL_FIELD_TO_TEXT["ai_chatbot_sentiment"], "type": "qpoll"}),
    (re.compile(r'해외여행|여행\s*선호'), {"field": "overseas_travel_pref", "description": QPOLL_FIELD_TO_TEXT["overseas_travel_pref"], "type": "qpoll"}),
    (re.compile(r'빠른\s*배송|새벽\s*배송'), {"field": "fast_delivery_usage", "description": QPOLL_FIELD_TO_TEXT["fast_delivery_usage"], "type": "qpoll"}),
    (re.compile(r'여름철\s*걱정'), {"field": "summer_worry", "description": QPOLL_FIELD_TO_TEXT["summer_worry"], "type": "qpoll"}),
    (re.compile(r'물건\s*처리|업사이클링'), {"field": "unused_item_disposal", "description": QPOLL_FIELD_TO_TEXT["unused_item_disposal"], "type": "qpoll"}),
    (re.compile(r'기상\s*알람'), {"field": "alarm_setting_style", "description": QPOLL_FIELD_TO_TEXT["alarm_setting_style"], "type": "qpoll"}),
    ("혼밥", {"field": "eating_alone_frequency", "description": QPOLL_FIELD_TO_TEXT["eating_alone_frequency"], "type": "qpoll"}),
    (re.compile(r'행복한\s*노년'), {"field": "happy_old_age_condition", "description": QPOLL_FIELD_TO_TEXT["happy_old_age_condition"], "type": "qpoll"}),
    (re.compile(r'여름\s*땀\s*불편함'), {"field": "sweat_discomfort", "description": QPOLL_FIELD_TO_TEXT["sweat_discomfort"], "type": "qpoll"}),
    (re.compile(r'다이어트\s*방법'), {"field": "most_effective_diet", "description": QPOLL_FIELD_TO_TEXT["most_effective_diet"], "type": "qpoll"}),
    ("야식", {"field": "late_night_snack_method", "description": QPOLL_FIELD_TO_TEXT["late_night_snack_method"], "type": "qpoll"}),
    (re.compile(r'여름철\s*간식'), {"field": "favorite_summer_snack", "description": QPOLL_FIELD_TO_TEXT["favorite_summer_snack"], "type": "qpoll"}),
    (re.compile(r'최근\s*지출'), {"field": "recent_major_spending", "description": QPOLL_FIELD_TO_TEXT["recent_major_spending"], "type": "qpoll"}),
    (re.compile(r'ai\s*서비스\s*활용', re.IGNORECASE), {"field": "ai_service_usage_area", "description": QPOLL_FIELD_TO_TEXT["ai_service_usage_area"], "type": "qpoll"}),
    (re.compile(r'미니멀리스트|맥시멀리스트'), {"field": "minimalist_maximalist", "description": QPOLL_FIELD_TO_TEXT["minimalist_maximalist"], "type": "qpoll"}),
    (re.compile(r'여행\s*스타일'), {"field": "travel_planning_style", "description": QPOLL_FIELD_TO_TEXT["travel_planning_style"], "type": "qpoll"}),
    (re.compile(r'비닐봉투\s*줄이기'), {"field": "plastic_bag_reduction_effort", "description": QPOLL_FIELD_TO_TEXT["plastic_bag_reduction_effort"], "type": "qpoll"}),
    (re.compile(r'포인트\s*적립'), {"field": "point_benefit_attention", "description": QPOLL_FIELD_TO_TEXT["point_benefit_attention"], "type": "qpoll"}),
    ("초콜릿", {"field": "chocolate_consumption_time", "description": QPOLL_FIELD_TO_TEXT["chocolate_consumption_time"], "type": "qpoll"}),
    ("개인정보보호", {"field": "personal_info_protection_habit", "description": QPOLL_FIELD_TO_TEXT["personal_info_protection_habit"], "type": "qpoll"}),
    (re.compile(r'여름\s*패션'), {"field": "summer_fashion_must_have", "description": QPOLL_FIELD_TO_TEXT["summer_fashion_must_have"], "type": "qpoll"}),
    (re.compile(r'우산\s*없을\s*때'), {"field": "no_umbrella_reaction", "description": QPOLL_FIELD_TO_TEXT["no_umbrella_reaction"], "type": "qpoll"}),
    (re.compile(r'갤러리\s*사진'), {"field": "most_saved_photo_type", "description": QPOLL_FIELD_TO_TEXT["most_saved_photo_type"], "type": "qpoll"}),
    (re.compile(r'물놀이\s*장소'), {"field": "favorite_summer_water_spot", "description": QPOLL_FIELD_TO_TEXT["favorite_summer_water_spot"], "type": "qpoll"}),

]

from utils import FIELD_NAME_MAP

def get_field_mapping(keyword: str) -> Optional[Dict[str, Any]]:
    """
    주어진 키워드에 대해 미리 정의된 매핑 리스트를 검색하여
    일치하는 필드 정보(field, description, type)를 반환합니다.
    
    Args:
        keyword (str): 사용자 쿼리에서 추출된 키워드.

    Returns:
        Optional[Dict[str, Any]]: 일치하는 매핑 정보를 담은 딕셔너리 또는 None.
    """
    # 쿼리 키워드를 소문자로 변환하여 검색 효율성 및 일관성 확보
    search_keyword = keyword.lower().strip()

    for pattern, mapping_info in KEYWORD_MAPPINGS:
        # mapping_info는 수정될 수 있으므로 복사해서 사용
        result_info = mapping_info.copy()
        rule_type = result_info.get("type", "filter")

        # insights.py의 로직을 통합: Q-Poll이 아닌 경우 FIELD_NAME_MAP에서 description을 가져옴
        if rule_type != 'qpoll':
            result_info["description"] = FIELD_NAME_MAP.get(result_info["field"], result_info.get("description", keyword))

        if isinstance(pattern, re.Pattern):
            # 정규 표현식 패턴인 경우: 매치 여부 확인
            if pattern.search(search_keyword):
                return result_info
        elif isinstance(pattern, str):
            # 문자열 패턴인 경우: 'in' 연산자로 부분 일치 확인 (더 유연함)
            if pattern.lower() in search_keyword:
                return result_info
                
    # 매핑 규칙을 찾지 못하면 unknown 객체 반환
    logging.warning(f" ⚠️  '{keyword}'에 대한 매핑 규칙 없음. 'unknown'(벡터)으로 처리.")
    return {
        "field": "unknown", 
        "description": keyword, 
        "type": "unknown"
    }