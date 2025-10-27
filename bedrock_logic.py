import os
import json
import boto3
from dotenv import load_dotenv
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer # sentence-transformers 라이브러리 추가

# .env 파일에서 환경 변수를 불러옵니다.
load_dotenv()

# =======================================================
# 1. KURE 임베딩 모델을 모듈 수준에서 한 번만 초기화한다.
# =======================================================
try:
    # KURE-v1 모델 로드 (Hugging Face Hub 경로 사용)
    # 모델 로딩은 시간이 걸리므로, 프로그램 시작 시 단 한 번만 수행됩니다.
    KURE_MODEL = SentenceTransformer("nlpai-lab/KURE-v1") 
    print("KURE 임베딩 모델 로드 완료! (다음은 Bedrock 호출 전입니다)")
except Exception as e:
    KURE_MODEL = None
    print(f"KURE 임베딩 모델 로드 실패: {e}")
    # 모델 로드 실패 시, 임베딩 관련 함수는 오류를 반환해야 합니다.

# AWS Bedrock에 연결하는 클라이언트 객체를 한 번만 생성하여 재사용합니다.
def get_bedrock_client():
    """
    AWS Bedrock 클라이언트를 생성하고 반환합니다.
    """
    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION"),  # .env의 AWS_REGION 환경 변수 사용
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        return client
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bedrock 클라이언트 생성 실패: {e}")

# =======================================================
# 2. get_kure_embedding 함수는 KURE_MODEL.encode()를 사용한다.
# =======================================================
def get_kure_embedding(text: str) -> list[float]:
    """
    주어진 텍스트를 KURE 모델을 사용하여 임베딩 벡터로 변환합니다.
    """
    if KURE_MODEL is None:
        raise HTTPException(status_code=500, detail="KURE 임베딩 모델이 초기화되지 않았습니다.")

    try:
        # KURE 모델 객체의 .encode() 메서드를 사용하여 임베딩 생성
        # NumPy 배열을 JSON 직렬화가 가능한 Python 리스트로 변환
        embedding = KURE_MODEL.encode(text).tolist() 
        return embedding

    except Exception as e:
        print(f"KURE 임베딩 생성 실패: {e}")
        # 임베딩 생성 실패 시 None 대신 HTTPException 발생을 권장 (process_hybrid_query와 연동)
        raise HTTPException(status_code=500, detail=f"KURE 임베딩 생성 실패: {e}")

# 하이브리드 검색을 위한 질의 분리 함수
def split_query_for_hybrid_search(query: str) -> dict:
    """
    Claude 4.1 Opus를 이용해 질의를 정형(Structured Filter)과 비정형(Semantic Keyword)으로 분리합니다.
    """
    client = get_bedrock_client() # get_bedrock_client 함수는 파일 상단에 정의되어 있어야 합니다.
    print("Bedrock 클라이언트 생성 완료! (다음은 API 호출 전입니다)")
    if not client:
        raise HTTPException(status_code=500, detail="Bedrock 클라이언트 생성 실패")

    # =======================================================
    # 프롬프트 디테일 강화 (오차 최소화) 적용 부분
    # =======================================================
    system_prompt = """
    당신은 사용자 질의를 하이브리드 검색에 사용될 JSON 객체로 변환하는 최고 전문가입니다.
**응답은 오직 하나의 완벽한 JSON 객체** 형태로만 반환해야 하며, 어떤 추가 설명이나 문장도 포함해서는 안 됩니다.
**현재 데이터 샘플은 최대 150개**이므로, target_count는 150을 초과하지 않도록 엄격하게 제한해야 합니다.

[핵심 규칙]
1. **정형 조건 (filters)**: '지역', '성별', '나이', '소득', '직무' 등 명확한 속성 필터는 'filters' 배열에 객체 형태로 변환하세요.
   - **컬럼 목록 확정**: 다음 확정된 컬럼 목록만 사용하세요. 질문에 직접 관련된 정보만 필터링하고, 나머지 컬럼은 무시하세요:
     **region_major, gender, birth_year, marital_status, education_level, job_duty, income_personal_monthly, car_ownership, drinking_experience, smoking_experience**
   - **연산자**: EQ(동일), BETWEEN(범위), GT(초과), LT(미만)만 사용하세요.
   - **값 표준화**: 
     a. **나이 변환**: 나이(예: 30~40대)는 **현재 연도(2025년)**를 기준으로 출생 연도(birth_year)의 **BETWEEN** 범위(예: [1985, 1995])로 변환하세요.
     b. **성별 변환**: '남자', '여자'만 사용하세요.
     c. **누락 처리**: 정형 조건에 해당하는 내용이 없으면 'filters' 배열은 빈 리스트(`[]`)로 반환하세요.
     
2. **비정형 조건 (semantic_query)**: 의미론적 검색어는 'semantic_query' 필드에 담으세요.
   - **[매우 중요] 핵심 키워드 추출**: 'semantic_query'는 **KURE 임베딩에 바로 사용할 핵심 명사/구문**만 남기고 불필요한 관형어나 문장 성분(예: '추천해줘', '찾아줘' 등)은 **모두 제거**해야 합니다.
   
3. **목표 수량 (target_count)**: 쿼리에 'n명', 'top k'와 같은 목표 수량이 있으면 숫자로 반환하세요. 없으면 `null`로 반환하세요. **150을 초과하지 않도록** 하세요.

[출력 스키마 예시]
// 입력 쿼리 예시: '경기 30~40대 남자 술을 먹은 사람 50명'
{
  "target_count": 50,
  "filters": [
    { "key": "region_major", "operator": "EQ", "value": "경기" },
    { "key": "birth_year", "operator": "BETWEEN", "value": [1985, 1995] }, 
    { "key": "gender", "operator": "EQ", "value": "남자" },
    { "key": "drinking_experience", "operator": "EQ", "value": "경험 있음" }
  ],
  "semantic_query": "" // 이 쿼리에는 비정형 조건이 없으므로 빈 문자열 반환
}
"""
    
    user_prompt = f"""
    다음 쿼리를 분석하여 JSON 형식으로 반환하세요.
    쿼리: '{query}'
    """
    
    try:
        response = client.invoke_model(
            modelId="anthropic.claude-opus-4-1-20250805-v1:0",
            accept="application/json",
            contentType="application/json",
            body=json.dumps({
                "max_tokens": 800,
                "temperature": 0.1, # 안정적인 JSON 출력을 위해 온도를 낮춤
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            })
        )
        result = json.loads(response["body"].read())
        text_output = result["content"][0]["text"].strip()
        parsed = json.loads(text_output)
        
        # 반환 구조는 이전 단계에서 정의된 대로 유지 (JSON 문자열로 반환)
        filters = parsed.get("filters", []) 
        semantic = parsed.get("semantic_query", "").strip()
        
        return {
            "structured_condition": json.dumps(filters), # DB 로직에서 파싱할 JSON 필터 배열
            "semantic_condition": semantic # KURE 임베딩에 사용할 핵심 키워드
        }

    except Exception as e:
        print("Bedrock 호출 에러:", e)
        raise HTTPException(status_code=500, detail=f"Bedrock 호출 에러: {e}")

# 두 기능을 결합하여 하이브리드 검색에 필요한 모든 정보를 반환하는 함수
def process_hybrid_query(query: str) -> dict:
    """
    사용자 질의를 정형/비정형 조건으로 분리하고, 비정형 조건을 임베딩 벡터로 변환합니다.

    Args:
        query (str): 사용자의 전체 검색 질의.

    Returns:
        dict: 정형 조건과 임베딩 벡터를 포함하는 딕셔너리.
              예: {"structured_condition": "지역 = '서울 강남구'", "embedding_vector": [...] }
    """
    # 1. 질의 분리 (split_query_for_hybrid_search 함수 호출)
    try:
        split_result = split_query_for_hybrid_search(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 분리 실패: {e}")

    structured_query = split_result.get("structured_condition")
    semantic_query = split_result.get("semantic_condition")

    # 2. 비정형 조건을 임베딩 벡터로 변환 (get_kure_embedding 함수 호출)
    try:
        embedding_vector = get_kure_embedding(semantic_query)
        if not embedding_vector:
            raise HTTPException(status_code=500, detail="임베딩 벡터 생성 실패")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 벡터 변환 실패: {e}")

    # 3. 정형 조건과 벡터를 함께 반환
    return {
        "structured_condition": structured_query,
        "embedding_vector": embedding_vector
    }

# 이 파일이 직접 실행될 때만 테스트 코드를 실행합니다.
if __name__ == "__main__":
    # 임베딩 함수 테스트
    test_text_embedding = "이것은 테스트 검색어입니다."
    try:
        # get_kure_embedding 호출
        embedding = get_kure_embedding(test_text_embedding)
        if embedding:
            print("KURE 임베딩 벡터가 성공적으로 생성되었습니다!")
            print(f"벡터의 길이: {len(embedding)}") # KURE-v1의 차원은 1024입니다.
            print(f"벡터의 일부: {embedding[:5]}...")
        else:
            # 로직 변경으로 이 경로는 실행되지 않고 HTTPException이 발생해야 합니다.
            print("KURE 임베딩 생성에 실패했습니다. (예외 처리 확인 필요)")
    except HTTPException as e:
        print(f"KURE 임베딩 테스트 실패: {e.detail}")

    # 질의 분리 및 통합 함수 테스트
    test_text_split = "IT 종사자, 커피를 자주 마시는 사람"
    try:
        processed_query = process_hybrid_query(test_text_split)
        print("질의 처리 함수가 성공적으로 완료되었습니다!")
        print("정형 조건:", processed_query["structured_condition"])
        print("임베딩 벡터의 길이:", len(processed_query["embedding_vector"]))
        print("임베딩 벡터의 일부:", processed_query["embedding_vector"][:5])
    except HTTPException as e:
        print(f"질의 처리 테스트 실패: {e.detail}")