import os
import json
import boto3
from dotenv import load_dotenv
from fastapi import HTTPException

# .env 파일에서 환경 변수를 불러옵니다.
load_dotenv()

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
        print(f"Bedrock 클라이언트 생성 실패: {e}")
        return None

# 텍스트를 임베딩 벡터로 변환하는 함수
def get_embedding_from_bedrock(text: str) -> list[float]:
    """
    주어진 텍스트를 Bedrock의 Titan 모델을 사용하여 임베딩 벡터로 변환합니다.

    Args:
        text (str): 임베딩할 텍스트.

    Returns:
        list[float]: 임베딩된 벡터.
    """
    client = get_bedrock_client()
    if not client:
        return None

    model_id = "amazon.titan-embed-text-v1" # 사용할 임베딩 모델 ID
    
    request_body = json.dumps({"inputText": text})

    try:
        response = client.invoke_model(
            body=request_body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        embedding = response_body.get("embedding")
        return embedding

    except Exception as e:
        print(f"임베딩 생성 실패: {e}")
        return None

# 하이브리드 검색을 위한 질의 분리 함수
def split_query_for_hybrid_search(query: str) -> dict:
    """
    Claude 3 Opus를 이용해 질의를 정형(SQL)과 비정형(임베딩)으로 분리합니다.
    """
    client = get_bedrock_client()
    if not client:
        raise HTTPException(status_code=500, detail="Bedrock 클라이언트 생성 실패")

    prompt = f"""
    사용자의 질의를 다음 두 가지 형태로 분리해줘.

    1. 정형(Structured): SQL 또는 키워드 기반으로 검색할 수 있는 명확한 조건
    2. 비정형(Semantic): 의미나 문맥을 임베딩 검색으로 처리해야 하는 부분

    예시:
    입력: "서울 강남구 근처에서 점심 먹기 좋은 한식집 추천해줘"
    출력(JSON):
    {{
      "structured_condition": "지역 = '서울 강남구' AND 음식종류 = '한식'",
      "semantic_condition": "점심 먹기 좋은 분위기의 맛집"
    }}

    입력: "{query}"
    출력(JSON):
    """
    try:
        response = client.invoke_model(
            modelId="anthropic.claude-3-opus-20240229-v1:0",
            accept="application/json",
            contentType="application/json",
            body=json.dumps({
                "max_tokens": 800,
                "temperature": 0.3,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        result = json.loads(response["body"].read())
        text_output = result["content"][0]["text"].strip()
        parsed = json.loads(text_output)
        
        structured = parsed.get("structured_condition", "").strip()
        semantic = parsed.get("semantic_condition", "").strip()
        
        return {
            "structured_condition": structured,
            "semantic_condition": semantic
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

    # 2. 비정형 조건을 임베딩 벡터로 변환 (get_embedding_from_bedrock 함수 호출)
    try:
        embedding_vector = get_embedding_from_bedrock(semantic_query)
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
    embedding = get_embedding_from_bedrock(test_text_embedding)
    if embedding:
        print("임베딩 벡터가 성공적으로 생성되었습니다!")
        print(f"벡터의 길이: {len(embedding)}")
        print(f"벡터의 일부: {embedding[:5]}...")
    else:
        print("임베딩 생성에 실패했습니다.")
    
    print("-" * 30)

    # 질의 분리 및 통합 함수 테스트
    test_text_split = "서울 강남구 근처에서 점심 먹기 좋은 한식집 추천해줘"
    try:
        processed_query = process_hybrid_query(test_text_split)
        print("질의 처리 함수가 성공적으로 완료되었습니다!")
        print("정형 조건:", processed_query["structured_condition"])
        print("임베딩 벡터의 길이:", len(processed_query["embedding_vector"]))
        print("임베딩 벡터의 일부:", processed_query["embedding_vector"][:5])
    except HTTPException as e:
        print(f"질의 처리 테스트 실패: {e.detail}")