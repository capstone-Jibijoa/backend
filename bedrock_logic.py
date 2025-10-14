import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv() # .env 파일에서 환경 변수를 불러오기

# AWS Bedrock에 연결하는 함수
def get_bedrock_client():
    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("us-east-1"),
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
    주어진 텍스트를 Bedrock을 사용하여 임베딩 벡터로 변환합니다.

    Args:
        text (str): 임베딩할 텍스트.

    Returns:
        list[float]: 임베딩된 벡터.
    """
    client = get_bedrock_client()
    if not client:
        return None

    model_id = "amazon.titan-embed-text-v1" # 사용할 임베딩 모델 ID
    accept = "application/json" # Bedrock에서 받고자 하는 응답 데이터 형식
    content_type = "application/json" # 내가 Bedrock에 보내는 데이터의 형식

    # Bedrock 모델에 전송할 요청 본문
    request_body = json.dumps({"inputText": text})

    try:
        # invoke_model을 호출하여 임베딩 생성
        response = client.invoke_model(
            body=request_body,
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )
        response_body = json.loads(response.get("body").read())
        embedding = response_body.get("embedding")
        return embedding

    except Exception as e:
        print(f"임베딩 생성 실패: {e}")
        return None

# 함수를 사용하여 연결 및 임베딩 테스트
if __name__ == "__main__":
    test_text = "이것은 테스트 검색어입니다."
    embedding = get_embedding_from_bedrock(test_text)

    if embedding:
        print("임베딩 벡터가 성공적으로 생성되었습니다!")
        print(f"벡터의 길이: {len(embedding)}")
        print(f"벡터의 일부: {embedding[:5]}...") # 벡터의 일부만 출력하여 확인
    else:
        print("임베딩 생성에 실패했습니다.")