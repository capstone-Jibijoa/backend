from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import json
import os

# AWS Bedrock 설정
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"  # 실제 사용 리전으로 수정하세요
)

# FastAPI 초기화
app = FastAPI(title="Hybrid Query Split API using Bedrock Opus3")

# 데이터 모델
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    structured_condition: str
    semantic_condition: str

# 질의 분리 함수
def split_query_for_hybrid_search(query: str) -> dict:
    """
    Claude 3 Opus를 이용해 질의를 정형(SQL 검색용)과 비정형(임베딩 검색용)으로 분리
    """
    prompt = f"""
    사용자의 질의를 다음 두 가지 형태로 분리해줘.

    1. 정형(Structured): SQL 또는 키워드 기반으로 검색할 수 있는 명확한 조건
    2. 비정형(Semantic): 의미나 문맥을 임베딩 검색(KURE-v1)으로 처리해야 하는 부분

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
        response = bedrock_runtime.invoke_model(
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

        # Claude 출력 JSON 파싱
        parsed = json.loads(text_output)

        structured = parsed.get("structured_condition", "").strip()
        semantic = parsed.get("semantic_condition", "").strip()

        return {
            "structured_condition": structured,
            "semantic_condition": semantic
        }

    except Exception as e:
        print("Bedrock 호출 에러:", e)
        raise HTTPException(status_code=500, detail=str(e))

# API 엔드포인트
@app.post("/split", response_model=QueryResponse)
async def split_query(request: QueryRequest):
    """
    POST /split
    {
      "query": "서울 강남구 근처에서 점심 먹기 좋은 한식집 추천해줘"
    }
    """
    result = split_query_for_hybrid_search(request.query)
    return QueryResponse(**result)

# 로컬 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
