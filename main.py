# FastAPI 웹 서버의 메인 파일.
# 전체 시스템의 시작점이자, 각 모듈을 연결하는 파일

# 자연어 검색 POST (/api/search) 요청을 처리하는 엔드포인트를 포함합니다.
# 검색 로그 기록 POST (/api/search/log) 요청을 처리하는 엔드포인트도 포함합니다.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bedrock_logic import process_hybrid_query, split_query_for_hybrid_search
from db_logic import query_database_with_vector, log_search_query, query_database_with_hybrid_search

app = FastAPI()
# FastAPI 초기화
# app = FastAPI(title="Hybrid Query Split API using Bedrock Opus3")# 

# 전체 검색 API(/api/search), 요청 본문의 유효성을 검사
class SearchQuery(BaseModel):
    query: str

class SearchLog(BaseModel):
    query: str
    results_count: int

# 하이브리드 검색 질의 분리 API (/split), 사용자 자연어 입력 qurey를
# structured과 semantic으로 분리하는 로직의 입력 데이터
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    structured_condition: str
    semantic_condition: str

@app.post("/api/search")
async def search_products(search_query: SearchQuery):
    """
    자연어 검색 요청을 처리하고 하이브리드 검색 결과를 반환합니다.
    """
    try:
        # 1. process_hybrid_query 함수를 호출하여 질의를 분리하고 임베딩 벡터를 생성
        processed_query_data = process_hybrid_query(search_query.query)
        
        # 2. 질의 처리 결과에서 정형 조건과 임베딩 벡터를 추출
        structured_condition = processed_query_data["structured_condition"]
        embedding_vector = processed_query_data["embedding_vector"]

        # 3. 정형 조건과 임베딩 벡터를 모두 사용하여 데이터베이스에서 하이브리드 검색
        # query_database_with_hybrid_search 함수는 db_logic.py에 새롭게 구현되어야 합니다.
        search_results = query_database_with_hybrid_search(
            structured_condition,
            embedding_vector
        )

        if search_results is None:
            raise HTTPException(status_code=500, detail="데이터베이스 검색에 실패했습니다.")

        # 4. 검색 결과를 JSON 형태로 반환
        return {"results": search_results}

    except Exception as e:
        # 예외 처리
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/log")
async def log_search(search_log: SearchLog):
    """
    사용자의 검색 활동을 데이터베이스에 기록합니다.
    """
    try:
        # 데이터베이스에 검색 로그 기록
        log_id = log_search_query(search_log.query, search_log.results_count)
        if log_id is None:
            raise HTTPException(status_code=500, detail="검색 로그 기록에 실패했습니다.")
        
        return {"message": "검색 로그가 성공적으로 기록되었습니다.", "log_id": log_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 하이브리드 검색 질의 분리 API 엔드포인트
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

# 서버 실행을 위한 uvicorn 명령어 (터미널에서 실행):
# uvicorn main:app --reload
