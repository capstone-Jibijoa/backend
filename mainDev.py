# FastAPI 웹 서버의 메인 파일.
# 전체 시스템의 시작점이자, 각 모듈을 연결하는 파일

# 자연어 검색 POST (/api/search) 요청을 처리하는 엔드포인트를 포함합니다.
# 검색 로그 기록 POST (/api/search/log) 요청을 처리하는 엔드포인트도 포함합니다.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bedrock_logic import get_embedding_from_bedrock
from db_logic import query_database_with_vector, log_search_query

app = FastAPI()

# 요청 본문을 위한 Pydantic 모델 정의
class SearchQuery(BaseModel):
    query: str

class SearchLog(BaseModel):
    query: str
    results_count: int

@app.post("/api/search")
async def search_products(search_query: SearchQuery):
    """
    자연어 검색 요청을 처리하고 검색 결과를 반환합니다.
    """
    try:
        # 1. Bedrock을 호출하여 검색어의 임베딩 벡터 생성
        embedding_vector = get_embedding_from_bedrock(search_query.query)
        if embedding_vector is None:
            raise HTTPException(status_code=500, detail="임베딩 벡터 생성에 실패했습니다.")

        # 2. 임베딩 벡터를 사용하여 데이터베이스에서 유사한 상품 검색
        search_results = query_database_with_vector(embedding_vector)
        if search_results is None:
            raise HTTPException(status_code=500, detail="데이터베이스 검색에 실패했습니다.")

        # 3. 검색 결과를 JSON 형태로 반환
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

# 서버 실행을 위한 uvicorn 명령어 (터미널에서 실행):
# uvicorn main:app --reload
