import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# 각 모듈에서 필요한 함수들을 임포트합니다.
from bedrock_logic import process_hybrid_query, split_query_for_hybrid_search
from db_logic import log_search_query, query_database_with_hybrid_search
from analysis_logic import analyze_search_results # 👈 최종 분석 로직 추가

# FastAPI 애플리케이션 초기화
app = FastAPI(title="Hybrid Search & Analysis API")

# API 요청 및 응답 본문 모델 정의

class SearchQuery(BaseModel):
    query: str

class SearchLog(BaseModel):
    query: str
    results_count: int

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    structured_condition: str
    semantic_condition: str


# ====================================================================
# 1. 메인 검색 및 분석 API 엔드포인트
# ====================================================================
@app.post("/api/search")
async def search_products(search_query: SearchQuery):
    """
    자연어 검색 요청을 처리하고 하이브리드 검색 결과를 분석하여 반환합니다.
    """
    query_text = search_query.query
    
    try:
        # 1. 질의 분리 및 임베딩 벡터 생성 (Bedrock Logic)
        processed_query_data = process_hybrid_query(query_text)
        
        # 2. 정형 조건과 임베딩 벡터 추출
        structured_condition = processed_query_data["structured_condition"]
        embedding_vector = processed_query_data["embedding_vector"]

        # 3. 하이브리드 검색 실행 (DB Logic)
        # top_k=150으로 설정하여 분석에 필요한 최대 데이터를 가져옵니다.
        search_results = query_database_with_hybrid_search(
            structured_condition,
            embedding_vector,
            top_k=150
        )

        if search_results is None:
            # DB 연결 실패 시, 이미 db_logic에서 에러 메시지가 출력되었을 수 있습니다.
            raise HTTPException(status_code=500, detail="데이터베이스 검색에 실패했습니다. DB 연결 또는 쿼리 실패를 확인하세요.")

        # 4. 검색 결과 분석 (Analysis Logic)
        # 검색된 데이터를 Claude 3 Opus 모델에게 전달하여 분석 보고서를 생성합니다.
        analysis_report = analyze_search_results(query_text, search_results)
        
        # 분석 실패 시 (LLM이 JSON 형식을 지키지 않았거나 오류 발생 시)
        if analysis_report.get("error") or analysis_report.get("raw_output"):
            log_search_query(query_text, len(search_results))
            # Bedrock API 호출 실패 또는 파싱 실패를 상세히 명시
            raise HTTPException(status_code=500, detail="검색 결과 분석(LLM)에 실패했습니다. Bedrock API 응답 및 파싱 로직을 확인하세요.")
        
        # 5. 검색 로그 기록 (DB Logic)
        log_search_query(query_text, len(search_results))

        # 6. 최종 분석 결과를 JSON 형태로 반환
        return {
            "query": query_text,
            "results_count": len(search_results),
            "analysis_report": analysis_report,
        }

    except HTTPException as e:
        # FastAPI HTTPException은 그대로 다시 발생시킵니다.
        raise e
    except Exception as e:
        # 기타 예상치 못한 예외 처리
        print(f"하이브리드 검색 및 분석 통합 실패: {e}")
        raise HTTPException(status_code=500, detail=f"하이브리드 검색 및 분석 통합 실패: {str(e)}")


# ====================================================================
# 2. 검색 로그 기록 엔드포인트
# ====================================================================
@app.post("/api/search/log")
async def log_search(search_log: SearchLog):
    """
    사용자의 검색 활동을 데이터베이스에 기록합니다.
    """
    try:
        log_id = log_search_query(search_log.query, search_log.results_count)
        if log_id is None:
            raise HTTPException(status_code=500, detail="검색 로그 기록에 실패했습니다.")
        
        return {"message": "검색 로그가 성공적으로 기록되었습니다.", "log_id": log_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# 3. 질의 분리 디버깅 엔드포인트
# ====================================================================
@app.post("/split", response_model=QueryResponse)
async def split_query(request: QueryRequest):
    """
    POST 요청으로 받은 자연어 쿼리를 정형 조건과 의미론적 조건으로 분리합니다. (디버깅용)
    """
    try:
        # bedrock_logic.py에 있는 split_query_for_hybrid_search 함수 호출
        result = split_query_for_hybrid_search(request.query)
        return QueryResponse(
            structured_condition=result["structured_condition"],
            semantic_condition=result["semantic_condition"]
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 분리 실패: {str(e)}")
    
# ----------------------------------------------------
# 🌟 요청 처리 엔드포인트 정의
# ----------------------------------------------------
# 예시: 하이브리드 검색 쿼리를 처리하는 엔드포인트
@app.post("/process_query")
async def handle_query(query: str):
    try:
        # 이전에 작성하신 process_hybrid_query 함수 호출
        result = process_hybrid_query(query) 
        return {"status": "success", "data": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")

# ----------------------------------------------------
# 🌟 루트 경로 '/' 정의 (선택 사항이지만, 404를 없애기 위해 권장)
# ----------------------------------------------------
@app.get("/")
def read_root():
    return {"Hello": "Welcome to the Hybrid Search API"}