## 📄 모듈 관계 분석 
### 🚀 프로젝트 개요

이 프로젝트는 **FastAPI**와 **LangChain**을 기반으로 구축된 하이브리드 검색 및 분석 API입니다. 사용자의 자연어 쿼리를 받아 정형 데이터 필터링(PostgreSQL)과 비정형 의미론적 검색(Qdrant)을 결합하여 결과를 도출하고, LLM(Claude 3)을 통해 결과를 심층 분석합니다.

### 🧩 모듈 관계 및 역할
프로젝트의 핵심 모듈들은 다음과 같은 의존성을 가집니다.

### main.py

- **역할**: FastAPI 애플리케이션의 진입점(Entry Point)이자 전체 워크플로우를 조율하는 오케스트레이터.
- **주요 기능**: `/api/search` 엔드포인트를 통해 요청을 받고, 각 모듈의 함수를 순서대로 호출하여 최종 결과를 반환합니다.
- **의존성**: `hybrid_logic.py`, `langchain_search_logic.py`, `analysis_logic.py`, `db_logic.py` (로그 기록용)

### hybrid_logic.py

- **역할**: 사용자 질의(Query) 분석 및 변환 담당.
- **주요 기능**: Claude 3 Sonnet 모델을 사용하여 자연어 질의를 **정형 필터 조건(JSON)**과 **비정형 의미론적 검색어(Text)**로 분리합니다.
- **의존성**: `langchain-anthropic`

### langchain_search_logic.py

- **역할**: **LangChain 기반의 핵심 하이브리드 검색 로직**을 담당합니다.
- **주요 기능**: LangChain Expression Language(LCEL)를 사용하여 검색 파이프라인을 구성합니다. PostgreSQL에서 정형 필터링을 수행하고, 그 결과를 바탕으로 Qdrant에서 벡터 검색을 실행하여 최종 데이터 목록을 반환합니다.
- **의존성**: `langchain`, `db_logic.py` (DB 연결용)

### db_logic.py

- **역할**: 데이터베이스 연결 및 보조 기능 제공.
- **주요 기능**: PostgreSQL 및 Qdrant 클라이언트 연결 객체를 제공하고, 검색 로그를 DB에 기록하는 함수(`log_search_query`)를 포함합니다.
- **의존성**: `psycopg2`, `qdrant-client`, `spl_queries.py`

### spl_queries.py

- **역할**: SQL 쿼리 저장소.
- **주요 기능**: `CREATE TABLE`과 같은 고정 SQL 구문을 변수로 저장합니다.
- **의존성**: 없음.

### analysis_logic.py

- **역할**: 검색 결과에 대한 LLM 기반 심층 분석 담당.
- **주요 기능**: `langchain_search_logic`으로부터 받은 검색 결과를 Claude 3 Opus 모델에 전달하여, 요약 및 시각화용 분석 데이터를 JSON 형식으로 생성합니다.
- **의존성**: `langchain-anthropic`

### 🗺️ 모듈 간 의존성 요약

`main.py` ─────┬──> `hybrid_logic.py` (질의 분리)
             ├─> `langchain_search_logic.py` (하이브리드 검색) ──> `db_logic.py` (연결) ──> `spl_queries.py`
             ├─> `analysis_logic.py` (결과 분석)
             └─> `db_logic.py` (로그 기록)

이 구조는 각 모듈의 역할을 명확히 분리하고, 유지보수 및 확장성을 높이는 데 기여합니다.

### 로직 흐름 (LangChain 기반)
[사용자 질의] in `main.py`
     |
     v
`split_query_for_hybrid_search()` in `hybrid_logic.py` --> [정형 조건(JSON)] & [비정형 텍스트]
     |
     v
`langchain_hybrid_chain.invoke()` in `langchain_search_logic.py`
     |
     +--> 1. `_get_filtered_uids_from_postgres()` (PostgreSQL에서 UID 필터링)
     +--> 2. `retriever.get_relevant_documents()` (Qdrant에서 필터링된 UID로 벡터 검색)
     +--> 3. `_extract_payload_from_documents()` (Qdrant 결과에서 최종 데이터 추출)
     |
     v
[최종 검색 결과 리스트]

### 가상환경 관련 설정
활성화 => .\venv\Scripts\activate
비활성화 => deactivate

### API 서버 실행
uvicorn main:app --reload
