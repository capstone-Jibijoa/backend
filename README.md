## 📄 모듈 관계 분석 
### 🚀 프로젝트 개요

이 프로젝트는 FastAPI 기반의 애플리케이션으로, 사용자의 자연어 쿼리를 처리하고 데이터베이스에 저장된 정보와 상호작용하는 기능을 제공합니다. 시스템은 명확한 역할 분담을 위해 여러 모듈로 나뉘어 있습니다.

### 🧩 모듈 관계 및 역할
프로젝트의 핵심 모듈들은 다음과 같은 의존성을 가집니다.

### main.py

역할: FastAPI 애플리케이션의 진입점(Entry Point)입니다.
주요 기능: API 엔드포인트(/api/search 등)를 정의하고, 전체 검색-분석 워크플로우를 조율하는 오케스트레이터(Orchestrator) 역할을 합니다.
의존성: `hybrid_logic.py`, `db_logic.py`, `analysis_logic.py`에 의존합니다.

### hybrid_logic.py

역할: 자연어 처리 및 AI 모델 로직을 담당합니다.
주요 기능: Claude 3 모델을 사용하여 사용자 질의를 정형/비정형 조건으로 분리하고, KURE 임베딩 모델을 사용해 비정형 조건을 벡터로 변환합니다.
의존성: `anthropic`, `sentence-transformers` 라이브러리에 의존합니다.

### db_logic.py

역할: 데이터베이스 및 벡터 DB 관련 로직을 모두 처리합니다.
주요 기능: PostgreSQL에서 정형 데이터를 필터링하고, Qdrant 벡터 DB에서 의미론적 검색을 수행합니다. 두 결과를 통합하여 최종 데이터를 반환하며, 검색 로그도 기록합니다.
의존성: `psycopg2`, `qdrant-client` 라이브러리를 사용하며, `spl_queries.py`에서 SQL 쿼리를 가져옵니다.

### spl_queries.py

역할: SQL 쿼리 저장소입니다.
주요 기능: 데이터베이스 테이블 생성을 위한 CREATE TABLE SQL 구문을 문자열 변수로 저장하고 제공합니다.
의존성: 다른 모듈에 의존하지 않는 단순 데이터 모듈입니다.

### analysis_logic.py

역할: 검색 결과 데이터 분석을 담당합니다.
주요 기능: `db_logic`을 통해 얻은 최종 데이터셋을 Claude 3 Opus 모델에 전달하여, 사용자 질의에 맞는 심층 분석 리포트(요약, 연관 토픽, 시각화 데이터 등)를 생성합니다.
의존성: `anthropic` 라이브러리에 의존합니다.

### 🗺️ 모듈 간 의존성 요약
다음은 각 모듈 간의 의존성을 도식화한 것입니다.

main.py ─────┬──> hybrid_logic.py
             ├──> db_logic.py ─────> spl_queries.py
             └──> analysis_logic.py

이 구조는 각 모듈의 역할을 명확히 분리하고, 유지보수 및 확장성을 높이는 데 기여합니다.

### 로직 흐름 (LangChain 기반)
[사용자 질의] in `main.py`
     |
     v
`split_query_for_hybrid_search()` in `hybrid_logic.py` --> [정형 조건] & [비정형 텍스트]
     |
     v
`langchain_hybrid_chain.invoke()` in `langchain_search_logic.py`
     |
     +--> 1. `_get_filtered_uids_from_postgres()` (PG에서 UID 필터링)
     +--> 2. `retriever.get_relevant_documents()` (Qdrant에서 벡터 검색)
     +--> 3. `_get_final_data_from_postgres()` (PG에서 최종 데이터 조회)
     |
     v
[최종 검색 결과 리스트]

### 가상환경 관련 설정
활성화 => .\venv\Scripts\activate
비활성화 => deactivate

### API 서버 실행
uvicorn main:app --reload
