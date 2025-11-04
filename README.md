## 📄 하이브리드 검색 및 분석 API: 모듈 관계 분석

### 🚀 프로젝트 개요

이 프로젝트는 **FastAPI**와 **LangChain**을 기반으로 구축된 고성능 하이브리드 검색 및 분석 API입니다. 사용자의 자연어 질의를 LLM(Claude 3)을 통해 **정형 필터**와 **비정형 검색어**로 분리하고, 이를 바탕으로 PostgreSQL과 Qdrant에서 복합 검색을 수행합니다. 최종적으로 검색 결과를 다시 LLM으로 심층 분석하여 구조화된 리포트를 생성합니다.

특히, 서버 재시작 없이 LangChain의 핵심 구성 요소(임베딩 모델, 벡터 저장소 등)를 동적으로 재로딩하는 '엔진 교체' 기능을 통해 운영 유연성을 극대화했습니다.

### 🧩 모듈 관계 및 역할

#### `main.py`

- **역할**: FastAPI 애플리케이션의 진입점(Entry Point)이자 전체 워크플로우를 조율하는 오케스트레이터.
- **주요 기능**:
    - `/api/search`: 핵심 검색 및 분석 API 엔드포인트.
    - `/admin/reload-langchain`: 서버 재시작 없이 LangChain 구성 요소를 다시 로드하는 관리자용 API.
    - 각 모듈의 함수를 순서대로 호출하여 최종 결과를 반환합니다.
- **의존성**: `hybrid_logic.py`, `langchain_search_logic.py`, `analysis_logic.py`, `db_logic.py`

#### `hybrid_logic.py`

- **역할**: 사용자 질의(Query) 분석 및 구조화 담당.
- **주요 기능**: Claude 3 Opus 모델을 사용하여 자연어 질의를 **정형 필터 조건(JSON)**과 **비정형 의미론적 검색어(Text)**로 분리합니다.
- **의존성**: `langchain-anthropic`

#### `langchain_search_logic.py`

- **역할**: **LangChain 기반의 핵심 하이브리드 검색 실행**을 담당합니다.
- **주요 기능**:
    - HuggingFace 임베딩 모델(`nlpai-lab/KURE-v1`)과 Qdrant 벡터 저장소를 초기화하고 관리합니다.
    - LCEL(LangChain Expression Language)을 사용하여 하이브리드 검색 체인(`chain`)을 구성합니다.
    - `force_reload_langchain_components`: 외부 요청에 의해 모델과 체인을 다시 로드하는 '엔진 교체' 로직을 수행합니다.
- **의존성**: `langchain`, `qdrant-client`, `db_logic.py`

#### `db_logic.py`

- **역할**: 데이터베이스 연결 및 SQL 관련 유틸리티 제공.
- **주요 기능**:
    - PostgreSQL 및 Qdrant 클라이언트 연결 객체를 생성하고 제공합니다.
    - `_build_jsonb_where_clause`: 정형 필터(JSON)를 SQL Injection에 안전한 `WHERE` 절과 파라미터로 변환합니다.
    - `log_search_query`: 검색 로그를 PostgreSQL DB에 기록합니다.
- **의존성**: `psycopg2`, `qdrant-client`

#### `analysis_logic.py`

- **역할**: 검색 결과에 대한 LLM 기반 심층 분석 담당.
- **주요 기능**: `langchain_search_logic`으로부터 받은 검색 결과를 Claude 3 Opus 모델에 전달하여, 요약 및 시각화용 분석 데이터를 Pydantic 모델 기반의 안전한 JSON 형식으로 생성합니다.
- **의존성**: `langchain-anthropic`

### 🗺️ 모듈 간 의존성 요약

```
main.py ─────┬──> hybrid_logic.py (1. 질의 분리)
             |
             ├─> langchain_search_logic.py (2. 하이브리드 검색) ──> db_logic.py (DB 연결/쿼리)
             |
             ├─> analysis_logic.py (3. 결과 분석)
             |
             └─> db_logic.py (4. 로그 기록)
```

### ⚙️ 핵심 로직 흐름 (LangChain 기반)

1.  **[main.py]**: 사용자 질의(`query`)가 `/api/search`로 인입됩니다.
2.  **[hybrid_logic.py]**: `split_query_for_hybrid_search` 함수가 Claude 3 Opus를 호출하여 질의를 `정형 조건(JSON)`과 `비정형 검색어(Text)`로 분리합니다.
3.  **[main.py]**: 분리된 결과를 `langchain_search_logic.py`의 `invoke` 메서드에 전달합니다.
4.  **[langchain_search_logic.py]**: LCEL로 구성된 하이브리드 검색 체인이 실행됩니다.
    1.  **`_get_filtered_uids_from_postgres`**: `정형 조건`을 `db_logic`을 통해 SQL `WHERE` 절로 변환 후, PostgreSQL에서 1차 필터링된 `UID` 목록을 가져옵니다.
    2.  **`_search_qdrant_or_pass_through`**:
        - **`비정형 검색어`가 있는 경우**: 1차 필터링된 `UID`를 조건으로 Qdrant에서 벡터 검색을 수행합니다.
        - **`비정형 검색어`가 없는 경우**: Qdrant 검색을 건너뛰고, 1차 필터링된 `UID` 목록을 다음 단계로 바로 전달하여 효율성을 높입니다.
    3.  **`_get_final_data_from_postgres`**: 최종 `UID` 목록을 사용하여 PostgreSQL에서 `ai_insights` 등 상세 데이터를 조회하여 최종 검색 결과를 반환합니다.
5.  **[main.py]**: `langchain_search_logic`이 반환한 검색 결과를 `analysis_logic.py`로 전달합니다.
6.  **[analysis_logic.py]**: `analyze_search_results_chain` 함수가 Claude 3 Opus를 호출하여 검색 결과를 분석하고, 시각화에 사용될 구조화된 `JSON 리포트`를 생성합니다.
7.  **[main.py]**: 최종 분석 리포트를 클라이언트에게 반환하고, `db_logic.py`를 통해 검색 활동을 로그로 기록합니다.

### ✨ 주요 특징 및 개선 사항

*   **동적 엔진 교체**: `/admin/reload-langchain` API를 통해 서비스 중단 없이 임베딩 모델, 벡터 저장소, LangChain 체인을 다시 로드할 수 있습니다.
*   **최적화된 검색 로직**: 비정형 검색어가 없는 경우, 불필요한 Qdrant 벡터 검색을 건너뛰어 응답 속도와 비용 효율성을 개선했습니다.
*   **안전한 SQL 처리**: `psycopg2`의 파라미터 바인딩을 활용하여 SQL Injection 공격을 원천적으로 방지합니다.
*   **안정적인 LLM 출력**: `Pydantic` 모델을 LangChain의 `JsonOutputParser`와 결합하여 LLM의 분석 결과가 항상 일관된 JSON 구조를 갖도록 보장합니다.

### 🛠️ 실행 방법

#### 1. 가상환경 활성화
```bash
.\venv\Scripts\activate
```

#### 2. API 서버 실행
```bash
uvicorn main:app --reload
```
