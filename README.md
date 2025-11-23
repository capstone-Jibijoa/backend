# 🚀 Multi-Table Hybrid Search API v3 (Optimized)

**FastAPI**, **Anthropic Claude 3 Haiku**, **Qdrant**를 기반으로 구축된 고성능 하이브리드 검색 및 분석 엔진입니다.
사용자의 자연어 질의를 분석하여 **SQL 기반의 정형 필터링**과 **Vector 기반의 의미 검색**을 결합하고, **부정 조건(Negative Filtering)**까지 정밀하게 제어하여 빠르고 정확한 결과를 제공합니다.

---

## ⚡️ 핵심 최적화 (v3.0 Highlights)

이전 버전 대비 **속도**와 **정확도** 측면에서 대폭적인 개선이 이루어졌습니다.

1.  **🚀 응답 속도 극대화 (10s → 3s 이내)**
    * **LLM 경량화**: 쿼리 파싱 모델을 `Claude 3.5 Sonnet`에서 **`Claude 3 Haiku`**로 교체하여 의도 분석 속도를 3배 이상 향상시켰습니다.
    * **Non-blocking Architecture**: 무거운 검색 로직(`search.py`)을 `asyncio.to_thread`로 별도 스레드에서 실행하여 Event Loop 차단을 방지합니다.
    * **Parallel Data Fetching**: PostgreSQL(테이블 데이터)과 Qdrant(설문 데이터) 조회를 `asyncio.gather`로 **병렬 실행**하여 대기 시간을 단축했습니다.

2.  **🎯 정확도 및 필터링 강화**
    * **Strict Negative Filtering**: "없음", "안 함" 등의 부정 답변을 **정규식(Regex)**과 **벡터 유사도(Vector)** 이중 검증으로 완벽하게 제외합니다.
    * **Smart Column Selection**: 검색 의도에 맞춰 사용자에게 보여줄 테이블 컬럼을 동적으로 최적화합니다.

3.  **⚙️ 리소스 효율성**
    * **Singleton Pattern**: DB Connection Pool 및 Qdrant Client를 전역 싱글톤으로 관리하여 연결 오버헤드를 제거했습니다.
    * **Caching**: `lru_cache`를 활용하여 설정 및 임베딩 모델 로딩을 최적화했습니다.

---

## 🛠️ 시스템 아키텍처 & 워크플로우

사용자가 질의(Query)를 입력했을 때의 데이터 처리 흐름입니다.

### 1. 🧠 Query Understanding (`llm.py`)
* **Role**: 입력된 자연어를 `Claude 3 Haiku`가 분석하여 구조화된 JSON으로 변환합니다.
* **Output**:
    * `Demographic Filters` (SQL): 나이, 지역, 성별 등 인구통계 조건.
    * `Semantic Conditions` (Vector): 취향, 라이프스타일, 소비 패턴 등.
    * `Negative Flags`: 제외해야 할 조건 식별 (`is_negative: true`).

### 2. 🔍 Hybrid Search Engine (`search.py`, `search_helpers.py`)
* **Step 1. SQL Filtering (Pre-filtering)**: PostgreSQL `welcome_meta2` 테이블에서 인구통계 조건에 맞는 `panel_id` 후보군을 1차적으로 추출합니다.
* **Step 2. Vector Search**: 추출된 후보군을 대상으로 Qdrant(`qpoll_vectors_v2` 등)에서 의미 기반 검색을 수행합니다.
* **Step 3. Strict Validation**:
    * **Text Filter**: `STRICT_NEGATIVE_PATTERNS` (정규식)을 사용해 부정적인 텍스트 답변을 강제 제외합니다.
    * **Vector Filter**: LLM이 식별한 부정 조건과 유사한 벡터를 가진 패널을 2차로 제외합니다.

### 3. 📊 Analysis & Aggregation (`insights.py`, `main.py`)
* **Analysis**: 검색된 패널들의 답변을 군집화(DBSCAN)하여 주요 특징을 분석하고 시각화 데이터(Chart)를 생성합니다.
* **Aggregation**: `main.py`에서 비동기 병렬 처리로 최종 테이블 데이터와 분석 결과를 조립하여 반환합니다.

---

## 📂 주요 모듈 설명

| 파일명 | 역할 및 핵심 기능 |
| :--- | :--- |
| **`main.py`** | **API Entrypoint & Async Controller**<br>- `/api/search` 등 엔드포인트 정의.<br>- `asyncio.to_thread`, `asyncio.gather`를 통한 비동기/병렬 처리 오케스트레이션. |
| **`search.py`** | **Search Logic Core**<br>- 하이브리드 검색의 전체 파이프라인(SQL → Vector → Negative Filter) 제어.<br>- 정밀 필터링 로직 구현. |
| **`llm.py`** | **Query Parser**<br>- LangChain & Claude 3 Haiku를 사용하여 자연어를 필터 조건으로 파싱.<br>- 부정 조건(`is_negative`) 식별 프롬프트 최적화. |
| **`search_helpers.py`** | **Query Builder & Embeddings**<br>- JSON 필터를 PostgreSQL `WHERE` 절로 변환.<br>- HuggingFace 임베딩 모델 로드 및 관리. |
| **`db.py`** | **Database Connector**<br>- PostgreSQL Connection Pool 및 Qdrant Client 싱글톤 관리.<br>- 리소스 누수 방지 및 재사용성 보장. |
| **`insights.py`** | **Data Analyst**<br>- 검색 결과에 대한 통계, 클러스터링 분석 및 차트 데이터 생성. |
| **`mapping_rules.py`** | **Knowledge Base**<br>- "MZ세대", "고소득" 같은 키워드 매핑 규칙 및 설문 템플릿 정의. |

---

## 💻 설치 및 실행 (Setup)

### 1. 환경 변수 설정 (.env)
프로젝트 루트에 `.env` 파일을 생성하고 다음 정보를 입력하세요.
```ini
# AWS / Database
DB_HOST=localhost
DB_NAME=your_db
DB_USER=postgres
DB_PASSWORD=your_password

# Vector DB
QDRANT_HOST=localhost
QDRANT_PORT=6333

# AI Models
ANTHROPIC_API_KEY=''

# 의존성 설치
pip install -r requirements.txt

### 🛠️ 실행 방법

#### 1. 가상환경 활성화
.\venv\Scripts\activate

#### 2. API 서버 실행
uvicorn main:app --reload
uvicorn main:app --reload --log-config log_config.json

### 🔍 API Endpoints
* POST /api/search (Lite Mode)

** 빠른 응답 속도 중시. 검색 결과 리스트와 테이블 데이터 반환.

*POST /api/search-and-analyze (Pro Mode)

** 심층 분석 모드. 검색 결과와 함께 통계 차트(Charts) 및 인사이트 제공.

* GET /health

** 서버 및 DB 연결 상태 확인.