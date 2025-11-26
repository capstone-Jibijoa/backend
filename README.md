# 🔍 Multi-Table Hybrid Search System (Backend)

이 프로젝트는 **FastAPI 기반 하이브리드 검색 엔진**으로,

**PostgreSQL(정형 데이터)**과 **Qdrant(벡터 기반 비정형 데이터)**를 결합하여

사용자의 자연어 질의를 SQL + Vector 검색으로 동시에 처리합니다.

---

## 📂 프로젝트 구조 (Project Structure)

백엔드는 **Layered Architecture**를 따릅니다.

```
├── api/                        # 🌐 Entry Points (Router)
│   ├── analysis.py             # [Pro 모드] 심층 분석 및 통계 요약 엔드포인트
│   ├── panels.py               # 패널 상세 정보 조회
│   ├── search.py               # [Lite 모드] 검색 엔드포인트
│   └── router.py               # API 라우터 통합
│
├── core/                       # ⚙️ Core Configuration & Singletons
│   ├── embeddings.py           # HuggingFace 임베딩 모델 (Singleton)
│   ├── llm_client.py           # Claude LLM 클라이언트
│   ├── semantic_router.py      # 질의 Intent → Target Field 매핑
│   └── settings.py             # 환경변수, AWS Secrets Manager
│
├── services/                   # 🧠 Business Logic
│   ├── search_service.py       # SQL + Vector + Reranking 하이브리드 검색 로직
│   ├── analysis_service.py     # 통계 분석 및 차트 생성
│   ├── llm_prompt.py           # LLM 프롬프트 / 자연어 → JSON 파싱
│   ├── llm_summarizer.py       # 인사이트 요약 생성
│   └── panel_service.py        # 패널 정보 조립
│
├── repositories/               # 💾 Data Access Layer
│   ├── panel_repo.py           # PostgreSQL 조회
│   ├── qpoll_repo.py           # Qdrant 벡터 검색
│   └── log_repo.py             # 검색 로그 적재
│
├── schemas/                    # 📝 Pydantic Models (DTO)
│   ├── search.py
│   └── analysis.py
│
├── utils/                      # 🛠️ Utilities
│   └── common.py               # 정규식 필터링, 텍스트 전처리 등
│
└── constants/
    └── mapping.py              # 필드명 매핑, 카테고리, 키워드 규칙

```

---

# 🏗️ 아키텍처 개요 (Architecture Overview)

사용자의 자연어 입력을 다음 단계로 처리합니다:

1. **Query Parsing (LLM)**
2. **Structured Filter (PostgreSQL)**
3. **Vector Semantic Search (Qdrant)**
4. **Hybrid Merge + Reranking**
5. **Data Assembly & Response**

---

## 🔁 전체 데이터 흐름 (Data Flow)

```
User Query (자연어)
        ↓
LLM Parser (Query → Structured JSON)
        ↓
Search Controller (Logic Branching)
        ├── Path A: SQL Filtering (PostgreSQL)
        └── Path B: Vector Semantic Search (Qdrant)
                ↓
Intersection & Reranking
        ↓
Aggregation & Formatting
        ↓
API Response

```

---

# 🚀 검색 워크플로우 상세 (Search Workflow)

### 예시 질의

> "서울 사는 30대 중 주말에 OTT를 즐겨보고 고양이를 안 키우는 사람 찾아줘"
> 

---

## 1. 질의 분석 및 파싱 (Query Understanding)

**담당:** `llm_prompt.py`, `semantic_router.py`

- 자연어 → JSON 변환 (LLM)
- SQL에서 처리 가능한 항목 정리
    - 나이, 성별, 지역 등 **Demographic Filters**
- 의미 기반 조건 추출
    - 긍정: “OTT 즐겨봄”
    - 부정: “고양이를 안 키움”
- Target Field 결정
    - 벡터 유사도 + 키워드로 가장 관련된 질문/문항 탐지

---

## 2. 정형 데이터 필터링 (SQL Pre-filtering)

**담당:** `panel_repo.py`

**사용 DB:** PostgreSQL `welcome_meta2 (JSONB)`, 'qpoll_meta'

- 나이대 변환
- 지역명/성별/직업 매핑
- 동의어 처리: (“남자” → Male)
- SQL WHERE로 후보군 패널 목록 확보

---

## 3. 벡터 검색 + 하이브리드 알고리즘

**담당:** `search_service.py`

**사용 DB:** Qdrant

| Case | 설명 | 처리 방식 |
| --- | --- | --- |
| A | 정형 조건만 있음 | SQL Only |
| B | 정형 + 비정형 조건 | SQL 후보군 위에 Vector Reranking |
| C | 정형 조건 없음 | Global Vector Search |
| D | 부정 조건 있음 | 부정 벡터와 유사한 패널을 **제외** |

사용 컬렉션:

- `qpoll_vectors_v2`
- `welcome_subjective_vectors`

---

## 4. 사후 필터링 및 결과 조립 (Post-Processing)

**담당:** `search_service.py`, `qpoll_repo.py`

- **Strict Filtering:**
    
    Regex 기반으로 “없음”, “관심 없음” 등 부정 응답 직접 제거
    
- **Data Merge:**
    
    확정된 panel_id 기준으로 PostgreSQL + Qdrant 병렬 조회
    
- **Dynamic Column Selection:**
    
    질의와 가장 관련성 높은 컬럼만 선택하여 응답 구성
    

---

# 📊 데이터베이스 스키마 (Database Schema)

## PostgreSQL (Structured Data)

### `welcome_meta2`

패널의 주요 인구통계 정보 저장

- `panel_id (PK)`
- `structured_data (JSONB)`
- `created_at`, etc.

### `qpoll_meta`

설문 메타데이터 저장

---

## Qdrant (Vector Data)

### `qpoll_vectors_v2`

- 패널의 설문 문항/응답 벡터 저장
- Payload:
    - `panel_id`
    - `question`
    - `sentence`
    - `vector`

### `welcome_subjective_vectors`

- 주관식 응답(가전보유, 차종 등) 벡터화 저장

---

# 📊 Insight Generation Logic

사용자의 검색 결과를 기반으로 의미 있는 인사이트를 자동으로 시각화하기 위한 **우선순위 기반 차트 생성 알고리즘**입니다. 단순 데이터 나열이 아닌, *새로운 정보·의미 있는 패턴*을 우선 제공하는 것을 목표로 합니다.

---

## 🎯 Core Principle: Obviousness Filtering

"**이미 사용자가 알고 있는 정보는 제외하고, 새로운 인사이트를 우선적으로 제공한다.**"

- 검색 조건에 이미 포함된 필드(예: 사용자가 “20대”로 검색 → 연령 분포 차트 제외)
- 응답이 극단적으로 한쪽으로 쏠린 경우(최대값 95% 이상) → 정보 가치 낮음 → 제외

---

## ⚙️ Chart Priority Algorithm

차트 생성 시 우선순위를 **Priority Queue 방식**으로 적용합니다.

### **Priority 0 — Derived / Mandatory Charts**

| 유형 | 설명 |
| --- | --- |
| 세부 지역(region_minor) | 광역 지역(region_major)로 필터링된 경우 자동 생성 |
| Target Field Charts | 사용자의 질의 의도가 특정 필드를 지목할 때 최우선 노출 |

---

### **Priority 1 — Semantic-Based Charts**

- 질의 키워드와 연관된 **Semantic Fields** 자동 분석
    - 예: “운동하는 사람”, “여행 자주 가는 사람” → 해당 semantic 필드 우선 노출
- 차량 관련:
    - *차량 소유 비율 ≥ 70%**일 때 → 차종 분포 차트 자동 추가

---

### **Priority 2 — Core Demographics**

- 성별(gender), 연령대(age), 지역(region) 등 기본 인구통계 정보
- 단, **검색 조건에 이미 사용된 필드는 제외**
    - 예: age=20대로 검색한 경우 → age 차트 생략

---

### **Priority 3 — Crosstab Analysis**

상위 Priority 차트가 5개 미만일 경우 실행.

- Target Field × 주요 demographic(성별, 연령대) 교차 분석
- 예: "OTT 보는 20대" 검색 시
    - "OTT 장르 선호 × 성별"
    - "OTT 시청 시간 × 연령대"

---

### **Priority 4 — High-Ratio Feature Discovery**

데이터 전체를 스캔해 다음 조건을 만족하는 필드를 자동 발굴:

- 특정 응답 비율이 **40% ~ 95% 사이**
- 한쪽으로 치우쳤지만 완전한 쏠림(95%↑)은 아님
- 뚜렷한 특징을 가진 필드를 자동 차트화

---

## 🚫 Smart Exclusion Rules

다음 조건에 해당하는 차트는 자동 제외됩니다.

### 1. 검색 필터 중복

- 사용자가 검색 조건(demographic_filters)에 명시한 필드
    - 예: age, gender, region 등 → 중복 차트 제외

### 2. 논리적 중복

- 서로 높은 상관관계로 인해 의미 없는 차트

| 필터 | 제외되는 차트 | 이유 |
| --- | --- | --- |
| children_count 존재 | marital_status 제외 | 기혼 확률 99% |
| car_model 조건 존재 | car_ownership 제외 | 차종 지정 = 소유 확정 |

### 3. 데이터 쏠림

- 특정 응답 비율이 **95% 이상**
    
    → 분석 가치 없음 → 제외
    
---

## 📦 Summary

- 우선순위 기반으로 *의미 있는 차트만* 자동 생성
- 사용자가 이미 설정한 조건은 제외
- 극단적으로 편향된 값도 제외
- 의미 있는 패턴, semantic 연관 데이터, 교차 분석은 우선적으로 제공

---

# 🛠️ 기술 스택 (Tech Stack)

### **Framework**

- FastAPI
- Uvicorn

### **Database**

- PostgreSQL (psycopg2 pool)
- Qdrant (Vector DB)

### **LLM & AI**

- LangChain
- Anthropic Claude 3 (Haiku / Sonnet)
- HuggingFace Embeddings (nlpai-lab/KURE-v1)

### **Utilities**

- Pandas
- NumPy
- Scikit-learn (Cosine Similarity)

---

## 🛠️ 실행 방법

#### 1. 가상환경 활성화
.\venv\Scripts\activate

#### 2. API 서버 실행
uvicorn main:app --reload

uvicorn main:app --reload --log-config log_config.json

