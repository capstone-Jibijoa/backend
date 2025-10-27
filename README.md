📄 모듈 관계 분석 
🚀 프로젝트 개요
이 프로젝트는 FastAPI 기반의 애플리케이션으로, 사용자의 자연어 쿼리를 처리하고 데이터베이스에 저장된 정보와 상호작용하는 기능을 제공합니다. 시스템은 명확한 역할 분담을 위해 여러 모듈로 나뉘어 있습니다.

🧩 모듈 관계 및 역할
프로젝트의 핵심 모듈들은 다음과 같은 의존성을 가집니다.

main.py

역할: 애플리케이션의 진입점(Entry Point) 역할을 수행합니다.

주요 기능: FastAPI 애플리케이션을 실행하고, API 엔드포인트 (/api/search, /api/search/log, /split)를 정의합니다.

의존성: bedrock_logic.py와 db_logic.py에 의존하여 전체 시스템의 흐름을 조율하는 오케스트레이터(Orchestrator) 역할을 합니다.

bedrock_logic.py

역할: AI 로직 처리를 담당합니다.

주요 기능: AWS Bedrock 서비스를 이용해 사용자 쿼리 분리 및 임베딩 생성과 같은 자연어 처리 작업을 수행합니다.

의존성: 외부 라이브러리인 boto3를 통해 AWS와 통신하며, 다른 내부 모듈에 대한 의존성이 없는 독립적인 모듈입니다.

db_logic.py

역할: 데이터베이스 로직 처리를 담당합니다.

주요 기능: PostgreSQL 데이터베이스와의 모든 상호작용(연결, 테이블 생성, 쿼리 실행, 로그 저장)을 관리합니다.

의존성: psycopg2 라이브러리를 사용하며, spl_queries.py로부터 테이블 생성을 위한 SQL 쿼리를 가져옵니다.

spl_queries.py

역할: SQL 쿼리 저장소입니다.

주요 기능: 데이터베이스 테이블 생성을 위한 CREATE TABLE SQL 구문을 문자열 변수로 저장하고 제공합니다.

의존성: 다른 모듈에 의존하지 않는 단순 데이터 모듈입니다.

🗺️ 모듈 간 의존성 요약
다음은 각 모듈 간의 의존성을 도식화한 것입니다.

main.py ─────┐
            ├──> bedrock_logic.py
            └──> db_logic.py ─────> spl_queries.py
이 구조는 각 모듈의 역할을 명확히 분리하고, 유지보수 및 확장성을 높이는 데 기여합니다.

### 가상환경 관련 설정
활성화 => .\venv\Scripts\activate
비활성화 => deactivate

### API 서버 실행
uvicorn main:app --reload
