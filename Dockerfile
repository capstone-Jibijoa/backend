# --------------------

# 1. 빌드 스테이지 (Build Stage)

# 의존성 설치 및 빌드를 담당하여 최종 이미지 크기를 줄입니다.

# --------------------

FROM python:3.11-slim as builder

# 1.1 환경 설정
# 컨테이너 내에서 작업을 수행할 디렉토리를 설정
WORKDIR /app

# 1.1.5 시스템 종속성 설치 (PostgreSQL client library 등)
# postgresql-dev는 psycopg2 등 DB 라이브러리 빌드에 필요
# build-essential은 pip 설치 시 필요한 컴파일러 도구
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 1.2 의존성 설치
# requirements.txt 파일을 복사
COPY requirements.txt .

# 1.3 가상 환경 생성 및 의존성 설치
# 빌드 시에만 필요한 툴과 의존성을 설치
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/venv -r requirements.txt

# --------------------
# 2. 프로덕션 스테이지 (Production Stage)
# 최종 실행 환경. 빌드 스테이지에서 설치된 의존성만 가져옵니다.
# --------------------

FROM python:3.11-slim

# 2.1 환경 설정
WORKDIR /app

# ⚠️ 수정됨: 비-루트 사용자 생성
# 사용자만 생성합니다. 대부분의 기본 OS 이미지에는 동일한 이름의 그룹이 자동으로 생성되지 않습니다.
RUN adduser --system --no-create-home appuser

# ⚠️ 수정됨: 작업 디렉토리의 소유권을 appuser로 변경
# 그룹 소유권(콜론과 그룹명)을 생략하여 chown: invalid group 오류를 방지합니다.
RUN chown -R appuser /app

# 2.2 빌드 스테이지에서 가상 환경 복사
# /venv 디렉토리 전체를 복사 (설치된 모든 의존성이 포함됨)
COPY --from=builder /venv /venv

# 2.3 PATH 환경 변수 설정
# venv의 bin 디렉토리를 PATH에 추가하여 설치된 명령어를 바로 사용 가능하게 함
ENV PATH="/venv/bin:$PATH"

# 2.4 애플리케이션 코드 복사
# 프로젝트 코드 전체를 복사합니다.
# ⚠️ 수정됨: 소유권 변경 시에도 그룹을 생략하여 오류를 방지합니다.
COPY --chown=appuser . .

# 2.5 비-루트 사용자 전환
USER appuser

# 2.6 포트 노출
EXPOSE 8000

# 2.7 실행 명령어
# Gunicorn을 사용하여 FastAPI 애플리케이션을 실행합니다.
ENV GUNICORN_WORKERS 4

CMD ["gunicorn", "main:app", "--workers", "${GUNICORN_WORKERS}", "--bind", "0.0.0.0:8000", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120"]