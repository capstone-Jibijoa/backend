# --------------------
# 1. 빌드 스테이지 (Build Stage)
# 의존성 설치 및 빌드를 담당하여 최종 이미지 크기를 줄입니다.
# --------------------

FROM python:3.11-slim as builder

# 1.1 환경 설정
WORKDIR /app

# 1.2 시스템 종속성 설치 (PostgreSQL client library 등)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 1.3 의존성 설치
COPY requirements.txt .

# 1.4 가상 환경 생성 및 의존성 설치
# python -m venv를 사용하여 올바른 가상환경 생성
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# 1.5 uvicorn 설치 확인
RUN /venv/bin/uvicorn --version

# --------------------
# 2. 프로덕션 스테이지 (Production Stage)
# 최종 실행 환경. 빌드 스테이지에서 설치된 의존성만 가져옵니다.
# --------------------

FROM python:3.11-slim

# 2.1 환경 설정
WORKDIR /app

# 2.2 런타임 시스템 종속성 설치 (psycopg2 실행에 필요)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 2.3 비-루트 사용자 생성
RUN adduser --system --no-create-home appuser

# 2.4 작업 디렉토리의 소유권 변경
RUN chown -R appuser /app

# 2.5 빌드 스테이지에서 가상 환경 복사
COPY --from=builder /venv /venv

# 2.6 환경 변수 설정
ENV PATH="/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1

# 2.7 애플리케이션 코드 복사
COPY --chown=appuser . .

# 2.8 비-루트 사용자 전환
USER appuser

# 2.9 포트 노출
EXPOSE 8000

# 2.10 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 2.11 실행 명령어
CMD ["/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]