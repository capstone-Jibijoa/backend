# 이 파일은 데이터베이스 테이블 생성을 위한 SQL 쿼리를 저장합니다.
# 기존 테이블이 존재할 경우 삭제하고 다시 생성하여 데이터를 초기화합니다.

# panels_master 테이블은 고객의 핵심 검색 데이터를 JSONB 형식으로 저장합니다.
CREATE_PANELS_MASTER_TABLE = """
CREATE TABLE IF NOT EXISTS panels_master (
    uid SERIAL PRIMARY KEY, -- 고객 고유 ID
    ai_insights JSONB,      -- AI 정제된 핵심 검색 데이터
    structured_data JSONB,  -- 원본 정형 데이터 (참조용)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, -- 생성 시각
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP  -- 수정 시각
);
"""

# 사용자 검색 로그 테이블
CREATE_SEARCH_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS search_log (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    results_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    uid INTEGER,
    is_uncertain BOOLEAN DEFAULT FALSE, -- 시스템이 검색 불확실성을 감지하여 후속 질문을 제안했는지 여부
    suggested_question TEXT,          -- 제안된 후속 질문 텍스트
    user_response TEXT,               -- 사용자의 실제 응답 내용  
    search_time_ms INTEGER,   -- 검색 처리에 소요된 시간 (밀리초 단위)
    FOREIGN KEY (uid) REFERENCES panels_master(uid) ON DELETE SET NULL
);
"""