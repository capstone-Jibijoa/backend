# 이 파일은 데이터베이스 테이블 생성을 위한 SQL 쿼리를 저장합니다.

# 기존 테이블이 존재할 경우 삭제하고 다시 생성하여 데이터를 초기화합니다.
# 'vector' 확장이 활성화되어 있어야 합니다.

# panels_master 테이블과 panel_vectors 테이블을 생성합니다.

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

# panel_vectors 테이블은 Kure 임베딩 벡터를 저장하며, panels_master 테이블과 외래 키로 연결됩니다.
CREATE_PANEL_VECTORS_TABLE = """
-- pgvector 확장을 사용하기 위해 데이터베이스에 설치 필요
-- CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS panel_vectors (
    uid INTEGER PRIMARY KEY,
    embedding VECTOR(1024), -- Kure 임베딩 벡터 (N은 실제 차원)
    FOREIGN KEY (uid) REFERENCES panels_master(uid) ON DELETE CASCADE
);
"""

# 사용자 검색 로그 테이블은 ERD에 명시되지 않았으므로 그대로 유지하거나 삭제할 수 있습니다.
CREATE_SEARCH_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS search_log (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    results_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    uid INTEGER,
    FOREIGN KEY (uid) REFERENCES panels_master(uid) ON DELETE SET NULL
);
"""