import os
import json
import psycopg2
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# .env 파일에서 환경 변수 로드
load_dotenv()

# =======================================================
# 1. 환경 변수 및 상수 정의
# =======================================================
# PostgreSQL 연결 정보
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Qdrant 연결 정보
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "panels_collection")

# 임베딩 모델 정보
EMBEDDING_MODEL_NAME = "nlpai-lab/KURE-v1"

# =======================================================
# 2. 유틸리티 함수
# =======================================================

def get_db_connection():
    """PostgreSQL 데이터베이스에 연결하고 연결 객체를 반환합니다."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("✅ PostgreSQL 연결 성공")
        return conn
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL 연결 실패: {e}")
        return None

def get_qdrant_client():
    """Qdrant 클라이언트를 생성하고 반환합니다."""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print("✅ Qdrant 클라이언트 연결 성공")
        return client
    except Exception as e:
        print(f"❌ Qdrant 클라이언트 연결 실패: {e}")
        return None

def create_qdrant_collection(client: QdrantClient):
    """Qdrant에 벡터 컬렉션을 생성합니다."""
    try:
        # KURE-v1 모델의 벡터 차원은 1024입니다.
        vector_size = 1024
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        print(f"✅ Qdrant 컬렉션 '{QDRANT_COLLECTION_NAME}' 생성/재생성 완료")
    except Exception as e:
        print(f"❌ Qdrant 컬렉션 생성 실패: {e}")
        raise

def convert_insight_to_text(insight: dict) -> str:
    """
    ai_insights JSON 객체를 의미론적 검색을 위한 자연어 텍스트로 변환합니다.
    """
    # 예시: "성별은 남자, 직업은 IT•인터넷, 월 개인 소득은 월 600~699만원 입니다."
    # 프로젝트에 맞게 변환 로직을 커스터마이징 할 수 있습니다.
    parts = []
    if 'gender' in insight:
        parts.append(f"성별 {insight['gender']}")
    if 'job_duty' in insight:
        parts.append(f"직무 {insight['job_duty']}")
    if 'income_personal_monthly' in insight:
        parts.append(f"개인 소득 {insight['income_personal_monthly']}")
    if 'drinking_experience' in insight:
        parts.append(f"음주 경험 {insight['drinking_experience']}")
    if 'smoking_experience' in insight:
        parts.append(f"흡연 경험 {insight['smoking_experience']}")
    
    return ", ".join(parts)

# =======================================================
# 3. 메인 인덱싱 로직
# =======================================================

def main():
    """
    PostgreSQL에서 데이터를 가져와 Qdrant에 인덱싱하는 메인 함수
    """
    pg_conn = None
    try:
        # --- 1. 서비스 연결 및 모델 로드 ---
        pg_conn = get_db_connection()
        qdrant_client = get_qdrant_client()
        
        if not pg_conn or not qdrant_client:
            print("데이터베이스 또는 Qdrant 연결 실패로 인덱싱을 중단합니다.")
            return

        print(f"⏳ 임베딩 모델 '{EMBEDDING_MODEL_NAME}' 로딩 중...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("✅ 임베딩 모델 로드 완료")

        # --- 2. Qdrant 컬렉션 준비 ---
        create_qdrant_collection(qdrant_client)

        # --- 3. PostgreSQL에서 데이터 가져오기 ---
        cursor = pg_conn.cursor()
        cursor.execute("SELECT uid, ai_insights FROM panels_master;")
        records = cursor.fetchall()
        cursor.close()
        print(f"✅ PostgreSQL에서 {len(records)}개의 레코드를 가져왔습니다.")

        # --- 4. 데이터 처리 및 Qdrant 포인트 생성 ---
        points_to_upsert = []
        texts_to_embed = []
        
        print("⏳ 임베딩을 위한 텍스트 데이터 준비 중...")
        for uid, ai_insights in records:
            # ai_insights를 자연어 텍스트로 변환
            text_for_embedding = convert_insight_to_text(ai_insights)
            texts_to_embed.append(text_for_embedding)
            
            # Qdrant PointStruct 준비 (id와 payload 설정)
            # 벡터는 나중에 일괄 생성 후 추가
            points_to_upsert.append(
                models.PointStruct(
                    id=uid,
                    payload={"uid": uid} # 검색 시 uid를 사용하기 위해 payload에 저장
                )
            )

        # --- 5. 텍스트 일괄 임베딩 ---
        print(f"⏳ {len(texts_to_embed)}개의 텍스트에 대한 임베딩 생성 중... (시간이 소요될 수 있습니다)")
        embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=True)

        # 생성된 임베딩을 PointStruct에 추가
        for i, embedding in enumerate(embeddings):
            points_to_upsert[i].vector = embedding.tolist()

        # --- 6. Qdrant에 데이터 업로드 (Upsert) ---
        print(f"⏳ Qdrant에 {len(points_to_upsert)}개의 포인트 업로드 중...")
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points_to_upsert,
            wait=True  # 작업이 완료될 때까지 대기
        )

        print("\n🎉 인덱싱 작업이 성공적으로 완료되었습니다!")
        print(f"총 {len(points_to_upsert)}개의 데이터가 '{QDRANT_COLLECTION_NAME}' 컬렉션에 저장되었습니다.")

    except Exception as e:
        print(f"\n❌ 인덱싱 중 오류 발생: {e}")
    finally:
        if pg_conn:
            pg_conn.close()
            print("🔌 PostgreSQL 연결 종료")

if __name__ == "__main__":
    main()