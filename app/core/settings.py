'''
import boto3
import os
import json
from typing import Optional
from functools import lru_cache

# -- 1. Constants (AWS 설정 및 Secret Name) ---
AWS_REGION = os.environ.get("DEFAULT_REGION", "ap-southeast-2")
# 하나의 Secret에 모든 값이 JSON 형태로 저장되어 있음
SECRET_NAME = os.environ.get("APP_SECRET_CONFIG_NAME", "prod/backend/secrets")

# -- 2. Secrets Manager 로드 함수 ---

secrets_client = None

def get_secrets_client():
    """Secrets Manager 클라이언트 초기화 (IAM Role 사용)"""
    global secrets_client
    if secrets_client is None:
        try:
            secrets_client = boto3.client('secretsmanager', region_name=AWS_REGION)
        except Exception as e:
            print(f"Boto3 Secrets Manager Client 초기화 실패: {e}")
            raise e
    return secrets_client

@lru_cache(maxsize=1)
def get_secrets_from_manager() -> dict:
    """Secrets Manager에서 JSON 형태의 비밀 값을 가져와 딕셔너리로 반환"""
    client = get_secrets_client()
    
    if not SECRET_NAME:
        raise ValueError(f"비밀 이름이 설정되지 않았습니다.")

    try:
        response = client.get_secret_value(SecretId=SECRET_NAME)
        
        if 'SecretString' in response:
            # JSON 형태의 SecretString을 파싱
            return json.loads(response['SecretString'])
        else:
            raise TypeError(f"Secret {SECRET_NAME}의 값이 SecretString 형태가 아닙니다.")

    except Exception as e:
        print(f"Secrets Manager 호출 실패 (Secret Name: {SECRET_NAME}). IAM 권한 및 ARN 확인 필요.")
        print(f"에러 상세: {e}")
        raise RuntimeError(f"애플리케이션 시작에 필요한 비밀 정보를 로드하지 못했습니다: {e}")


# -- 3. 설정 클래스 정의 및 값 로드 ---

class Settings:
    # 1. 일반 환경 변수 (docker-compose.yml에서 주입)
    DB_HOST: str = os.environ.get("DB_HOST", "localhost")
    DB_NAME: str = os.environ.get("DB_NAME", "default_db")
    DB_USER: str = os.environ.get("DB_USER", "postgres")
    PORT: int = int(os.environ.get("PORT", 5432))

    QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_WELCOME_NAME: str = os.environ.get("QDRANT_COLLECTION_WELCOME_NAME", "welcome")
    QDRANT_COLLECTION_QPOLL_NAME: str = os.environ.get("QDRANT_COLLECTION_QPOLL_NAME", "qpoll")

    # 2. 비밀 변수 (초기값 None, Secrets Manager에서 로드 예정)
    DB_PASSWORD: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    
    def load_secrets(self):
        """Secrets Manager에서 민감한 비밀 정보를 로드하여 클래스 속성에 할당"""
        
        # 환경변수에 이미 값이 있으면 Secrets Manager 호출 건너뛰기 (로컬 개발용)
        if os.environ.get("DB_PASSWORD"):
            print("환경변수에서 DB_PASSWORD 로드")
            self.DB_PASSWORD = os.environ.get("DB_PASSWORD")
            self.ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
            return
        
        # Secrets Manager에서 JSON 형태로 모든 비밀 로드
        try:
            secrets = get_secrets_from_manager()
            
            # JSON에서 개별 값 추출
            self.DB_PASSWORD = secrets.get("DB_PASSWORD")
            self.ANTHROPIC_API_KEY = secrets.get("ANTHROPIC_API_KEY")
            
            # 필수 값 검증
            if not self.DB_PASSWORD:
                raise ValueError("DB_PASSWORD가 Secrets Manager에 없습니다.")
            if not self.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY가 Secrets Manager에 없습니다.")
                
            print("Secrets Manager에서 비밀 정보 로드 성공")
            
        except Exception as e:
            print(f"Secrets Manager 로드 실패: {e}")
            raise


# -- 4. 전역 인스턴스 생성 및 로드 (Import 시 1회 실행) ---

settings = Settings()
try:
    # 애플리케이션 시작 시 Boto3를 호출하여 비밀 정보를 로드합니다.
    settings.load_secrets() 
except RuntimeError as e:
    # 비밀 로드 실패 시 애플리케이션 시작을 막습니다.
    print("FATAL: 핵심 비밀 정보 로드 실패. 인프라 설정(IAM/Secrets Manager)을 확인하세요.")
    raise
    '''