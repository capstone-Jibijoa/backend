import boto3
import os
from typing import Optional
import anthropic

AWS_REGION = os.environ.get("DEFAULT_REGION", "ap-southeast-2")
DB_PASSWORD_SECRET_NAME = "DB_PASSWORD"  
ANTHROPIC_KEY_SECRET_NAME = "ANTHROPIC_API_KEY"


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


def get_single_secret_value(secret_name: str) -> str:
    """Secrets Manager에서 단일 비밀 값을 가져와 문자열로 반환"""
    client = get_secrets_client()
    
    if not secret_name:
        raise ValueError(f"비밀 이름 ({secret_name})이 설정되지 않았습니다.")

    try:
        response = client.get_secret_value(SecretId=secret_name)
        
        if 'SecretString' in response:
            return response['SecretString']
        else:
            raise TypeError(f"Secret {secret_name}의 값이 SecretString 형태가 아닙니다.")

    except Exception as e:
        print(f"Secrets Manager 호출 실패 (Secret Name: {secret_name}). IAM 권한 및 ARN 확인 필요.")
        raise RuntimeError(f"애플리케이션 시작에 필요한 비밀 정보를 로드하지 못했습니다: {e}")


class Settings:
    # 환경 변수에서 로드
    DB_HOST: str = os.environ.get("DB_HOST", "localhost")
    DB_NAME: str = os.environ.get("DB_NAME", "default_db")
    DB_USER: str = os.environ.get("DB_USER", "postgres")
    PORT: int = int(os.environ.get("PORT", 8000))

    QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_WELCOME_NAME: str = os.environ.get("QDRANT_COLLECTION_WELCOME_NAME", "welcome")
    QDRANT_COLLECTION_QPOLL_NAME: str = os.environ.get("QDRANT_COLLECTION_QPOLL_NAME", "qpoll")

    # Secrets Manager에서 로드
    DB_PASSWORD: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    
    def load_secrets(self):
        """Secrets Manager에서 민감한 비밀 정보를 로드하여 클래스 속성에 할당"""
        self.DB_PASSWORD = get_single_secret_value(DB_PASSWORD_SECRET_NAME)
        self.ANTHROPIC_API_KEY = get_single_secret_value(ANTHROPIC_KEY_SECRET_NAME)


settings = Settings()
try:
    settings.load_secrets() 
except RuntimeError as e:
    print("FATAL: 핵심 비밀 정보 로드 실패. 인프라 설정(IAM/Secrets Manager)을 확인하세요.")
    raise