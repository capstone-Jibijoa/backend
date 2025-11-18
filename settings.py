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
# import boto3
# import os
# from typing import Optional
# import anthropic

# # --- 1. Constants (AWS 설정 및 Secret Name) ---
# # Secrets Manager 접근에 필요한 비민감 정보
# # 이 값들은 docker-compose.yml 파일의 environment 섹션에서 주입됩니다.
# AWS_REGION = os.environ.get("DEFAULT_REGION", "ap-southeast-2") # 리전 환경 변수를 사용하거나 기본값 설정
# # Secrets Manager에 저장한 비밀 이름 (실제 저장한 이름으로 대체해야 합니다)
# DB_PASSWORD_SECRET_NAME = "DB_PASSWORD"  
# ANTHROPIC_KEY_SECRET_NAME = "ANTHROPIC_API_KEY"


# # --- 2. Secrets Manager 로드 함수 ---

# secrets_client = None

# def get_secrets_client():
#     """Secrets Manager 클라이언트 초기화 (IAM Role 사용)"""
#     global secrets_client
#     if secrets_client is None:
#         try:
#             # EC2 인스턴스에 연결된 IAM Role 권한을 사용
#             secrets_client = boto3.client('secretsmanager', region_name=AWS_REGION)
#         except Exception as e:
#             print(f"Boto3 Secrets Manager Client 초기화 실패: {e}")
#             raise e
#     return secrets_client


# def get_single_secret_value(secret_name: str) -> str:
#     """Secrets Manager에서 단일 비밀 값을 가져와 문자열로 반환"""
#     client = get_secrets_client()
    
#     if not secret_name:
#         raise ValueError(f"비밀 이름 ({secret_name})이 설정되지 않았습니다.")

#     try:
#         response = client.get_secret_value(SecretId=secret_name)
        
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
#         if 'SecretString' in response:
#             # SecretString이 JSON이 아닌 일반 문자열이라고 가정 (DB 비밀번호, API Key)
#             return response['SecretString']
#         else:
#             raise TypeError(f"Secret {secret_name}의 값이 SecretString 형태가 아닙니다.")

#     except Exception as e:
#         print(f"Secrets Manager 호출 실패 (Secret Name: {secret_name}). IAM 권한 및 ARN 확인 필요.")
#         raise RuntimeError(f"애플리케이션 시작에 필요한 비밀 정보를 로드하지 못했습니다: {e}")


# # --- 3. 설정 클래스 정의 및 값 로드 ---

# class Settings:
#     # 1. 일반 환경 변수 (docker-compose.yml에서 주입)
#     DB_HOST: str = os.environ.get("DB_HOST", "localhost")
#     DB_NAME: str = os.environ.get("DB_NAME", "default_db")
#     DB_USER: str = os.environ.get("DB_USER", "postgres")
#     PORT: int = int(os.environ.get("PORT", 8000)) # 문자열을 정수로 변환

#     QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "localhost")
#     QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", 6333)) # 문자열을 정수로 변환
#     QDRANT_COLLECTION_WELCOME_NAME: str = os.environ.get("QDRANT_COLLECTION_WELCOME_NAME", "welcome")
#     QDRANT_COLLECTION_QPOLL_NAME: str = os.environ.get("QDRANT_COLLECTION_QPOLL_NAME", "qpoll")

#     # 2. 비밀 변수 (초기값 None, Secrets Manager에서 로드 예정)
#     DB_PASSWORD: Optional[str] = None
#     ANTHROPIC_API_KEY: Optional[str] = None
    
    
#     def load_secrets(self):
#         """Secrets Manager에서 민감한 비밀 정보를 로드하여 클래스 속성에 할당"""
        
#         # DB 비밀번호 로드
#         self.DB_PASSWORD = get_single_secret_value(DB_PASSWORD_SECRET_NAME)
        
#         # Anthropic API 키 로드
#         self.ANTHROPIC_API_KEY = get_single_secret_value(ANTHROPIC_KEY_SECRET_NAME)


# # --- 4. 전역 인스턴스 생성 및 로드 (Import 시 1회 실행) ---

# settings = Settings()
# try:
#     # 애플리케이션 시작 시 Boto3를 호출하여 비밀 정보를 로드합니다.
#     settings.load_secrets() 
# except RuntimeError as e:
#     # 비밀 로드 실패 시 애플리케이션 시작을 막습니다.
#     print("FATAL: 핵심 비밀 정보 로드 실패. 인프라 설정(IAM/Secrets Manager)을 확인하세요.")
#     raise # 예외 발생

# # 이제 다른 파일에서는 settings.DB_PASSWORD, settings.DB_HOST 등으로 접근할 수 있습니다.
