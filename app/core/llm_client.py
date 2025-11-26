import logging
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
#from app.core.settings import settings 

load_dotenv()

def get_claude_client():
    """Claude 클라이언트 싱글톤 또는 팩토리"""
    try:
        client = ChatAnthropic(
            model="claude-sonnet-4-5",
            temperature=0.1,
            # api_key=settings.ANTHROPIC_API_KEY 
        )
        return client
    except Exception as e:
        logging.error(f"Anthropic 클라이언트 생성 실패: {e}")
        return None

# 전역 변수로 초기화 (필요시 사용)
CLAUDE_CLIENT = get_claude_client()