from pydantic import BaseModel

class SearchQuery(BaseModel):
    """
    검색 요청 모델
    """
    query: str
    search_mode: str = "all"