import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, SearchParams
from langchain_huggingface import HuggingFaceEmbeddings

@lru_cache(maxsize=None)
def initialize_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="nlpai-lab/KURE-v1", model_kwargs={'device': 'cpu'})
    except Exception as e:
        logging.error(f"임베딩 로드 실패: {e}")
        raise