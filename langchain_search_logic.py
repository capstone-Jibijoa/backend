# langchain_search_logic.py

import os
from operator import itemgetter
from dotenv import load_dotenv

# LangChain 및 관련 라이브러리 임포트
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

# 기존 모듈에서 필요한 함수 및 클라이언트 임포트
from db_logic import get_db_connection, _build_jsonb_where_clause, get_qdrant_client

load_dotenv()

# =======================================================
# 1. LangChain 구성 요소 초기화 (모듈 로드 시 1회 실행)
# =======================================================

def initialize_components():
    """LangChain에서 사용할 임베딩 모델과 벡터 저장소를 초기화합니다."""
    try:
        # 1. 임베딩 모델 초기화 (KURE 모델)
        # HuggingFaceEmbeddings 래퍼를 사용하여 LangChain과 호환되도록 만듭니다.
        print("⏳ LangChain: KURE 임베딩 모델 로딩 중...")
        embeddings = HuggingFaceEmbeddings(
            model_name="nlpai-lab/KURE-v1",
            model_kwargs={'device': 'cpu'} # 또는 'cuda'
        )
        print("✅ LangChain: KURE 임베딩 모델 로드 완료")

        # 2. Qdrant 벡터 저장소 초기화
        qdrant_client = get_qdrant_client()
        if not qdrant_client:
            raise ConnectionError("Qdrant 클라이언트 연결에 실패했습니다.")
        
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "panels_collection")
        
        # Qdrant를 LangChain의 VectorStore 인터페이스로 래핑합니다.
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=embeddings
        )
        print(f"✅ LangChain: Qdrant 벡터 저장소 ('{collection_name}') 준비 완료")
        
        return vector_store, embeddings

    except Exception as e:
        print(f"❌ LangChain 구성 요소 초기화 실패: {e}")
        return None, None

VECTOR_STORE, EMBEDDINGS = initialize_components()

# =======================================================
# 2. LangChain 체인(Chain)의 각 단계를 구성하는 함수
# =======================================================

def _get_filtered_uids_from_postgres(structured_condition: str) -> list[int]:
    """[체인 1단계] 정형 조건을 사용하여 PostgreSQL에서 UID 목록을 필터링합니다."""
    pg_conn = None
    try:
        pg_conn = get_db_connection()
        if not pg_conn: return []
        
        cur = pg_conn.cursor()
        where_clause, where_params = _build_jsonb_where_clause(structured_condition)
        
        pg_query = f"SELECT uid FROM panels_master {where_clause}"
        cur.execute(pg_query, tuple(where_params))
        
        filtered_uids = [row[0] for row in cur.fetchall()]
        cur.close()
        print(f"LANGCHAIN_CHAIN: PostgreSQL 필터링 결과 {len(filtered_uids)}개의 UID 발견.")
        return filtered_uids
    except Exception as e:
        print(f"LANGCHAIN_CHAIN: PostgreSQL UID 필터링 중 오류: {e}")
        return []
    finally:
        if pg_conn: pg_conn.close()

def _get_final_data_from_postgres(documents: list[Document]) -> list[dict]:
    """[체인 3단계] Qdrant 검색 결과(Document)에서 UID를 추출하여 최종 데이터를 조회합니다."""
    if not documents:
        return []
    
    # Document의 metadata에서 uid를 추출합니다.
    final_uids = [doc.metadata['uid'] for doc in documents if 'uid' in doc.metadata]
    if not final_uids:
        return []

    pg_conn = None
    try:
        pg_conn = get_db_connection()
        if not pg_conn: return []

        cur = pg_conn.cursor()
        final_query = "SELECT ai_insights FROM panels_master WHERE uid IN %s"
        cur.execute(final_query, (tuple(final_uids),))
        
        final_results = [row[0] for row in cur.fetchall()]
        cur.close()
        print(f"LANGCHAIN_CHAIN: 최종 데이터 {len(final_results)}개 조회 완료.")
        return final_results
    except Exception as e:
        print(f"LANGCHAIN_CHAIN: 최종 데이터 조회 중 오류: {e}")
        return []
    finally:
        if pg_conn: pg_conn.close()

# =======================================================
# 3. 하이브리드 검색을 위한 LangChain 체인 정의
# =======================================================

def create_langchain_hybrid_retriever_chain():
    """하이브리드 검색 로직을 수행하는 LangChain 체인을 생성합니다."""
    if not VECTOR_STORE:
        raise RuntimeError("벡터 저장소가 초기화되지 않았습니다. 서버를 재시작하세요.")

    # Qdrant VectorStore를 Retriever로 변환합니다.
    # search_kwargs를 통해 검색 시 동적으로 필터를 적용할 수 있습니다.
    retriever = VECTOR_STORE.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 150} # top_k 설정
    )

    # LangChain Expression Language (LCEL)로 체인 구성 (입력 구조 개선)
    chain = (
        {
            # 'structured' 키로 들어온 입력을 _get_filtered_uids_from_postgres 함수로 전달
            "uids": itemgetter("structured") | RunnableLambda(_get_filtered_uids_from_postgres),
            # 'semantic' 키로 들어온 입력을 'question'이라는 키로 그대로 통과
            "question": itemgetter("semantic")
        }
        | RunnableLambda(
            # 2. Qdrant 검색. search_kwargs를 동적으로 수정하여 필터 적용
            lambda x: retriever.get_relevant_documents(
                x["question"],
                filter={"must": [{"key": "uid", "match": {"any": x["uids"]}}]}
            ) if x["uids"] else []
        )
        | RunnableLambda(_get_final_data_from_postgres) # 3. 최종 데이터 조회
    )
    return chain

# 메인 함수에서 사용할 체인 객체
langchain_hybrid_chain = create_langchain_hybrid_retriever_chain()