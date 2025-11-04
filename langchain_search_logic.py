# langchain_search_logic.py

import os
from operator import itemgetter
from dotenv import load_dotenv

# LangChain 및 관련 라이브러리 임포트
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

# Qdrant 필터 관련 임포트 추가
from qdrant_client.models import Filter, FieldCondition, MatchAny

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
            embeddings=embeddings,
            content_payload_key="text",
        )
        print(f"✅ LangChain: Qdrant 벡터 저장소 ('{collection_name}') 준비 완료")
        
        return vector_store, embeddings

    except Exception as e:
        print(f"❌ LangChain 구성 요소 초기화 실패: {e}")
        return None, None

# 전역 변수
VECTOR_STORE = None
EMBEDDINGS = None

def _initialize_langchain_components():
    """
    [새로 만들거나 수정]
    전역 변수를 실제로 초기화하고 설정하는 내부 함수.
    """
    global VECTOR_STORE, EMBEDDINGS, _chain_cache
    
    print("🔄 LangChain 구성 요소 초기화를 시작합니다...")
    VECTOR_STORE, EMBEDDINGS = initialize_components()
    
    if not VECTOR_STORE:
        _chain_cache = None # 실패 시 캐시 비움
        raise RuntimeError("벡터 저장소가 초기화되지 않았습니다.")
    
    _chain_cache = create_langchain_hybrid_retriever_chain()
    print("✅ LangChain 체인 캐시 생성 완료.")

def get_langchain_hybrid_chain():
    """
    체인 객체를 반환합니다. 필요할 때만 초기화를 호출합니다.
    """
    global _chain_cache
    if _chain_cache is None:
        _initialize_langchain_components() # 캐시가 없으면 초기화
    return _chain_cache

def force_reload_langchain_components():
    """
    [핵심 로직]
    서버 재시작 없이 LangChain 구성 요소를 강제로 다시 로드하는 함수.
    """
    print("🔥 LangChain 구성 요소 강제 리로드를 요청받았습니다.")
    
    # 3. '엔진 교체' 작업을 강제로 실행합니다.
    _initialize_langchain_components() 
    
    return {"status": "success", "message": "LangChain components reloaded successfully."}

# =======================================================
# 2. LangChain 체인(Chain)의 각 단계를 구성하는 함수
# =======================================================

def _get_filtered_uids_from_postgres(structured_condition: str) -> list[int]:
    """[체인 1단계] 정형 조건을 사용하여 PostgreSQL에서 UID 목록을 필터링합니다."""
    pg_conn = None
    try:
        pg_conn = get_db_connection()
        if not pg_conn: 
            return []
        
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
        if pg_conn: 
            pg_conn.close()

def _search_qdrant_or_pass_through(x: dict) -> list[Document]:
    """
    시맨틱 검색어(x["question"])가 있으면 Qdrant 검색을 수행하고,
    없으면 PostgreSQL에서 받은 UID를 metadata로 하는
    가상 Document 리스트를 생성하여 3단계로 바로 전달합니다.
    """
    semantic_query = x.get("question", "").strip()
    uids = x.get("uids", [])

    if not uids:
        # 1단계(PostgreSQL)에서 아무것도 못 찾았으면 즉시 종료
        print("LANGCHAIN_CHAIN: 필터링된 UID가 없어 2단계를 건너뜁니다.")
        return []

    # --- 1. 시맨틱 검색어가 *있는* 경우 (기존 Qdrant 검색 로직) ---
    if semantic_query:
        print(f"LANGCHAIN_CHAIN: 시맨틱 검색어 '{semantic_query}'로 Qdrant 검색 수행...")
        global VECTOR_STORE
        try:
            qdrant_filter = Filter(
                must=[ FieldCondition(key="uid", match=MatchAny(any=uids)) ]
            )
            k_to_search = max(150, len(uids))
            
            print(f"🔍 DEBUG: Qdrant k={k_to_search}로 검색 (필터된 UID 개수: {len(uids)})")
            print(f"🔍 DEBUG: Qdrant 검색 질문: {semantic_query}")
            if len(uids) < 10:
                 print(f"🔍 DEBUG: Qdrant 필터 UID 목록: {uids}")

            results = VECTOR_STORE.similarity_search(
                query=semantic_query,
                k=k_to_search,
                filter=qdrant_filter
            )
            print(f"LANGCHAIN_CHAIN: Qdrant 검색 결과 {len(results)}개 발견.")
            return results
            
        except Exception as e:
            print(f"LANGCHAIN_CHAIN: Qdrant 검색 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    # --- 2. 시맨틱 검색어가 *없는* 경우 ---
    else:
        print("LANGCHAIN_CHAIN: 시맨틱 검색어가 없습니다. Qdrant 검색(2단계)을 건너뜁니다.")
        print("LANGCHAIN_CHAIN: PostgreSQL UID(1단계)를 3단계(최종 조회)로 바로 전달합니다.")
        
        # 3단계(_get_final_data_from_postgres)는 Document 리스트를 기대합니다.
        # 따라서, 1단계에서 받은 UID 목록을 가상 Document 형식으로 변환하여 넘겨줍니다.
        
        virtual_documents = []
        for uid in uids:
            # metadata에 uid만 포함된 가상 Document 생성
            virtual_documents.append(
                Document(page_content="", metadata={"uid": uid})
            )
            
        # 3단계 함수는 이 Document 리스트를 받아 metadata.uid를 추출하여
        # 최종 DB 조회를 실행할 것입니다. 
        return virtual_documents

def _get_final_data_from_postgres(documents: list[Document]) -> list[dict]:
    """[체인 3단계] Qdrant 검색 결과(Document)에서 UID를 추출하여 최종 데이터를 조회합니다."""
    print(f"🔍 DEBUG: 받은 documents 개수: {len(documents)}")
    if not documents:
        return []
    
    # ================= [ 🐞 디버깅 로그 추가 ] =================
    try:
        print(f"🔍 DEBUG: [첫 번째 문서] metadata 타입: {type(documents[0].metadata)}")
        print(f"🔍 DEBUG: [첫 번째 문서] metadata 내용: {documents[0].metadata}")
    except Exception as e:
        print(f"🔍 DEBUG: 첫 번째 문서 접근 오류: {e}")
    # =======================================================
    
    # 이제 metadata는 항상 dict 타입이므로 복잡한 분기 처리가 필요 없습니다.
    # 예: documents[0].metadata -> {'uid': 338370929131356}
    final_uids = [doc.metadata.get('uid') for doc in documents if doc.metadata and doc.metadata.get('uid') is not None]

    print(f"LANGCHAIN_CHAIN: Qdrant 결과에서 {len(final_uids)}개의 유효한 UID 추출 완료.")

    if not final_uids:
        return []

    pg_conn = None
    try:
        pg_conn = get_db_connection()
        if not pg_conn: 
            return []

        cur = pg_conn.cursor()
        final_query = "SELECT ai_insights FROM panels_master WHERE uid IN %s"
        
        # 중복 제거하여 쿼리 효율성 향상
        unique_uids = tuple(set(final_uids))
        cur.execute(final_query, (unique_uids,))
        
        final_results = [row[0] for row in cur.fetchall()]
        cur.close()
        print(f"LANGCHAIN_CHAIN: 최종 데이터 {len(final_results)}개 조회 완료.")
        return final_results
    except Exception as e:
        print(f"LANGCHAIN_CHAIN: 최종 데이터 조회 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        if pg_conn: 
            pg_conn.close()

# =======================================================
# 3. 하이브리드 검색을 위한 LangChain 체인 정의
# =======================================================

def create_langchain_hybrid_retriever_chain():
    """하이브리드 검색 로직을 수행하는 LangChain 체인을 생성합니다."""
    global VECTOR_STORE, EMBEDDINGS
    
    # VECTOR_STORE가 아직 초기화되지 않았다면 초기화
    if VECTOR_STORE is None:
        VECTOR_STORE, EMBEDDINGS = initialize_components()

    if not VECTOR_STORE:
        raise RuntimeError("벡터 저장소가 초기화되지 않았습니다. 서버를 재시작하세요.")

    # LangChain Expression Language (LCEL)로 체인 구성
    chain = (
        {
            # 'structured' 키로 들어온 입력을 _get_filtered_uids_from_postgres 함수로 전달
            "uids": itemgetter("structured") | RunnableLambda(_get_filtered_uids_from_postgres),
            # 'semantic' 키로 들어온 입력을 'question'이라는 키로 그대로 통과
            "question": itemgetter("semantic")
        }
        | RunnableLambda(_search_qdrant_or_pass_through) 
        | RunnableLambda(_get_final_data_from_postgres)  # 3단계 함수는 그대로 유지
    )
    return chain

# 체인 캐시
_chain_cache = None

def get_langchain_hybrid_chain():
    """체인 객체를 반환합니다. 필요할 때 초기화합니다."""
    global _chain_cache
    if _chain_cache is None:
        _chain_cache = create_langchain_hybrid_retriever_chain()
    return _chain_cache