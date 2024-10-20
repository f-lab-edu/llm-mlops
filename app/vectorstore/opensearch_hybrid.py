import os
import sys
from typing import List
from langchain.docstore.document import Document

from dotenv import load_dotenv, find_dotenv

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from opensearchpy import OpenSearch, RequestsHttpConnection

import pandas as pd

load_dotenv(override=True)

OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
OPENSEARCH_PORT = os.getenv("OPENSEARCH_PORT")
OPENSEARCH_PW = os.getenv("OPENSEARCH_INITIAL_ADMIN_PASSWORD")
OPENSEARCH_BLOG_DATA_INDEX = os.getenv("OPENSEARCH_BLOG_DATA_INDEX")
HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL")

class OpenSearchHybridSearch:
    def __init__(
        self,
        host: str = OPENSEARCH_HOST,
        port: str = OPENSEARCH_PORT,
        user: str = OPENSEARCH_USER,
        pw: str = OPENSEARCH_PW,
    ):
        """초기화 메서드
        OpenSearchHybridSearch 클래스의 인스턴스를 초기화합니다.

        Args:
            host (str): OpenSearch 호스트 주소.
            port (str): OpenSearch 포트 번호.
            user (str): OpenSearch 사용자 이름.
            pw (str): OpenSearch 비밀번호.
        """
        # HuggingFace를 사용하여 임베딩 모델 설정
        self.embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)

        # OpenSearch HTTP 인증 정보 설정
        self.opensearch_http_auth = (user, pw)
        # OpenSearch 클라이언트 초기화
        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=self.opensearch_http_auth,
            use_ssl=True,
            verify_certs=False,
            connection_class=RequestsHttpConnection,
        )
        # OpenSearch URL 구성
        self.openserach_url = f"https://{host}:{port}"
        # OpenSearch 벡터 스토어 초기화
        self.vector_store = OpenSearchVectorSearch(
            index_name=OPENSEARCH_BLOG_DATA_INDEX,
            embedding_function=self.embeddings,
            opensearch_url=self.openserach_url,
            http_auth=self.opensearch_http_auth,
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
        )

    def _reciprocal_rank_fusion(
        self, result_df_list: List[pd.DataFrame], k: int = 60
    ) -> pd.DataFrame:
        """Reciprocal Rank Fusion 알고리즘을 사용하여 결과를 결합합니다.

        Args:
            result_df_list (List[pd.DataFrame]): 검색 결과의 데이터프레임 리스트.
            k (int): 순위별 가중치 계산을 위한 상수.

        Returns:
            pd.DataFrame: 결합된 결과의 데이터프레임.
        """
        # 결과를 하나의 데이터프레임으로 결합 및 중복 제거
        df_rrf_result = (
            pd.concat(result_df_list, axis=0, join="outer")
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # RRF 점수 계산 함수 정의
        def rrf_score(row):
            # BM25 순위 관련 RRF 점수 계산
            rrf_bm25 = (
                1 / (k + row["BM25_rank"]) if not pd.isna(row["BM25_rank"]) else 0
            )
            # 코사인 순위 관련 통한 RRF 점수 계산
            rrf_cosine = (
                1 / (k + row["cosine_rank"]) if not pd.isna(row["cosine_rank"]) else 0
            )
            return sum([rrf_bm25, rrf_cosine])

        # 각 문서에 대해 RRF 점수를 계산 후, 점수에 따라 정렬
        df_rrf_result["rrf_score"] = df_rrf_result.apply(rrf_score, axis=1)

        df_rrf_result = df_rrf_result.sort_values(
            "rrf_score", ascending=False
        ).reset_index(drop=True)

        return df_rrf_result

    def parse_search_result(self, result: dict, similarity_type: str) -> pd.DataFrame:
        """검색 결과를 파싱하여 데이터프레임으로 반환합니다.

        Args:
            result (dict): 검색 결과 데이터.
            similarity_type (str): 유사성 평가 방법 (예: BM25, cosine).

        Returns:
            pd.DataFrame: 파싱된 결과의 데이터프레임.
        """
        parsed_result = list()
        # 검색 결과에서 각 문서의 정보 추출
        for indiv_doc_info in result["hits"]["hits"]:
            url = indiv_doc_info["_source"]["metadata"]["source"]
            text = indiv_doc_info["_source"]["text"]
            title = indiv_doc_info["_source"]["metadata"]["title"]
            score = indiv_doc_info["_score"]
            parsed_result.append((url, text, title, score))
        # 파싱된 결과를 데이터프레임으로 변환 및 랭킹 계산
        df_parsed = pd.DataFrame(
            parsed_result, columns=["url", "text", "title", f"{similarity_type}_score"]
        )
        df_parsed[f"{similarity_type}_rank"] = df_parsed[
            f"{similarity_type}_score"
        ].rank(ascending=False)
        return df_parsed

    def insert_docs(self, doc_list: List[Document]) -> None:
        """문서를 인덱스에 삽입합니다.

        Args:
            doc_list (List[Document]): 삽입할 문서 목록.
        """
        # 문서를 벡터 스토어에 삽입
        self.vector_store.add_documents(doc_list)

    def hybrid_search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """하이브리드 검색을 수행합니다.

        Args:
            query (str): 검색할 쿼리 문자열.
            top_k (int): 검색 결과에서 상위 문서 수.

        Returns:
            pd.DataFrame: 최종 하이브리드 검색 결과의 데이터프레임.
        """
        # 쿼리 임베딩 생성
        query_embedding = list(self.embeddings.embed_query(query))

        # syntactic search 쿼리 구성
        syntactic_search_query = {
            "size": top_k * 2,
            "query": {"match": {"text_entry": query}},
        }
        # semantic search 검색 쿼리 구성
        semantic_search_query = {
            "size": top_k * 2,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "knn_score",
                        "lang": "knn",
                        "params": {
                            "field": "vector_field",
                            "query_value": query_embedding,
                            "space_type": "cosinesimil",
                        },
                    },
                }
            },
        }
        # OpenSearch를 사용한 syntactic search 수행
        syntactic_search_result = self.client.search(
            body=syntactic_search_query, index=OPENSEARCH_BLOG_DATA_INDEX
        )

        # OpenSearch를 사용한 semantic search 검색 수행
        semantic_search_result = self.client.search(
            body=semantic_search_query, index=OPENSEARCH_BLOG_DATA_INDEX
        )

        # 검색 결과 파싱
        df_syntactic = self.parse_search_result(
            syntactic_search_result, similarity_type="BM25"
        )
        df_semantic = self.parse_search_result(
            semantic_search_result, similarity_type="cosine"
        )
        total_search_result_list = [df_syntactic, df_semantic]

        # RRF를 사용하여 최종 결과 결합
        df_rrf = self._reciprocal_rank_fusion(total_search_result_list)
        df_rrf = df_rrf.iloc[:top_k, :]

        return df_rrf

# def vectorstore_hybrid_search():


if __name__ == "__main__":

    vector_store_obj = OpenSearchHybridSearch()
    # q = "What is anthropic working on?"
    q = "ncsoft에서 최근에 어떤 기술들을 연구하고 있어?"
    res = vector_store_obj.hybrid_search(q)

    print(res)
