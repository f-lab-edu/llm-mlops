import getpass

from app.vectorstore.opensearch_hybrid import OpenSearchHybridSearch
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# tool에 필요한 class 초기화
duckduckgo_search = DuckDuckGoSearchRun()

opensearch_password = getpass.getpass("Enter your Opensearch password: ")
opensearch = OpenSearchHybridSearch(user="admin", pw=opensearch_password)

@tool(parse_docstring=True)
def web_search(query: str):
    """A tool to use when websearch is needed. Use this tool when there isn't enough information in OpenSearch vector store Database

    Args:
        query : a query to websearch

    Returns:
        list[str]: websearch result in list of strings
    """
    res = duckduckgo_search.invoke({"query": query})
    return [res]


@tool(parse_docstring=True)
def vectorstore_search(query: str, search_type: str = "hybrid"):
    """search OpenSearch vector databsee for documents that are relevant to a given query.
    The vectorstore database contains blog posts about Machine Learning and LLM (Large Language Models) from Anthropic, Naver(네이버), NCSoft (엔씨소프트), and OpenAI.

    Args:
        query: a query to search vector database
        search_type (str, optional): one of three similarity search method: hybrid, BM25, and Cosine similarity search Defaults to "hybrid".

    Returns:
        list[pd.DataFrmae]: list of langchain document
    """
    if search_type == "hybrid":
        result = opensearch.hybrid_search(query)
    elif search_type == "bm25":
        result = opensearch.bm25_search(query)
    elif search_type == "cosine":
        result = opensearch.cosine_similarity_search(query)

    result["text"].tolist()

    return result


