import logging
from functools import partial
from typing import Literal

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from state import AgentState

load_dotenv()
# Bentoml 서버 로깅 설정
bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.setLevel(logging.INFO)


# 모델 output 형식 class
class is_related(BaseModel):
    """모델 판단 결과 형식 정하는 class"""

    is_related: str = Field(
        description="one of 'yes' or 'no' string value depending on the query relevance to Machine Learning, Deep Learning or Data Science"
    )


def check_relevance(
    state: AgentState, model: ChatHuggingFace
) -> Literal["format_question", END]:
    """질문(query)가 데이터 사이언스, ML/DL 토픽과 관련이 있는지 여부를 판단합니다.

    Args:
        state: 그래프의 현재 상태가 담긴 AgenticState

    Returns:
        str: User가 입력한 query가 데이터 사이언스, ML/DL과 관련이 있는지 여부에 대한 결정
            - "format_question": 관련 있으면 다음 node 이름 출력
            - END: 관련 없으면 그래프 종료 (END 출력)
    """
    bentoml_logger.info("====check_relevance====")
    query = state["query"]
    model_with_tool = model.with_structured_output(is_related)

    messages = [
        SystemMessage(
            content="""Please read the query from the user and determine whether the query is related to any of the following topics: \n 
            - Machine Learning
            - Deep Learning
            - Data Science

        If the query contains keyword(s) or semantic meaning related to the given topics, grade it as relevant. \n
        Respond in JSON with `is_related` key.
        Only return a binary score 'yes' or 'no' without further explanation."""
        ),
        HumanMessage(content=f"Here is the query: {query}"),
    ]

    score = model_with_tool.invoke(messages)
    score = score.is_related
    if score == "yes":
        return "format_question"
    elif score == "no":
        return END


def format_question(state: AgentState):
    """Graph Placeholder: 추후 로직 추가 예정"""
    return {"query": state["query"]}


def build_graph(model: ChatHuggingFace) -> CompiledStateGraph:
    """최종 graph build하는 함수

    Returns:
        CompiledStateGraph: custom build한 nodes과 edges포함한 graph
    """
    workflow = StateGraph(AgentState)
    workflow.add_conditional_edges(
        START, partial(check_relevance, model=model)
    )  # partial를 사용하여 bemtoml LlmService에서 생성된 모델을 node안에서 사용할 수 있도록 합니다.: https://github.com/langchain-ai/langgraph/discussions/341#discussioncomment-11148281 참조
    workflow.add_node("format_question", format_question)
    workflow.add_edge("format_question", END)
    graph = workflow.compile()

    # 생서된 graph가 잘 연결되어 있는지 확인하기 위해 graph 시각화 후 저장
    png_data = graph.get_graph(xray=True).draw_mermaid_png()

    # PNG파일 저장
    with open("langgraph_diagram.png", "wb") as file:
        file.write(png_data)

    print("LangGraph diagram saved as langgraph_diagram.png")
    return graph
