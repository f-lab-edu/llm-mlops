import logging
from functools import partial
from typing import Literal

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition


from langchain_core.language_models.chat_models import BaseChatModel

from pydantic import BaseModel, Field

from state import AgentState
from tools import vectorstore_search, web_search

load_dotenv()
# Bentoml 서버 로깅 설정
bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.setLevel(logging.INFO)

# Agent가 사용할 tools 초기화
toolset = [vectorstore_search, web_search]


# 모델 output 형식 class
class is_related(BaseModel):
    """모델 판단 결과 형식 정하는 class"""

    is_related: str = Field(
        description="one of 'yes' or 'no' string value depending on the query relevance to Machine Learning, Deep Learning or Data Science"
    )


def check_relevance(state: AgentState, model: BaseChatModel) -> Literal["agent", END]:
    """질문(query)가 데이터 사이언스, ML/DL 토픽과 관련이 있는지 여부를 판단합니다.

    Args:
        state(AgentState): 그래프의 현재 상태가 담긴 AgentState
        model (BaseChatModel): bentoml API에서 받아오는 쿼리를 처리할 모델

    Returns:
        str: User가 입력한 query가 데이터 사이언스, ML/DL과 관련이 있는지 여부에 대한 결정
            - "agent": 관련 있으면 다음 node 이름 출력
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
        return "agent"
    elif score == "no":
        return END


def agent(state: AgentState, model: BaseChatModel) -> dict:
    """
    에이전트 상태에서 user query를 불러와 모델을 사용하여 쿼리를 처리하고 응답을 반환합니다

    Args:
        state (AgentState): 그래프의 현재 상태가 담긴 AgenticState: "query" 에 user 질문을 포함
        model (BaseChatModel): bentoml API에서 받아오는 쿼리를 처리할 모델

    Returns:
        return_response (dict): 처리된 쿼리에 대해 어떤 tool을 사용할지에 대한 정보가 담긴 응답 메시지를 포함하는 dictionary
    """

    query = state["query"]
    model_with_tools = model.bind_tools(toolset)
    response = model_with_tools.invoke(query)

    return_response = {"messages": response}

    return return_response


def build_graph(model: BaseChatModel) -> CompiledStateGraph:
    """최종 graph build하는 함수

    Args:
        model (BaseChatModel): bentoml API에서 받아오는 쿼리를 처리할 모델입니다

    Returns:
        CompiledStateGraph: custom build한 nodes과 edges포함한 graph
    """
    workflow = StateGraph(AgentState)

    # Nodes 추가
    workflow.add_node(
        "agent", partial(agent, model=model)
    )  # partial를 사용하여 bemtoml LlmService에서 생성된 모델을 node안에서 사용할 수 있도록 합니다.: https://github.com/langchain-ai/langgraph/discussions/341#discussioncomment-11148281 참조
    retrieve = ToolNode(toolset) # ToolNode를 사용하여 tools를 사용할 수 있도록 합니다.
    workflow.add_node("retrieve", retrieve)

    # Edges/conditional edges 추가
    workflow.add_conditional_edges(START, partial(check_relevance, model=model))
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_edge("retrieve", END)
    graph = workflow.compile()

    # 생서된 graph가 잘 연결되어 있는지 확인하기 위해 graph 시각화 후 저장
    png_data = graph.get_graph().draw_mermaid_png()

    # PNG파일 저장
    with open("langgraph_diagram.png", "wb") as file:
        file.write(png_data)

    print("LangGraph diagram saved as langgraph_diagram.png")
    return graph
