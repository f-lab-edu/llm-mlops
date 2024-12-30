import logging
import os

import bentoml
from bentoml.exceptions import InvalidArgument, NotFound
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from graph import build_graph
from utils.env_setup import get_device

load_dotenv()
# Bentoml 서버 로깅 설정
bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.setLevel(logging.INFO)


class LlmGennerationParams(BaseModel):
    prompt: str = Field(
        default="What is the tallest building in the world?", description="Prompt Text"
    )


# Define the BentoML Service
@bentoml.service
class LlmService:
    def __init__(self):
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        # OpenAI 모델 사용 여부 예외처리
        use_openai = os.getenv("USE_OPENAI").lower()
        if use_openai not in ["true", "false"]:
            raise InvalidArgument("OpenAI 사용여부 설정이 제대로 되있지 않습니다.")

        if use_openai == "true":
            bentoml_logger.info("Using OpenAI Model")
            self.model = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL"), max_tokens=os.getenv("MAX_TOKENS")
            )
        else:
            self.device = get_device()
            tokenizer = AutoTokenizer.from_pretrained(os.getenv("LLM_MODEL"))
            model = AutoModelForCausalLM.from_pretrained(os.getenv("LLM_MODEL")).to(
                self.device
            )
            model.generation_config.pad_token_id = tokenizer.eos_token_id

            if tokenizer.chat_template is None:
                raise NotFound(
                    "Huggingface tokenizer에 chat_template이 설정되지 않았습니다! tokenizer_config.json에 chat_template이 있는지 확인해주세요."
                )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=int(os.getenv("MAX_TOKENS")),
                device=self.device,
            )

            llm = HuggingFacePipeline(pipeline=pipe)
            self.model = ChatHuggingFace(llm=llm)
            bentoml_logger.info(f"""{os.getenv("LLM_MODEL")} loaded to {self.device}""")

        self.graph = build_graph(self.model)

    @bentoml.api
    def generate(self, params: LlmGennerationParams) -> dict:
        """prompt를 입력받으면, __init__에서 설정한 tokenizer과 LLM 모델로 답변을 생성합니다

        Args:
            prompt (str): 사용자의 입력 prompt
            temperature (float): LLM모델 답변의 무작위성 조정 파라미터 (0~1 사이)

        Returns:
            str: LLM으로 생성된 prompt에 대한 답변
        """
        prompt = params.prompt

        inputs = {"query": prompt}
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                bentoml_logger.info(f"Output from node '{key}':")
                bentoml_logger.info("---")
                bentoml_logger.info(value)
            bentoml_logger.info("\n---\n")

        return {"success": True}
