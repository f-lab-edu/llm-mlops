import logging
import os
from typing import List

import bentoml
import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
# Bentoml 서버 로깅 설정
bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.setLevel(logging.INFO)


class MultiStringInput(BaseModel):
    sentences: List[str] = Field(
        description="List of multiple sentences to embed (in string type)"
    )


class SingleStringInput(BaseModel):
    sentence: str = Field(description="A single sentences to embed (in string type)")


@bentoml.service
class SentenceEmbeddingService:
    def __init__(self) -> None:
        """초기화 메서드
        Embedding BentoML 서비스 클래스의 인스턴스를 초기화합니다.
        """
        from sentence_transformers import SentenceTransformer

        # API 서버에 가용 가능한 GPU를 설정
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            bentoml_logger.info("Using CUDA for inference.")

        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            bentoml_logger.info("Using MPS (Apple Silicon) for inference.")
        else:
            # 사용 가능한 GPU가 없다면 cpu로 inference 대체
            self.device = torch.device("cpu")
            bentoml_logger.info("Using CPU for inference.")

        # bentofile.yaml에 env 변수를 참조하여 특정 모델을 초기화합니다
        self.embedding_model = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL"), trust_remote_code=True
        ).to(self.device)

    @bentoml.api
    def multiple_embed(self, sentences: MultiStringInput) -> np.ndarray:
        """list에 담긴 각 sentence를 embedding 모델로 embed하는 API 함수

        Args:
            sentences (list[str]): Embedding하고자 하는 input 문장/문단이 담긴 list

        Returns:
            np.ndarray: embedding 결과 [(len(sentences), embed_dim) shape의 ndarray 반환]
        """
        embeddings = self.embedding_model.encode(sentences.sentences)

        return embeddings

    @bentoml.api
    def embed(self, sentence: SingleStringInput) -> np.ndarray:
        """string input 문장을 embedding 모델로 embed하는 API 함수

        Args:
            sentence (str): Embedding하고자 하는 input 문장/문단

        Returns:
            np.ndarray: embedding 결과 [(embed_dim, ) shape의 ndarray 반환]
        """
        embeddings = self.embedding_model.encode(sentence.sentence)

        return embeddings
