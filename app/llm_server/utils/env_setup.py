import logging

import torch

bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.setLevel(logging.INFO)


def get_device():

    # API 서버에 가용 가능한 GPU를 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
        bentoml_logger.info("Using CUDA for inference.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        bentoml_logger.info("Using MPS (Apple Silicon) for inference.")
    else:
        # 사용 가능한 GPU가 없다면 cpu로 inference 대체
        device = torch.device("cpu")
        bentoml_logger.info("Using CPU for inference.")
    return device
