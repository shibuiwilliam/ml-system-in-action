import os
from typing import List
import constants
import logging

logger = logging.getLogger(__name__) 


def get_model_files() -> List[str]:
    return [f for f in os.listdir("./ml/models/") if f.split(".")[-1] in constants.CONSTANTS.MODEL_EXTENTIONS]


def get_model_file(name: str) -> str:
    files = get_model_files()
    index = files.index(name) if name in files else -1
    return f"./ml/models/{files[index]}" if index != -1 else None
