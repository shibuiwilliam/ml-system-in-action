import os
from typing import List
from constants import CONSTANTS
import logging

logger = logging.getLogger(__name__) 


def get_model_files() -> List[str]:
    return [f for f in os.listdir(CONSTANTS.MODEL_DIRECTORY) if f.split(".")[-1] in CONSTANTS.MODEL_EXTENTIONS]


def get_model_file(name: str) -> str:
    files = get_model_files()
    index = files.index(name) if name in files else -1
    return os.path.join(CONSTANTS.MODEL_DIRECTORY, files[index]) if index != -1 else None
