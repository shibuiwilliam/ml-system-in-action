import os
from typing import List
from constants import CONSTANTS
import logging

logger = logging.getLogger(__name__)


def get_model_files() -> List[str]:
    return [f for f in os.listdir(CONSTANTS.MODEL_DIRECTORY)
            if f.split(".")[-1] in CONSTANTS.MODEL_EXTENTIONS]


def get_model_file(name: str, fallback_name: str) -> str:
    is_file_exist = True if name in get_model_files() else False
    found_model_filepath = os.path.join(
        CONSTANTS.MODEL_DIRECTORY, name
    ) if is_file_exist else os.path.join(
        CONSTANTS.MODEL_DIRECTORY, fallback_name
    )
    logger.info(f'found: {found_model_filepath}')
    return found_model_filepath
