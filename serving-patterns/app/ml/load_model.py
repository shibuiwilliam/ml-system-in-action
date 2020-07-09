import os
from typing import List
from app.constants import CONSTANTS
import logging

logger = logging.getLogger(__name__)


def get_model_files(
        model_directory: str = CONSTANTS.MODEL_DIRECTORY) -> List[str]:
    return [f for f in os.listdir(CONSTANTS.MODEL_DIRECTORY)
            if f.split(".")[-1] in CONSTANTS.MODEL_EXTENTIONS]


def get_model_file(
        name: str,
        model_directory: str = CONSTANTS.MODEL_DIRECTORY) -> str:
    found_model_filepath = name if model_directory in name else \
        os.path.join(model_directory, name)
    logger.info(f'found: {found_model_filepath}')
    return found_model_filepath
