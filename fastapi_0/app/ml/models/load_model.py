import os
from typing import List


def get_model_files() -> List[str]:
    files = [f for f in os.listdir("./") if f.endswith(".pkl")]
    return files


def get_model_file(name: str) -> str:
    files = get_model_files()

