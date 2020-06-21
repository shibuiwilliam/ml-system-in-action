from typing import Dict


def health() -> Dict[str, str]:
    return {"health": "ok"}


def health_sync() -> Dict[str, str]:
    return health()


def health_async() -> Dict[str, str]:
    return health()