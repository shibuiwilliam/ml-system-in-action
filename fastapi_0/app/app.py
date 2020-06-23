import os
from fastapi import FastAPI
import logging

from api import health, iris

TITLE = os.getenv("FASTAPI_TITLE", "fastapi application")
DESCRIPTION = os.getenv("FASTAPI_DESCRIPTION", "fastapi description")
VERSION = os.getenv("FASTAPI_VERSION", "fastapi version")

logger = logging.getLogger(__name__) 
logger.info(f"Starts {TITLE}:{VERSION}")

app = FastAPI(
    title=TITLE,
    description=DESCRIPTION,
    version=VERSION
)

app.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

app.include_router(
    iris.router,
    prefix="/iris",
    tags=["iris"]
)