import os
from fastapi import FastAPI

from api import health

TITLE = os.getenv("FASTAPI_TITLE", "fastapi application")
DESCRIPTION = os.getenv("FASTAPI_DESCRIPTION", "fastapi description")
VERSION = os.getenv("FASTAPI_VERSION", "fastapi version")

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
