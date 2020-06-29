import os
from fastapi import FastAPI
import logging

from api import health, predict
from constants import CONSTANTS, PLATFORM_ENUM

TITLE = os.getenv('FASTAPI_TITLE', 'ML Rest')
DESCRIPTION = os.getenv('FASTAPI_DESCRIPTION', 'ML rest description')
VERSION = os.getenv('FASTAPI_VERSION', '0.1')

# can be docker_compose or kubernetes
PLATFORM = os.getenv('PLATFORM', PLATFORM_ENUM.DOCKER_COMPOSE.value)
PLATFORM = PLATFORM if PLATFORM in (
    PLATFORM_ENUM.DOCKER_COMPOSE.value,
    PLATFORM_ENUM.KUBERNETES.value) else PLATFORM_ENUM.DOCKER_COMPOSE.value

logger = logging.getLogger(__name__)
logger.info(f'starts {TITLE}:{VERSION} in {PLATFORM}')

os.makedirs(CONSTANTS.DATA_DIRECTORY, exist_ok=True)


app = FastAPI(
    title=TITLE,
    description=DESCRIPTION,
    version=VERSION
)

app.include_router(
    health.router,
    prefix='/health',
    tags=['health']
)

app.include_router(
    predict.router,
    prefix='/predict',
    tags=['predict']
)
