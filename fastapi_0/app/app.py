import os
from fastapi import FastAPI
import logging

from api import health, predict
from constants import CONSTANTS

TITLE = os.getenv('FASTAPI_TITLE', 'fastapi application')
DESCRIPTION = os.getenv('FASTAPI_DESCRIPTION', 'fastapi description')
VERSION = os.getenv('FASTAPI_VERSION', '0.1')

# can be docker, docker_compose, or kubernetes
PLATFORM = os.getenv('PLATFORM', CONSTANTS.PLATFORM_DOCKER)
PLATFORM = PLATFORM if PLATFORM in (
    CONSTANTS.PLATFORM_DOCKER,
    CONSTANTS.PLATFORM_DOCKER_COMPOSE,
    CONSTANTS.PLATFORM_KUBERNETES) else CONSTANTS.PLATFORM_DOCKER

logger = logging.getLogger(__name__)
logger.info(f'starts {TITLE}:{VERSION} in {PLATFORM}')

os.makedirs(CONSTANTS.DATA_DIRECTORY, exist_ok=True)

if PLATFORM == CONSTANTS.PLATFORM_DOCKER:
    os.makedirs(CONSTANTS.DATA_FILE_DIRECTORY, exist_ok=True)


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
