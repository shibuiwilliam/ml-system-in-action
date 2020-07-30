import os
from fastapi import FastAPI
import logging

from src.configurations.configurations import _PlatformConfigurations
from src.app.routers import health, predict_asynchronous
from src.app.configurations import _FastAPIConfigurations

logger = logging.getLogger(__name__)
logger.info(
    f'starts {_FastAPIConfigurations().title}:{_FastAPIConfigurations().version}')
logger.info(f'platform: {_PlatformConfigurations().platform}')


app = FastAPI(
    title=_FastAPIConfigurations().title,
    description=_FastAPIConfigurations().description,
    version=_FastAPIConfigurations().version,
)

app.include_router(
    health.router,
    prefix='/health',
    tags=['health']
)

app.include_router(
    predict_asynchronous.router,
    prefix='/predict',
    tags=['predict']
)
