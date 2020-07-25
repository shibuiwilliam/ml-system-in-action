from fastapi import FastAPI
import logging

from api_composition_proxy.routers import proxy
from api_composition_proxy.configurations import _FastAPIConfigurations, _PlatformConfigurations

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
    proxy.router,
    prefix='',
    tags=['']
)
