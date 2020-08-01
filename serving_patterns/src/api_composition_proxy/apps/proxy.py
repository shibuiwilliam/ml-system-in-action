from fastapi import FastAPI
import logging

from src.api_composition_proxy.routers import proxy, health
from src.api_composition_proxy.configurations import FastAPIConfigurations
from src.configurations.configurations import PlatformConfigurations


logger = logging.getLogger(__name__)
logger.info(f'starts {FastAPIConfigurations.title}:{FastAPIConfigurations.version}')
logger.info(f'platform: {PlatformConfigurations.platform}')

app = FastAPI(
    title=FastAPIConfigurations.title,
    description=FastAPIConfigurations.description,
    version=FastAPIConfigurations.version,
)

app.include_router(
    health.router,
    prefix='/health',
    tags=['health']
)

app.include_router(
    proxy.router,
    prefix='/redirect',
    tags=['redirect']
)
