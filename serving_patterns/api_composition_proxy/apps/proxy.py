from fastapi import FastAPI
import logging

from api_composition_proxy.routers import proxy

logger = logging.getLogger(__name__)
logger.info(f'starts proxy')

app = FastAPI(
    title='proxy',
    description='proxy',
    version='0.1'
)

app.include_router(
    proxy.router,
    prefix='',
    tags=['']
)
