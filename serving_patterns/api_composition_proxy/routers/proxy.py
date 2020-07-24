from urllib.parse import urlparse, urlunparse
from fastapi import APIRouter
import os
import logging
import aiohttp

from api_composition_proxy.configurations import Services

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get('/health')
def health():
    return {'health': 'ok'}


@router.get('/redirect')
async def redirect():
    logger.info('redirect!')
    responses = {}
    for k,v in Services().services.items():
        async with aiohttp.request('GET', v) as response:
            logger.info(f'target: {v}')
            logger.info(f'status_code: {response.status}')
            responses[v] = response.status
    return responses
