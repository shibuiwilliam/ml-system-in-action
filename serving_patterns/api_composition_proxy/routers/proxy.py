from urllib.parse import urlparse, urlunparse
from fastapi import APIRouter
import requests
import os
import logging

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
        response = requests.request('GET', v, allow_redirects=True)
        logger.info(f'target: {v}')
        logger.info(f'status_code: {response.status_code}')
        responses[v] = response.status_code
    return responses
