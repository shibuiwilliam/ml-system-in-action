import asyncio
from urllib.parse import urlparse, urlunparse
import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)


async def get_async(url: str) -> Dict[str, str]:
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, requests.request, 'GET', url)
    logger.info(f'{url}: {response.status_code}')
    return response.status_code