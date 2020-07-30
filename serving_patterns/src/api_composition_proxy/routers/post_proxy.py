from fastapi import APIRouter, Body
import os
import logging
import aiohttp
import asyncio
from PIL import Image
from typing import Dict, Any
from pydantic import BaseModel
import uuid

from src.api_composition_proxy.configurations import _Services
from src.api_composition_proxy import helpers

logger = logging.getLogger(__name__)

router = APIRouter()


class Data(BaseModel):
    data: Any = None


async def _post_redirect_json(session, url: str, data: Dict[Any, Any]) -> Dict[str, Any]:
    async with session.post(url, json=data) as response:
        response_json = await response.json()
        resp = {
            url: {
                'response': response_json,
                'status_code': response.status
            }
        }
        logger.info(f'response: {resp}')
        return resp


@router.post('/{redirect_path:path}')
async def redirect_json(redirect_path: str, data: Data = Body(...)) -> Dict[str, Any]:
    logger.info(f'POST redirect to: /{redirect_path}')
    responses = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
        tasks = [
            asyncio.ensure_future(
                _post_redirect_json(
                    session,
                    helpers.path_builder(v, redirect_path),
                    data.data)
                ) for v in _Services().urls.values()]
        _responses = await asyncio.gather(*tasks)
        for r in _responses:
            for k, v in r.items():
                responses[k] = v
        logger.info(f'responses: {responses}')
        return responses
