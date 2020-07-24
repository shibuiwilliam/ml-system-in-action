from urllib.parse import urlparse, urlunparse
from fastapi import APIRouter, Body
import os
import logging
import aiohttp
import asyncio
from typing import Dict, Any
from pydantic import BaseModel


from api_composition_proxy.configurations import Services
from api_composition_proxy import helpers

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get('/health')
def health():
    return {'health': 'ok'}


async def _redirect_get(session, url: str) -> Dict[str, Any]:
    async with session.get(url) as response:
        response_json = await response.json()
        resp = {
            url: {
                'response': response_json,
                'status_code': response.status
                }
            }
        logger.info(f'response: {resp}')
        return resp


@router.get('/redirect/{redirect_path:path}')
async def redirect_get(redirect_path: str) -> Dict[str, Any]:
    logger.info(f'GET redirect to: {redirect_path}')
    responses = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
        tasks = [
            asyncio.ensure_future(
                _redirect_get(
                    session,
                    helpers.url_builder(v, redirect_path))) for v in Services().services.values()]
        _responses = await asyncio.gather(*tasks)
        for r in _responses:
            for k, v in r.items():
                responses[k] = v
        logger.info(f'responses: {responses}')
        return responses


class Data(BaseModel):
    data: Any = None


async def _redirect_post_json(session, url: str, data: Dict[Any, Any]) -> Dict[str, Any]:
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


@router.post('/redirect/{redirect_path:path}')
async def redirect_post_json(redirect_path: str, data: Data = Body(...)) -> Dict[str, Any]:
    logger.info(f'POST redirect to: {redirect_path}')
    logger.info(f'POST body: {data}')
    responses = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
        tasks = [
            asyncio.ensure_future(
                _redirect_post_json(
                    session,
                    helpers.url_builder(v, redirect_path),
                    data.data)) for v in Services().services.values()]
        _responses = await asyncio.gather(*tasks)
        for r in _responses:
            for k, v in r.items():
                responses[k] = v
        logger.info(f'responses: {responses}')
        return responses
