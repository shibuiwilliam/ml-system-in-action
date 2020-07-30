from fastapi import APIRouter, Body
import os
import logging
import aiohttp
import asyncio
from typing import Dict, Any
from pydantic import BaseModel
import uuid

from src.api_composition_proxy.configurations import _Services
from src.api_composition_proxy import helpers

logger = logging.getLogger(__name__)

router = APIRouter()


async def _get_redirect(session, url: str) -> Dict[str, Any]:
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


@router.get('/health')
async def health() -> Dict[str, Any]:
    logger.info(f'Health check target urls')
    responses = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
        tasks = [
            asyncio.ensure_future(
                _get_redirect(
                    session,
                    helpers.path_builder(v, '/health'))
                ) for v in _Services().urls.values()]
        _responses = await asyncio.gather(*tasks)
        for r in _responses:
            for k, v in r.items():
                responses[k] = v
        logger.info(f'responses: {responses}')
        return responses


@router.get('/{redirect_path:path}')
async def get_redirect(redirect_path: str) -> Dict[str, Any]:
    logger.info(f'GET redirect to: /{redirect_path}')
    responses = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
        tasks = [
            asyncio.ensure_future(
                _get_redirect(
                    session,
                    helpers.path_builder(v, redirect_path))
                ) for v in _Services().urls.values()]
        _responses = await asyncio.gather(*tasks)
        for r in _responses:
            for k, v in r.items():
                responses[k] = v
        logger.info(f'responses: {responses}')
        return responses
