from fastapi import APIRouter, Body, UploadFile, File
import os
import logging
import aiohttp
import asyncio
from PIL import Image
import io
from typing import Dict, Any
from pydantic import BaseModel
import uuid

from api_composition_proxy.configurations import Services
from api_composition_proxy import helpers

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get('/health')
def health():
    return {'health': 'ok'}


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


@router.get('/redirect/{redirect_path:path}')
async def get_redirect(redirect_path: str) -> Dict[str, Any]:
    logger.info(f'GET redirect to: /{redirect_path}')
    responses = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
        tasks = [
            asyncio.ensure_future(
                _get_redirect(
                    session,
                    helpers.url_builder(
                        v,
                        redirect_path))) for v in Services().services.values()]
        _responses = await asyncio.gather(*tasks)
        for r in _responses:
            for k, v in r.items():
                responses[k] = v
        logger.info(f'responses: {responses}')
        return responses


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


@router.post('/redirect_json/{redirect_path:path}')
async def redirect_json(redirect_path: str, data: Data = Body(...)) -> Dict[str, Any]:
    logger.info(f'POST redirect to: /{redirect_path}')
    responses = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
        tasks = [
            asyncio.ensure_future(
                _post_redirect_json(
                    session,
                    helpers.url_builder(v, redirect_path),
                    data.data)) for v in Services().services.values()]
        _responses = await asyncio.gather(*tasks)
        for r in _responses:
            for k, v in r.items():
                responses[k] = v
        logger.info(f'responses: {responses}')
        return responses


async def _post_redirect_file(session, url: str, data: Any) -> Dict[str, Any]:
    data.file.seek(0)
    _data = aiohttp.FormData()
    _data.add_field(
        'file',
        data.file.read(),
        filename='image.jpeg',
        content_type='multipart/form-data')
    async with session.post(url, data=_data) as response:
        response_json = await response.json()
        resp = {
            url: {
                'response': response_json,
                'status_code': response.status
            }
        }
        logger.info(f'response: {resp}')
        return resp


@router.post('/redirect_file/{redirect_path:path}')
async def post_redirect_file(redirect_path: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    logger.info(f'POST redirect to: /{redirect_path}')
    responses = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
        tasks = [
            asyncio.ensure_future(
                _post_redirect_file(
                    session,
                    helpers.url_builder(v, redirect_path),
                    file)) for v in Services().services.values()]
        _responses = await asyncio.gather(*tasks)
        for r in _responses:
            for k, v in r.items():
                responses[k] = v
        logger.info(f'responses: {responses}')
        return responses
