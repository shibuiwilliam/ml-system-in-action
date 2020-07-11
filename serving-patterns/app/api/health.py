from fastapi import APIRouter
from app.api import _health
import logging

from app.middleware.profiler import do_cprofile

logger = logging.getLogger(__name__)
router = APIRouter()


@do_cprofile
@router.get('')
def health():
    return _health.health()


@do_cprofile
@router.get('/sync')
def health_sync():
    return _health.health_sync()


@do_cprofile
@router.get('/async')
async def health_async():
    result = await _health.health_async()
    return result
