from fastapi import APIRouter, BackgroundTasks
import logging

from . import _predict
from app.ml.active_predictor import Data

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get('')
def test():
    prediction = _predict._test()
    return prediction


@router.post('')
async def predict_async(data: Data,
                        background_tasks: BackgroundTasks):
    return await _predict._predict_async_post(data, background_tasks)


@router.get('/{job_id}')
def predict_async(job_id: str):
    return _predict._predict_async_get(job_id)
