from fastapi import APIRouter, BackgroundTasks, UploadFile, File
import logging

from app.api import _predict_image
from app.ml.active_predictor import Data

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get('')
async def test():
    result = await _predict_image._test()
    return result


@router.post('')
async def predict(file: UploadFile = File(...),
                  background_tasks: BackgroundTasks = BackgroundTasks()):
    result = await _predict_image._predict(file, background_tasks)
    return result


@router.post('/json')
async def predict_json(data: Data,
                       background_tasks: BackgroundTasks = BackgroundTasks()):
    result = await _predict_image._predict_json(data, background_tasks)
    return result


@router.get('/labels')
def labels():
    return _predict_image._labels()


@router.get('/label')
async def test_label():
    result = await _predict_image._test_label()
    return result


@router.post('/label')
async def predict_label(file: UploadFile = File(...),
                        background_tasks: BackgroundTasks = BackgroundTasks()):
    result = await _predict_image._predict_label(file, background_tasks)
    return result


@router.post('/label/json')
async def predict_label_json(data: Data,
                             background_tasks: BackgroundTasks = BackgroundTasks()):
    result = await _predict_image._predict_label_json(data, background_tasks)
    return result


@router.post('/async')
async def predict_async(file: UploadFile = File(...),
                        background_tasks: BackgroundTasks = BackgroundTasks()):
    return await _predict_image._predict_async_post(file, background_tasks)


@router.post('/async/json')
async def predict_async_json(data: Data,
                             background_tasks: BackgroundTasks = BackgroundTasks()):
    return await _predict_image._predict_async_post_json(data, background_tasks)


@router.get('/async/{job_id}')
def predict_async(job_id: str):
    return _predict_image._predict_async_get(job_id)


@router.get('/async/label/{job_id}')
def predict_async(job_id: str):
    return _predict_image._predict_async_get_label(job_id)
