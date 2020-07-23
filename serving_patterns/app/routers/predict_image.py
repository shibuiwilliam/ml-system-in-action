from fastapi import APIRouter, BackgroundTasks, UploadFile, File
import logging

from app.api import _predict_image

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get('')
def test():
    return _predict_image._test()


@router.post('')
def predict(file: UploadFile = File(...),
            background_tasks: BackgroundTasks = BackgroundTasks()):
    return _predict_image._predict(file, background_tasks)


@router.get('/labels')
def labels():
    return _predict_image._labels()


@router.get('/label')
def test_label():
    return _predict_image._test_label()


@router.post('/label')
def predict_label(file: UploadFile = File(...),
                  background_tasks: BackgroundTasks = BackgroundTasks()):
    return _predict_image._predict_label(file, background_tasks)


@router.post('/async')
async def predict_async(file: UploadFile = File(...),
                        background_tasks: BackgroundTasks = BackgroundTasks()):
    return await _predict_image._predict_async_post(file, background_tasks)


@router.get('/async/{job_id}')
def predict_async(job_id: str):
    return _predict_image._predict_async_get(job_id)


@router.get('/async/label/{job_id}')
def predict_async(job_id: str):
    return _predict_image._predict_async_get_label(job_id)

