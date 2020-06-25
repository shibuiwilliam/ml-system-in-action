from fastapi import APIRouter, BackgroundTasks
import logging

from . import _iris
from ml.iris.iris_predictor import IrisData

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get('/test')
def test():
    prediction = _iris.test()
    return prediction


@router.post('/predict')
def predict(iris_data: IrisData,
            background_tasks: BackgroundTasks):
    prediction = _iris.predict(iris_data, background_tasks)
    return prediction


@router.post('/predict_async')
async def predict_async(iris_data: IrisData,
                        background_tasks: BackgroundTasks):
    job_id = await _iris.predict_async_post(iris_data, background_tasks)
    return job_id


@router.get('/predict_async/{job_id}')
def predict_async(job_id: str):
    return _iris.predict_async_get(job_id)
