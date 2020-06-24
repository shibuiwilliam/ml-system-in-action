from fastapi import APIRouter
from . import _iris
import logging

logger = logging.getLogger(__name__) 
router = APIRouter()


@router.get('/test')
def test():
    prediction = _iris.test()
    return prediction


@router.post('/predict')
def predict(iris_data: _iris.IrisData):
    prediction = _iris.predict(iris_data)
    return prediction


@router.post('/predict_async')
async def predict_async(iris_data: _iris.IrisData):
    _id = await _iris.predict_async(iris_data)
    return _id
