from fastapi import APIRouter, BackgroundTasks
import logging

from src.app.api import _predict
from src.app.ml.active_predictor import Data

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get('')
def test():
    return _predict._test()


@router.post('')
def predict(data: Data,
            background_tasks: BackgroundTasks):
    return _predict._predict(data, background_tasks)


@router.get('/labels')
def labels():
    return _predict._labels()


@router.get('/label')
def test_label():
    return _predict._test_label()


@router.post('/label')
def predict_label(data: Data,
                  background_tasks: BackgroundTasks = BackgroundTasks()):
    return _predict._predict_label(data, background_tasks)
