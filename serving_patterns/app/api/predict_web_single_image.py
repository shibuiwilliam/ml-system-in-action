from fastapi import APIRouter, BackgroundTasks, UploadFile, File
import logging

from app.api import _predict_image
from app.ml.active_predictor import Data

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
