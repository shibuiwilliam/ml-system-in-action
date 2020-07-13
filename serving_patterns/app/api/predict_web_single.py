from fastapi import APIRouter, BackgroundTasks
import logging

from app.api import _predict
from app.ml.active_predictor import Data

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get('')
def test():
    return _predict._test()


@router.post('')
def predict(data: Data,
            background_tasks: BackgroundTasks):
    return _predict._predict(data, background_tasks)
