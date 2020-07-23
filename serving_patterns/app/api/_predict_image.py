from typing import Dict, Any, List
from fastapi import BackgroundTasks, UploadFile, File
import uuid
import logging
from PIL import Image
import io
import numpy as np
import base64

from app.middleware.profiler import do_cprofile
from app.jobs import store_data_job
from app.ml.active_predictor import Data, DataInterface, DataConverter, active_predictor
from app.constants import CONSTANTS, PLATFORM_ENUM
from app.configurations import _PlatformConfigurations, _CacheConfigurations, _FileConfigurations
from app.middleware.redis_client import redis_client
from app.api import _predict as _parent_predict

logger = logging.getLogger(__name__)


@do_cprofile
def _save_data_job(data: Data,
                   background_tasks: BackgroundTasks,
                   enqueue: bool = False) -> str:
    return _parent_predict._save_data_job(data, background_tasks, enqueue)


@do_cprofile
def __predict(data: Data):
    if isinstance(data.image_data, Image.Image):
        image_data = data.image_data
    elif isinstance(data.image_data, np.ndarray):
        image_data = Image.fromarray(data.image_data)
    elif isinstance(data.image_data, List):
        image_data = Image.fromarray(np.array(data.image_data))
    else:
        image_data = Image.open(data.image_data)
    output_np = active_predictor.predict(image_data)
    reshaped_output_nps = DataConverter.reshape_output(output_np)
    data.prediction = reshaped_output_nps.tolist()
    logger.info(f'prediction: {data.__dict__}')


@do_cprofile
def __predict_label(data: Data) -> Dict[str, float]:
    return _parent_predict.__predict_label(data, __predict)


def _predict_from_redis_cache(job_id: str, data_class: callable = Data) -> Data:
    data_dict = store_data_job.load_data_redis(job_id)
    if data_dict is None:
        return None
    if isinstance(data_dict['image_data'], Image.Image):
        pass
    else:
        data_dict['image_data'] = Image.open(data_dict['image_data'])
    data = data_class(**data_dict)
    __predict(data)
    return data


def _labels(data_class: callable = Data) -> Dict[str, List[str]]:
    return _parent_predict._labels(data_class)


def _test(data: Data = Data()) -> Dict[str, int]:
    return _parent_predict._test(data, __predict)


def _test_label(data: Data = Data()) -> Dict[str, Dict[str, float]]:
    return _parent_predict._test_label(data, __predict_label)


def _predict(file: UploadFile = File(...),
             background_tasks: BackgroundTasks = BackgroundTasks(),
             data: Data = Data()) -> Dict[str, List[float]]:
    file_read = file.file.read()
    io_bytes = io.BytesIO(file_read)
    data.image_data = Image.open(io_bytes)
    __predict(data)
    _save_data_job(data, background_tasks, False)
    return {'prediction': data.prediction}


def _predict_label(file: UploadFile = File(...),
                   background_tasks: BackgroundTasks = BackgroundTasks(),
                   data: Data = Data()) -> Dict[str, Dict[str, float]]:
    file_read = file.file.read()
    io_bytes = io.BytesIO(file_read)
    data.image_data = Image.open(io_bytes)
    label_proba = __predict_label(data)
    _save_data_job(data, background_tasks, False)
    return {'prediction': label_proba}


async def _predict_async_post(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = BackgroundTasks(),
        data: Data = Data()) -> Dict[str, str]:
    file_read = file.file.read()
    io_bytes = io.BytesIO(file_read)
    data.image_data = Image.open(io_bytes)
    job_id = _save_data_job(data, background_tasks, True)
    return {'job_id': job_id}


@do_cprofile
def _predict_async_get(job_id: str) -> Dict[str, List[float]]:
    return _parent_predict._predict_async_get(job_id)


@do_cprofile
def _predict_async_get_label(job_id: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    return _parent_predict._predict_async_get_label(job_id)
