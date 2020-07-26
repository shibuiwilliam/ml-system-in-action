from typing import Dict, Any, List
from fastapi import BackgroundTasks
import uuid
import logging
from PIL import Image
import io
import numpy as np
import base64

from middleware.profiler import do_cprofile
from middleware.redis_client import redis_client
from jobs import store_data_job
from configurations.constants import PLATFORM_ENUM
from configurations.configurations import _PlatformConfigurations
from app.ml.active_predictor import Data, DataInterface, DataConverter, active_predictor
from configurations.configurations import _CacheConfigurations
from configurations.configurations import _FileConfigurations
from app.api import _predict as _parent_predict

logger = logging.getLogger(__name__)


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


def _predict_from_redis_cache(job_id: str,
                                    data_class: callable = Data) -> Data:
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


async def _test(data: Data = Data()) -> Dict[str, int]:
    data.image_data = data.test_data
    return _parent_predict._test(data, __predict)


async def _test_label(data: Data = Data()) -> Dict[str, Dict[str, float]]:
    data.image_data = data.test_data
    return _parent_predict._test_label(data, __predict_label)


async def _predict(data: Data,
                   background_tasks: BackgroundTasks = BackgroundTasks()) -> Dict[str, List[float]]:
    image = base64.b64decode(str(data.image_data))
    io_bytes = io.BytesIO(image)
    data.image_data = Image.open(io_bytes)
    __predict(data)
    job_id = store_data_job._save_data_job(data, background_tasks, False)
    return {'prediction': data.prediction, 'job_id': job_id}


async def _predict_label(data: Data,
                         background_tasks: BackgroundTasks = BackgroundTasks()) -> Dict[str, List[float]]:
    image = base64.b64decode(str(data.image_data))
    io_bytes = io.BytesIO(image)
    data.image_data = Image.open(io_bytes)
    label_proba = __predict_label(data)
    job_id = store_data_job._save_data_job(data, background_tasks, False)
    return {'prediction': label_proba, 'job_id': job_id}


async def _predict_async_post(data: Data,
                              background_tasks: BackgroundTasks = BackgroundTasks()) -> Dict[str, List[float]]:
    image = base64.b64decode(str(data.image_data))
    io_bytes = io.BytesIO(image)
    data.image_data = Image.open(io_bytes)
    job_id = store_data_job._save_data_job(data, background_tasks, True)
    return {'job_id': job_id}


@do_cprofile
def _predict_async_get(job_id: str) -> Dict[str, List[float]]:
    return _parent_predict._predict_async_get(job_id)


@do_cprofile
def _predict_async_get_label(
        job_id: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    return _parent_predict._predict_async_get_label(job_id)
