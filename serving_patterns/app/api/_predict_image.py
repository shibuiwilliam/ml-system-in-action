from typing import Dict, Any, List
from fastapi import BackgroundTasks, UploadFile, File
import uuid
import logging
from PIL import Image
import io
import numpy as np

from app.middleware.profiler import do_cprofile
from app.jobs import store_data_job
from app.ml.active_predictor import Data, DataInterface, DataConverter, active_predictor
from app.constants import CONSTANTS, PLATFORM_ENUM
from app.configurations import _PlatformConfigurations
from app.middleware.redis_client import redis_client


logger = logging.getLogger(__name__)


@do_cprofile
def _save_data_job(data: Data,
                   background_tasks: BackgroundTasks,
                   enqueue: bool = False) -> str:
    if _PlatformConfigurations().platform == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        incr = redis_client.get(CONSTANTS.REDIS_INCREMENTS)
        num_files = 0 if incr is None else incr
        job_id = f'{str(uuid.uuid4())}_{num_files}'
        task = store_data_job.SaveDataRedisJob(
            job_id=job_id,
            data=data,
            enqueue=enqueue)

    elif _PlatformConfigurations().platform == PLATFORM_ENUM.KUBERNETES.value:
        pass
    else:
        pass
    background_tasks.add_task(task)
    return job_id


@do_cprofile
def __predict(data: Data):
    image_data = Image.open(data.image_data)
    output_np = active_predictor.predict(image_data)
    reshaped_output_nps = DataConverter.reshape_output(output_np)
    data.prediction = reshaped_output_nps.tolist()
    logger.info(f'prediction: {data.__dict__}')


@do_cprofile
def __predict_label(data: Data) -> Dict[str, float]:
    image_data = Image.open(data.image_data)
    output_np = active_predictor.predict(image_data)
    reshaped_output_nps = DataConverter.reshape_output(output_np)
    argmax = int(np.argmax(reshaped_output_nps[0]))
    data.prediction = reshaped_output_nps.tolist()
    logger.info(f'prediction: {data.__dict__}')
    return {data.labels[argmax]: data.prediction[0][argmax]}


# def _predict_from_redis_cache(job_id: str) -> Data:
#     data_dict = store_data_job.load_data_redis(job_id)
#     if data_dict is None:
#         return None
#     data = Data(**data_dict)
#     __predict(data)
#     return data


def _labels() -> Dict[str, List[str]]:
    return {'labels': Data().labels}


def _test() -> Dict[str, int]:
    data = Data()
    data.image_data = data.test_data
    __predict(data)
    return {'prediction': data.prediction}


def _test_label() -> Dict[str, int]:
    data = Data()
    data.image_data = data.test_data
    label_proba = __predict_label(data)
    return {'prediction': label_proba}


def _predict(file: UploadFile = File(...),
             background_tasks: BackgroundTasks = BackgroundTasks()) -> Dict[str, int]:
    data = Data()
    data.image_data = io.BytesIO(file.file.read())
    __predict(data)
    _save_data_job(data, background_tasks, False)
    return {'prediction': data.prediction}


def _predict_label(file: UploadFile = File(...),
             background_tasks: BackgroundTasks = BackgroundTasks()) -> Dict[str, int]:
    data = Data()
    data.image_data = io.BytesIO(file.file.read())
    label_proba = __predict_label(data)
    _save_data_job(data, background_tasks, False)
    return {'prediction': label_proba}


# async def _predict_async_post(
#         data: Data,
#         background_tasks: BackgroundTasks) -> Dict[str, str]:
#     job_id = _save_data_job(data, background_tasks, True)
#     return {'job_id': job_id}


# @do_cprofile
# def _predict_async_get(job_id: str) -> Dict[str, int]:
#     result = {job_id: {'prediction': []}}
#     if _PlatformConfigurations().platform == PLATFORM_ENUM.DOCKER_COMPOSE.value:
#         data_dict = store_data_job.load_data_redis(job_id)
#         result[job_id]['prediction'] = data_dict['prediction']
#         return result

#     elif _PlatformConfigurations().platform == PLATFORM_ENUM.KUBERNETES.value:
#         pass

#     else:
#         pass
