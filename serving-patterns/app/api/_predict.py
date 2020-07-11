from typing import Dict, Any
from fastapi import BackgroundTasks
import uuid
import logging

from app.middleware.profiler import do_cprofile
from app.jobs import store_data_job
from app.ml.active_predictor import Data, DataExtension, active_predictor
from app.constants import CONSTANTS, PLATFORM_ENUM
from app.configurations import _PlatformConfigurations
from app.middleware.redis_client import redis_client


logger = logging.getLogger(__name__)


def _save_data_job(data: Data,
                   background_tasks: BackgroundTasks) -> str:
    if _PlatformConfigurations().platform == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        incr = redis_client.get(CONSTANTS.REDIS_INCREMENTS)
        num_files = 0 if incr is None else incr
        job_id = f'{str(uuid.uuid4())}_{num_files}'
        task = store_data_job.SaveDataRedisJob(
            job_id=job_id,
            data=data)

    elif _PlatformConfigurations().platform == PLATFORM_ENUM.KUBERNETES.value:
        pass
    else:
        pass
    background_tasks.add_task(task)
    return job_id


def __predict(data: Data):
    data_extension = DataExtension(data)
    input_np = data_extension.convert_input_data_to_np()
    output_np = active_predictor.predict(input_np)
    reshaped_output_nps = data_extension.reshape_output(output_np)
    data.prediction = reshaped_output_nps.tolist()
    logger.info(f'prediction: {data.__dict__}')


def _predict_from_redis_cache(job_id: str) -> Data:
    data_dict = store_data_job.load_data_redis(job_id)
    if data_dict is None:
        return None
    data = Data(**data_dict)
    __predict(data)
    return data


def _test(data: Data = Data()) -> Dict[str, int]:
    data.data = data.test_data
    __predict(data)
    return {'prediction': data.prediction}


def _predict(data: Data,
             background_tasks: BackgroundTasks) -> Dict[str, int]:
    __predict(data)
    _save_data_job(data, background_tasks)
    return {'prediction': data.prediction}


async def _predict_async_post(
        data: Data,
        background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = _save_data_job(data, background_tasks)
    return {'job_id': job_id}


def _predict_async_get(job_id: str) -> Dict[str, int]:
    result = {job_id: {'prediction': []}}
    if _PlatformConfigurations().platform == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        data_dict = store_data_job.load_data_redis(job_id)
        result[job_id]['prediction'] = data_dict['prediction']
        return result

    elif _PlatformConfigurations().platform == PLATFORM_ENUM.KUBERNETES.value:
        pass

    else:
        pass
