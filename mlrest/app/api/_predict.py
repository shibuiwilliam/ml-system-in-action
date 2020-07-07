from typing import Dict
from fastapi import BackgroundTasks
import numpy as np
import uuid
import os
import json
import logging

from app.jobs import store_data_job, predict_job
from app.ml.active_predictor import Data, DataExtension, predictor
from app.constants import CONSTANTS, PLATFORM_ENUM
from app.middleware import redis


logger = logging.getLogger(__name__)

PLATFORM = os.getenv('PLATFORM', PLATFORM_ENUM.DOCKER_COMPOSE.value)


def _save_data_job(data: Data,
                   background_tasks: BackgroundTasks) -> str:
    if PLATFORM == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        incr = redis.redis_connector.get(CONSTANTS.REDIS_INCREMENTS)
        num_files = 0 if incr is None else incr
        job_id = f'{str(uuid.uuid4())}_{num_files}'
        data_dict = {}
        for k, v in data.__dict__.items():
            data_dict[k] = v.tolist() if isinstance(v, np.ndarray) else v
        task = store_data_job.SaveDataRedisJob(
            job_id=job_id,
            data=data_dict)

    elif PLATFORM == PLATFORM_ENUM.KUBERNETES.value:
        pass
    else:
        pass
    background_tasks.add_task(task)
    return job_id


def _predict_job(job_id: str,
                 background_tasks: BackgroundTasks) -> str:
    if PLATFORM == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        task = predict_job.PredictFromRedisJob(
            job_id=job_id,
            predictor=predictor,
            baseData=Data,
            baseDataExtentions=DataExtension
        )
    elif PLATFORM == PLATFORM_ENUM.KUBERNETES.value:
        pass
    else:
        pass

    background_tasks.add_task(task)
    return job_id


def __predict(data: Data):
    data_extension = DataExtension(data)
    data_extension.convert_input_data_to_np_data()
    data.output = predictor.predict(data)
    data_extension.convert_output_to_np()
    data.prediction = data.output.tolist()


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
    job_id = _predict_job(job_id, background_tasks)
    return {'job_id': job_id}


def _predict_async_get(job_id: str) -> Dict[str, int]:
    result = {job_id: {'prediction': []}}
    if PLATFORM == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        data_dict = store_data_job.load_data_redis(job_id)
        result[job_id]['prediction'] = data_dict['output']
        return result

    elif PLATFORM == PLATFORM_ENUM.KUBERNETES.value:
        pass

    else:
        pass
