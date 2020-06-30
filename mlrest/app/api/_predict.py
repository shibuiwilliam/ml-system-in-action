from typing import Dict
from fastapi import BackgroundTasks
from sklearn import datasets
import numpy as np
import uuid
import os
import json
import logging

from app.jobs import save_data_job, predict_job
from app.ml.active_predictor import Data, predictor
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
        _proba = data.prediction_proba.tolist(
        ) if data.prediction_proba is not None else [-0.1]
        data = {
            'prediction': data.prediction,
            'prediction_proba': _proba,
            'data': data.data,
        }
        task = save_data_job.SaveDataRedisJob(
            job_id=job_id,
            data=data)

    elif PLATFORM == PLATFORM_ENUM.KUBERNETES.value:
        pass
    else:
        pass
    background_tasks.add_task(task)
    return job_id


def _predict_job(job_id: str,
                 data: Data,
                 background_tasks: BackgroundTasks) -> str:
    if PLATFORM == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        task = predict_job.PredictFromRedisJob(
            job_id=job_id,
            predictor=predictor
        )
    elif PLATFORM == PLATFORM_ENUM.KUBERNETES.value:
        pass
    else:
        pass

    background_tasks.add_task(task)
    return job_id


def test() -> Dict[str, int]:
    sample_data = Data()
    sample_data.data = sample_data.test_data
    _proba = predictor.predict_proba(sample_data)
    sample_data.prediction = int(np.argmax(_proba[0]))
    return {'prediction': sample_data.prediction}


def predict(data: Data,
            background_tasks: BackgroundTasks) -> Dict[str, int]:
    _proba = predictor.predict_proba(data)
    data.prediction = int(np.argmax(_proba[0]))
    _save_data_job(data, background_tasks)
    return {'prediction': data.prediction}


async def predict_async_post(
        data: Data,
        background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = _save_data_job(data, background_tasks)
    job_id = _predict_job(job_id, data, background_tasks)
    return {'job_id': job_id}


def predict_async_get(job_id: str) -> Dict[str, int]:
    result = {job_id: {'prediction': CONSTANTS.PREDICTION_DEFAULT}}
    if PLATFORM == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        data_dict = redis.redis_connector.hgetall(job_id)
        result[job_id]['prediction'] = int(
            data_dict.get(
                'prediction',
                CONSTANTS.PREDICTION_DEFAULT)) if data_dict is not None else CONSTANTS.PREDICTION_DEFAULT
        return result

    elif PLATFORM == PLATFORM_ENUM.KUBERNETES.value:
        pass

    else:
        pass
