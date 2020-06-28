from typing import Dict
from fastapi import BackgroundTasks
from sklearn import datasets
import numpy as np
import uuid
import os
import json
import logging

from jobs import save_data_job, predict_job
from ml.base_predictor import Data, classifier
from constants import CONSTANTS, PLATFORM_ENUM
from middleware import redis


logger = logging.getLogger(__name__)

PLATFORM = os.getenv('PLATFORM', PLATFORM_ENUM.DOCKER.value)

sample_data = Data(
    np_data=datasets.load_iris().data[0].reshape((1, -1))[0])


def _save_data_job(data: Data,
                   background_tasks: BackgroundTasks) -> str:
    if PLATFORM == PLATFORM_ENUM.DOCKER.value:
        num_files = str(len(os.listdir(CONSTANTS.DATA_FILE_DIRECTORY)))
    elif PLATFORM == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        incr = redis.redis_connector.get(CONSTANTS.REDIS_INCREMENTS)
        num_files = 0 if incr is None else incr
    elif PLATFORM == PLATFORM_ENUM.KUBERNETES.value:
        pass
    else:
        pass

    job_id = f'{str(uuid.uuid4())}_{num_files}'
    _proba = data.prediction_proba.tolist(
    ) if data.prediction_proba is not None else [-0.1]
    data = {
        'prediction': data.prediction,
        'prediction_proba': _proba,
        'data': data.data,
    }

    if PLATFORM == PLATFORM_ENUM.DOCKER.value:
        task = save_data_job.SaveDataFileJob(
            job_id=job_id,
            directory=CONSTANTS.DATA_FILE_DIRECTORY,
            data=data)
    elif PLATFORM == PLATFORM_ENUM.DOCKER_COMPOSE.value:
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
    if PLATFORM == PLATFORM_ENUM.DOCKER.value:
        task = predict_job.PredictFromFileJob(
            job_id=job_id,
            directory=CONSTANTS.DATA_FILE_DIRECTORY,
            predictor=classifier
        )
    elif PLATFORM == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        task = predict_job.PredictFromRedisJob(
            job_id=job_id,
            predictor=classifier
        )
    elif PLATFORM == PLATFORM_ENUM.KUBERNETES.value:
        pass
    else:
        pass

    background_tasks.add_task(task)
    return job_id


def test() -> Dict[str, int]:
    _proba = classifier.predict_proba(sample_data)
    sample_data.prediction = int(np.argmax(_proba[0]))
    return {'prediction': sample_data.prediction}


def predict(data: Data,
            background_tasks: BackgroundTasks) -> Dict[str, int]:
    _proba = classifier.predict_proba(data)
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
    if PLATFORM == PLATFORM_ENUM.DOCKER.value:
        file_path = os.path.join(
            CONSTANTS.DATA_FILE_DIRECTORY, job_id + '.json')
        if not os.path.exists(file_path):
            return {job_id: CONSTANTS.PREDICTION_DEFAULT}
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
        return {
            job_id: data_dict.get('prediction', CONSTANTS.PREDICTION_DEFAULT)
        }

    elif PLATFORM == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        data_dict = redis.redis_connector.hgetall(job_id)
        if data_dict is None:
            return {job_id: CONSTANTS.PREDICTION_DEFAULT}
        return {job_id: int(data_dict.get(
                'prediction', CONSTANTS.PREDICTION_DEFAULT))
                }
