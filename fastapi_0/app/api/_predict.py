from typing import Dict
from fastapi import BackgroundTasks
from sklearn import datasets
import numpy as np
import uuid
import os
import json
import logging

from jobs import save_data_job, predict_job
from ml.iris.iris_predictor import IrisClassifier, IrisData
from constants import CONSTANTS
from middleware import redis


logger = logging.getLogger(__name__)

PLATFORM = os.getenv('PLATFORM', CONSTANTS.PLATFORM_DOCKER)

default_iris_model = CONSTANTS.IRIS_MODEL
iris_model = os.getenv('IRIS_MODEL', CONSTANTS.IRIS_MODEL)
logger.info(f'active model name: {iris_model}')

iris_classifier = IrisClassifier(iris_model)

sample_data = IrisData(
    np_data=datasets.load_iris().data[0].reshape((1, -1))[0])


def _save_data_job(iris_data: IrisData,
                   background_tasks: BackgroundTasks) -> str:
    if PLATFORM == CONSTANTS.PLATFORM_DOCKER:
        num_files = str(len(os.listdir(CONSTANTS.DATA_FILE_DIRECTORY)))
    elif PLATFORM == CONSTANTS.PLATFORM_DOCKER_COMPOSE:
        incr = redis.redis_connector.get(CONSTANTS.REDIS_INCREMENTS)
        num_files = 0 if incr is None else incr

    job_id = f'{str(uuid.uuid4())}_{num_files}'
    _proba = iris_data.prediction_proba.tolist(
    ) if iris_data.prediction_proba is not None else [-0.1]
    data = {
        'prediction': iris_data.prediction,
        'prediction_proba': _proba,
        'data': iris_data.data,
    }

    if PLATFORM == CONSTANTS.PLATFORM_DOCKER:
        task = save_data_job.SaveDataFileJob(
            job_id=job_id,
            directory=CONSTANTS.DATA_FILE_DIRECTORY,
            data=data)
    elif PLATFORM == CONSTANTS.PLATFORM_DOCKER_COMPOSE:
        task = save_data_job.SaveDataRedisJob(
            job_id=job_id,
            data=data)
    background_tasks.add_task(task)
    return job_id


def _predict_job(job_id: str,
                 iris_data: IrisData,
                 background_tasks: BackgroundTasks) -> str:
    if PLATFORM == CONSTANTS.PLATFORM_DOCKER:
        task = predict_job.PredictFromFileJob(
            job_id=job_id,
            directory=CONSTANTS.DATA_FILE_DIRECTORY,
            predictor=iris_classifier
        )
    if PLATFORM == CONSTANTS.PLATFORM_DOCKER_COMPOSE:
        task = predict_job.PredictFromRedisJob(
            job_id=job_id,
            predictor=iris_classifier
        )
    background_tasks.add_task(task)
    return job_id


def test() -> Dict[str, int]:
    _proba = iris_classifier.predict_proba(sample_data)
    sample_data.prediction = int(np.argmax(_proba[0]))
    return {'prediction': sample_data.prediction}


def predict(iris_data: IrisData,
            background_tasks: BackgroundTasks) -> Dict[str, int]:
    _proba = iris_classifier.predict_proba(iris_data)
    iris_data.prediction = int(np.argmax(_proba[0]))
    _save_data_job(iris_data, background_tasks)
    return {'prediction': iris_data.prediction}


async def predict_async_post(
        iris_data: IrisData,
        background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = _save_data_job(iris_data, background_tasks)
    job_id = _predict_job(job_id, iris_data, background_tasks)
    return {'job_id': job_id}


def predict_async_get(job_id: str) -> Dict[str, int]:
    if PLATFORM == CONSTANTS.PLATFORM_DOCKER:
        file_path = os.path.join(CONSTANTS.DATA_FILE_DIRECTORY, job_id + '.json')
        if not os.path.exists(file_path):
            return {job_id: CONSTANTS.PREDICTION_DEFAULT}
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
        return {job_id: data_dict.get('prediction', CONSTANTS.PREDICTION_DEFAULT)}

    elif PLATFORM == CONSTANTS.PLATFORM_DOCKER_COMPOSE:
        data_dict = redis.redis_connector.hgetall(job_id)
        if data_dict is None:
            return {job_id: CONSTANTS.PREDICTION_DEFAULT}
        return {job_id: data_dict.get('prediction', CONSTANTS.PREDICTION_DEFAULT)}
