from typing import Dict
from fastapi import BackgroundTasks
from sklearn import datasets
import numpy as np
import uuid
import os
import json
import logging

from api.jobs import save_data_job, predict_job
from ml.iris.iris_predictor import IrisClassifier, IrisData
from constants import CONSTANTS

logger = logging.getLogger(__name__)

default_iris_model = CONSTANTS.IRIS_MODEL
iris_model = os.getenv("IRIS_MODEL", CONSTANTS.IRIS_MODEL)
logger.info(f'active model name: {iris_model}')

iris_classifier = IrisClassifier(iris_model)

sample_data = IrisData(
    np_data=datasets.load_iris().data[0].reshape((1, -1))[0])


def _save_data_job(iris_data: IrisData,
                   background_tasks: BackgroundTasks) -> str:
    num_files = str(len(os.listdir(CONSTANTS.IRIS_DATA_DIRECTORY)))
    job_id = f'{str(uuid.uuid4())}_{num_files}'
    _proba = iris_data.prediction_proba.tolist(
    ) if iris_data.prediction_proba is not None else None
    data = {
        "prediction": iris_data.prediction,
        "prediction_proba": _proba,
        "data": iris_data.data,
    }
    task = save_data_job.SaveDataJob(
        job_id=job_id,
        directory=CONSTANTS.IRIS_DATA_DIRECTORY,
        data=data)
    background_tasks.add_task(task)
    return job_id


def _predict_job(job_id: str,
                 iris_data: IrisData,
                 background_tasks: BackgroundTasks) -> str:
    file_path = os.path.join(CONSTANTS.IRIS_DATA_DIRECTORY, f'{job_id}.json')
    task = predict_job.PredictJob(
        job_id=job_id,
        file_path=file_path,
        predictor=iris_classifier
    )
    background_tasks.add_task(task)
    return job_id


def test() -> Dict[str, int]:
    iris_classifier.predict(sample_data)
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
    file_path = job_id + '.json'
    if file_path not in os.listdir(CONSTANTS.IRIS_DATA_DIRECTORY):
        return {job_id: CONSTANTS.PREDICTION_DEFAULT}
    with open(os.path.join(CONSTANTS.IRIS_DATA_DIRECTORY, file_path), 'r') as f:
        iris_dict = json.load(f)
    if iris_dict.get('prediction', CONSTANTS.PREDICTION_DEFAULT) == CONSTANTS.PREDICTION_DEFAULT:
        return {job_id: CONSTANTS.PREDICTION_DEFAULT}
    return {job_id: iris_dict['prediction']}
