from typing import Dict, List
from fastapi import BackgroundTasks
from sklearn import datasets
import numpy as np
import uuid
import os
import json
import logging
from pydantic import BaseModel

from ml.iris.iris_predictor import IrisClassifier, IrisData
from constants import CONSTANTS

logger = logging.getLogger(__name__)

default_iris_model = CONSTANTS.IRIS_MODEL
iris_model = os.getenv("IRIS_MODEL", CONSTANTS.IRIS_MODEL)
logger.info(f'active model name: {iris_model}')

iris_classifier = IrisClassifier(iris_model)

sample_data = IrisData(
    data=datasets.load_iris().data[0].reshape((1, -1))[0].tolist())


class SaveDataJob(BaseModel):
    job_id: str
    directory: str
    data: List[float]
    is_completed: bool = False

    def __call__(self):
        jobs[self.job_id] = self
        filePath = os.path.join(self.directory, f'{self.job_id}.json')
        with open(filePath, 'w') as f:
            json.dump(self.data, f)
        self.is_completed = True
        logger.info(f'completed save data: {self.job_id}')


jobs: Dict[str, SaveDataJob] = {}


def test() -> Dict[str, int]:
    y = iris_classifier.predict(sample_data)
    return {'prediction': y}


def predict(iris_data: IrisData,
            background_tasks: BackgroundTasks) -> Dict[str, int]:
    num_files = str(len(os.listdir(CONSTANTS.IRIS_DATA_DIRECTORY)))
    job_id = f'{str(uuid.uuid4())}_{num_files}'
    task = SaveDataJob(job_id=job_id,
                       directory=CONSTANTS.IRIS_DATA_DIRECTORY,
                       data=iris_data.data)
    background_tasks.add_task(task)
    y = iris_classifier.predict(iris_data)
    return {'prediction': y}


async def predict_async(iris_data: IrisData,
                        background_tasks: BackgroundTasks) -> Dict[str, str]:
    num_files = str(len(os.listdir(CONSTANTS.IRIS_DATA_DIRECTORY)))
    job_id = f'{str(uuid.uuid4())}_{num_files}'
    task = SaveDataJob(job_id=job_id,
                       directory=CONSTANTS.IRIS_DATA_DIRECTORY,
                       data=iris_data.data)
    background_tasks.add_task(task)
    return {'job_id': job_id}
