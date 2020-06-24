from typing import Dict
from sklearn import datasets
import numpy as np
import uuid
import os
import json

from ml.iris.iris_predictor import IrisClassifier, IrisData
from constants import CONSTANTS
import logging

logger = logging.getLogger(__name__)

iris_classifier = IrisClassifier(CONSTANTS.IRIS_MODEL)
sample = datasets.load_iris().data[0].reshape((1, -1))


def test() -> Dict[str, int]:
    y = iris_classifier.predict(sample).tolist()
    return {'prediction': y[0]}


def predict(iris_data: IrisData) -> Dict[str, int]:
    data = np.array(iris_data.data).reshape((1, -1))
    y = iris_classifier.predict(data).tolist()
    return {'prediction': y[0]}


async def predict_async(iris_data: IrisData) -> Dict[str, str]:
    num_files = str(len(os.listdir(CONSTANTS.DATA_DIRECTORY)))
    _id = f'{str(uuid.uuid4())}_{num_files}'
    filePath = os.path.join(CONSTANTS.DATA_DIRECTORY, f'{_id}.json')
    with open(filePath, 'w') as f:
        json.dump(iris_data.data, f)
    return {'id': _id}
