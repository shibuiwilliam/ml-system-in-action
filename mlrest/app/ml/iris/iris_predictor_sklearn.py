from typing import List
import joblib
import numpy as np

from app.ml.base_predictor import BaseData, BaseDataExtension, BasePredictor
import logging


logger = logging.getLogger(__name__)


class IrisData(BaseData):
    test_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]


class IrisDataExtension(BaseDataExtension):
    pass


class IrisClassifier(BasePredictor):
    def __init__(self, model_path):
        self.model_path = model_path
        self.classifier = None
        self.load_model()

    def load_model(self):
        logger.info(f'run load model in {self.__class__.__name__}')
        self.classifier = joblib.load(self.model_path)
        logger.info(f'initialized {self.__class__.__name__}')

    def predict(self, iris_data: IrisData) -> np.ndarray:
        logger.info(f'run predict proba in {self.__class__.__name__}')
        if iris_data.np_data is None:
            raise ValueError()

        iris_data.output = self.classifier.predict_proba(iris_data.np_data)

        logger.info({
            'prediction': {
                'data': iris_data.np_data,
                'output': iris_data.output
            }
        })
        return iris_data.output
