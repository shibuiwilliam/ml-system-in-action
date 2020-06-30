from typing import List, Dict, Tuple
import joblib
import numpy as np

from app.constants import CONSTANTS
from app.ml import load_model
from app.ml.abstract_predictor import BaseData, BasePredictor
import logging


logger = logging.getLogger(__name__)


class IrisData(BaseData):
    test_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]
    data_shape: Tuple[int] = (1, 4)
    proba_shape: Tuple[int] = (1, 3)


class IrisClassifier(BasePredictor):
    def __init__(self,
                 model_filename,
                 fallback_model_filname):
        self.model_filename = model_filename
        self.fallback_model_filname = fallback_model_filname
        self.active_model = None
        self.classifier = None
        self.load_model()

    def load_model(self):
        logger.info(f'run load model in {self.__class__.__name__}')
        self.active_model = load_model.get_model_file(
            self.model_filename,
            self.fallback_model_filname)
        self.classifier = joblib.load(self.active_model)
        logger.info(
            f'initialized {self.__class__.__name__} for {self.active_model}')

    def predict_proba(self, iris_data: IrisData) -> np.ndarray:
        logger.info(f'run predict proba in {self.__class__.__name__}')

        if iris_data.np_data is None:
            if iris_data.data is None:
                raise ValueError()
            iris_data.np_data = np.array(iris_data.data).astype(np.float64)
        if iris_data.np_data.shape != iris_data.data_shape:
            iris_data.np_data = iris_data.np_data.reshape(
                iris_data.data_shape)

        iris_data.prediction_proba = self.classifier.predict_proba(
            iris_data.np_data)
        if iris_data.prediction_proba.shape != iris_data.proba_shape:
            iris_data.prediction_proba = iris_data.prediction_proba.reshape(
                iris_data.proba_shape)

        logger.info({
            'prediction': {
                'data': iris_data.np_data,
                'prediction_proba': iris_data.prediction_proba
            }
        })
        return iris_data.prediction_proba

    def predict_proba_from_dict(self, iris_dict: Dict) -> np.ndarray:
        logger.info(
            f'run predict proba from dict in {self.__class__.__name__}')

        iris_data = IrisData()
        iris_data.data = iris_dict.get('data', None)
        iris_data.np_data = iris_dict.get('np_data', None)
        return self.predict_proba(iris_data)
