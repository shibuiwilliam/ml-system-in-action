from typing import List, Tuple
import joblib
import numpy as np
from pydantic import BaseModel, Extra

from constants import CONSTANTS
from ml import load_model
from ml.abstract_predictor import Predictor
import logging


logger = logging.getLogger(__name__)


class IrisData(BaseModel):
    data: List[float]
    np_data: np.ndarray = np.array([[0.0, 0.0, 0.0, 0.0]]).astype(np.float64)
    input_shape: Tuple[int] = (1, 4)
    prediction: int = -1
    prediction_proba: np.ndarray = np.array([[0.0, 0.0, 0.0, 0.0]]).astype(np.float64)

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow


class IrisClassifier(Predictor):
    def __init__(self,
                 model_filename,
                 fallback_model_filname=CONSTANTS.IRIS_MODEL):
        self.model_filename = model_filename
        self.fallback_model_filname = fallback_model_filname
        self.active_model = None
        self.classifier = None
        self.load_model()

    def load_model(self):
        logger.info(f'load model {self.__class__.__name__}')
        self.active_model = load_model.get_model_file(
            self.model_filename,
            self.fallback_model_filname)
        self.classifier = joblib.load(self.active_model)
        logger.info(
            f'initialized {self.__class__.__name__} for {self.active_model}')

    def predict(self, iris_data: IrisData) -> int:
        logger.info(f'predict {self.__class__.__name__}')
        iris_data.prediction_proba = self.predict_proba(iris_data)
        iris_data.prediction = np.argmax(iris_data.prediction_proba)
        return iris_data.prediction

    def predict_proba(self, iris_data: IrisData) -> np.ndarray:
        logger.info(f'predict proba {self.__class__.__name__}')
        iris_data.np_data = iris_data.data \
            if isinstance(iris_data.data, np.ndarray) \
                else np.array((iris_data.data)).astype(np.float64)
        iris_data.prediction_proba = self.classifier.predict_proba(
            iris_data.np_data)
        logger.info({
            'prediction': {
                'input': iris_data.np_data,
                'prediction_proba': iris_data.prediction_proba
            }
        })
        return iris_data.prediction_proba
