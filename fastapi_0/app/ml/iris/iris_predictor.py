from typing import List, Tuple, Dict
import joblib
import numpy as np
from pydantic import BaseModel, Extra

from constants import CONSTANTS
from ml import load_model
from ml.abstract_predictor import Predictor
import logging


logger = logging.getLogger(__name__)


class IrisData(BaseModel):
    data: List[float] = None
    np_data: np.ndarray = None
    input_shape: Tuple[int] = (1, 4)
    prediction: int = CONSTANTS.PREDICTION_DEFAULT
    prediction_proba: np.ndarray = None

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

    def predict_proba(self, iris_data: IrisData) -> np.ndarray:
        logger.info(f'predict proba {self.__class__.__name__}')
        if iris_data.np_data is None:
            iris_data.np_data = np.array(iris_data.data).astype(np.float64)
        if iris_data.np_data.shape != iris_data.input_shape:
            iris_data.np_data = iris_data.np_data.reshape(
                iris_data.input_shape)
        iris_data.prediction_proba = self.classifier.predict_proba(
            iris_data.np_data)
        logger.info({
            'prediction': {
                'data': iris_data.np_data,
                'prediction_proba': iris_data.prediction_proba
            }
        })
        return iris_data.prediction_proba

    def predict_proba_from_dict(self, iris_dict: Dict) -> np.ndarray:
        iris_data = IrisData()
        iris_data.data = iris_dict.get('data', None)
        iris_data.np_data = iris_dict.get('np_data', None)
        return self.predict_proba(iris_data)
