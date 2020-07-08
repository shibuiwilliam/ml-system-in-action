from typing import List
import numpy as np
import onnxruntime as rt

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
        self.input_name = None
        self.load_model()

    def load_model(self):
        logger.info(f'run load model in {self.__class__.__name__}')
        self.classifier = rt.InferenceSession(self.model_path)
        self.input_name = self.classifier.get_inputs()[0].name
        logger.info(f'initialized {self.__class__.__name__}')

    def predict(self, iris_data: IrisData) -> np.ndarray:
        logger.info(f'run predict proba in {self.__class__.__name__}')
        if iris_data.np_data is None:
            raise ValueError()

        _prediction = self.classifier.run(
            None,
            {self.input_name: iris_data.np_data.astype(np.float32)}
        )
        iris_data.output = np.array(list(_prediction[1][0].values()))

        logger.info({
            'prediction': {
                'data': iris_data.np_data,
                'output': iris_data.output
            }
        })
        return iris_data.output
