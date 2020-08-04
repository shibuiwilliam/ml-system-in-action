from typing import List, Any
import numpy as np
import onnx
import caffe2.python.onnx.backend as backend
from PIL import Image
from collections import OrderedDict
import joblib
import os

from src.app.configurations import _ModelConfigurations
from src.app.constants import MODEL_RUNTIME
from src.app.ml.base_predictor import BaseData, BaseDataInterface, BaseDataConverter, BasePredictor
from src.app.ml.transformers import PytorchImagePreprocessTransformer, SoftmaxTransformer
from src.app.ml.save_helper import load_labels

import logging


logger = logging.getLogger(__name__)

LABELS = load_labels(_ModelConfigurations().options['label_filepath'])


class _Data(BaseData):
    image_data: Any = None
    test_data: str = os.path.join('./src/app/ml/data', 'good_cat.jpg')
    labels: List[str] = LABELS


class _DataInterface(BaseDataInterface):
    pass


class _DataConverter(BaseDataConverter):
    pass


class _Classifier(BasePredictor):
    def __init__(self, model_runners):
        self.model_runners = model_runners
        self.classifiers = OrderedDict()
        self.load_model()

    def load_model(self):
        logger.info(f'run load model in {self.__class__.__name__}')
        for m in self.model_runners:
            logger.info(f'{m.items()}')
            for k, v in m.items():
                if v == MODEL_RUNTIME.SKLEARN.value:
                    self.classifiers[k] = {
                        'runner': v,
                        'predictor': joblib.load(k)
                    }
                elif v == MODEL_RUNTIME.PYTORCH_CAFFE2.value:
                    model = onnx.load(k)
                    onnx.checker.check_model(model)
                    self.classifiers[k] = {
                        'runner': v,
                        'predictor': backend.prepare(model, device='CPU')
                    }
                else:
                    pass
        logger.info(f'initialized {self.__class__.__name__}')

    def predict(self, input_data: Image) -> np.ndarray:
        logger.info(f'run predict proba in {self.__class__.__name__}')
        _prediction = input_data
        for k, v in self.classifiers.items():
            if v['runner'] == MODEL_RUNTIME.SKLEARN.value:
                _prediction = np.array(v['predictor'].transform(_prediction))
            elif v['runner'] == MODEL_RUNTIME.PYTORCH_CAFFE2.value:
                _prediction = np.array(v['predictor'].run(_prediction.astype(np.float32)))
        output = _prediction
        return output
