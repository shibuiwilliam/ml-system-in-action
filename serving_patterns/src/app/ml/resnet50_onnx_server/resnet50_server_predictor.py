from typing import List, Any
import numpy as np
from PIL import Image
from collections import OrderedDict
import joblib
import requests
import json
import base64
import os
from onnx import numpy_helper
from google.protobuf.json_format import MessageToJson

from src.app.configurations import ModelConfigurations
from src.app.constants import MODEL_RUNTIME
from src.app.ml.base_predictor import BaseData, BaseDataInterface, BaseDataConverter, BasePredictor
from src.app.ml.transformers import PytorchImagePreprocessTransformer, SoftmaxTransformer
from src.app.ml.save_helper import load_labels

import logging


logger = logging.getLogger(__name__)

LABELS = load_labels(ModelConfigurations.options['label_filepath'])
ONNX_RUNTIME_SERVER_HTTP = os.getenv('ONNX_RUNTIME_SERVER_HTTP', 'prep_pred_onnx_resnet50:8001')
MODEL_NAME = os.getenv('MODEL_NAME', 'default')
VERSION = int(os.getenv('VERSION', 1))
TIMEOUT_SECOND = int(os.getenv('TIMEOUT_SECOND', 5.0))


class _Data(BaseData):
    image_data: Any = None
    test_data: str = os.path.join(
        './src/app/ml/resnet50_onnx/data',
        'good_cat.jpg')
    labels: List[str] = LABELS


class _DataInterface(BaseDataInterface):
    pass


class _DataConverter(BaseDataConverter):
    pass


class _Classifier(BasePredictor):
    def __init__(self, model_runners):
        self.model_runners = model_runners
        self.classifiers = OrderedDict()
        self.input_name = None
        self.output_name = None
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
                elif v == MODEL_RUNTIME.ONNX_RUNTIME_SERVER.value:
                    self.classifiers[k] = {
                        'runner': v,
                        'predictor': f'http://{ONNX_RUNTIME_SERVER_HTTP}/v1/models/{MODEL_NAME}/versions/{VERSION}:predict'
                    }
                else:
                    pass
        self.input_name = ModelConfigurations.options['input_name']
        self.output_name = ModelConfigurations.options['output_name']
        logger.info(f'initialized {self.__class__.__name__}')

    def predict(self, input_data: Image) -> np.ndarray:
        logger.info(f'run predict proba in {self.__class__.__name__}')
        _prediction = input_data
        for k, v in self.classifiers.items():
            if v['runner'] == MODEL_RUNTIME.SKLEARN.value:
                _prediction = np.array(v['predictor'].transform(_prediction))
            elif v['runner'] == MODEL_RUNTIME.ONNX_RUNTIME_SERVER.value:
                tensor_proto = numpy_helper.from_array(_prediction)
                json_str = MessageToJson(tensor_proto, use_integers_for_enums=True)
                data = json.loads(json_str)
                _input_data = {self.input_name: data}
                _output_filters = [self.output_name]
                payload = {'inputs': _input_data, 'outputFilter': _output_filters}
                response = requests.post(
                    v['predictor'],
                    json.dumps(payload),
                    headers={'Content-Type': 'application/json', 'Accept': 'application/json'}
                )
                raw_data = json.loads(response.text)['outputs'][self.output_name]['rawData']
                _prediction = np.frombuffer(base64.b64decode(raw_data), dtype=np.float32)
        output = _prediction
        return output
