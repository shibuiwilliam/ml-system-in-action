from typing import List, Any
import numpy as np

from PIL import Image
from collections import OrderedDict
import joblib
import os

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from app.configurations import _ModelConfigurations
from app.constants import MODEL_RUNTIME
from app.ml.base_predictor import BaseData, BaseDataInterface, BaseDataConverter, BasePredictor
from app.ml.transformers import ONNXImagePreprocessTransformer, SoftmaxTransformer
from app.ml.save_helper import load_labels

import logging


logger = logging.getLogger(__name__)


LABELS = load_labels(_ModelConfigurations().options['label_filepath'])


class _Data(BaseData):
    image_data: Any = None
    test_data: str = os.path.join('./app/ml/data', 'good_cat.jpg')
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
                else:
                    pass
        logger.info(f'initialized {self.__class__.__name__}')

    def predict(self, input_data: Image) -> np.ndarray:
        logger.info(f'run predict proba in {self.__class__.__name__}')
        _prediction = input_data
        for k, v in self.classifiers.items():
            if v['runner'] == MODEL_RUNTIME.SKLEARN.value:
                _prediction = np.array(v['predictor'].transform(_prediction))
            else:
                channel = grpc.insecure_channel('prep_pred_tfs:8501')
                stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
                request = predict_pb2.PredictRequest()
                request.model_spec.name = 'inceptionv3'
                request.model_spec.signature_name = 'serving_default'
                request.inputs['input_1'].CopyFrom(
                    tf.make_tensor_proto(_prediction, shape=[1, 299, 299, 3]))
                result = stub.Predict(request, 10.0)
                logging.info(f'result: {result}')
        output = _prediction
        return output
