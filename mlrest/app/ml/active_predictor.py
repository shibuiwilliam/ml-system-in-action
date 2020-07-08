from typing import Sequence
import logging

from app.constants import PREDICTION_RUNTIME
from app.configurations import _Configurations

if _Configurations().prediction_runtime == PREDICTION_RUNTIME.ONNX_RUNTIME.value:
    from app.ml.iris.iris_predictor_onnx import IrisClassifier, IrisData, IrisDataExtension
elif _Configurations().prediction_runtime == PREDICTION_RUNTIME.SKLEARN.value:
    from app.ml.iris.iris_predictor_sklearn import IrisClassifier, IrisData, IrisDataExtension


logger = logging.getLogger(__name__)
ActiveData = IrisData
ActiveDataExtension = IrisDataExtension
ActivePredictor = IrisClassifier


class Data(ActiveData):
    input_shape: Sequence[int] = _Configurations().io['input_shape']
    input_type: str = _Configurations().io['input_type']
    output_shape: Sequence[int] = _Configurations().io['output_shape']
    output_type: str = _Configurations().io['output_type']


class DataExtension(ActiveDataExtension):
    pass


class Predictor(ActivePredictor):
    pass


active_predictor = Predictor(_Configurations().model_filepath)
