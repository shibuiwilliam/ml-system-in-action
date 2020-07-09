from typing import Sequence
import logging

from app.constants import PREDICTION_RUNTIME
from app.configurations import _ModelConfigurations

if _ModelConfigurations().prediction_runtime == PREDICTION_RUNTIME.ONNX_RUNTIME.value:
    from app.ml.iris.iris_predictor_onnx import IrisClassifier, IrisData, IrisDataExtension
elif _ModelConfigurations().prediction_runtime == PREDICTION_RUNTIME.SKLEARN.value:
    from app.ml.iris.iris_predictor_sklearn import IrisClassifier, IrisData, IrisDataExtension


logger = logging.getLogger(__name__)
ActiveData = IrisData
ActiveDataExtension = IrisDataExtension
ActivePredictor = IrisClassifier


class Data(ActiveData):
    input_shape: Sequence[int] = _ModelConfigurations().io['input_shape']
    input_type: str = _ModelConfigurations().io['input_type']
    output_shape: Sequence[int] = _ModelConfigurations().io['output_shape']
    output_type: str = _ModelConfigurations().io['output_type']


class DataExtension(ActiveDataExtension):
    pass


class Predictor(ActivePredictor):
    pass


active_predictor = Predictor(_ModelConfigurations().model_filepath)
