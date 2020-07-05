from typing import Sequence
import logging

from app.ml.iris.iris_predictor import IrisClassifier, IrisData, IrisDataExtension
from app.configurations import _Configurations


logger = logging.getLogger(__name__)
ActiveData = IrisData
ActiveDataExtension = IrisDataExtension
ActivePredictor = IrisClassifier


class Data(ActiveData):
    input_shape: Sequence[int] = _Configurations().io_interface['input_shape']
    input_type: str = _Configurations().io_interface['input_type']
    output_shape: Sequence[int] = _Configurations().io_interface['output_shape']
    output_type: str = _Configurations().io_interface['output_type']


class DataExtension(ActiveDataExtension):
    pass


class Predictor(ActivePredictor):
    pass


predictor = Predictor(_Configurations().model_filepath)
