import logging

from app.constants import PREDICTION_RUNTIME
from app.configurations import _ModelConfigurations

if _ModelConfigurations().prediction_runtime == PREDICTION_RUNTIME.ONNX_RUNTIME.value:
    from app.ml.iris.iris_predictor_onnx import IrisClassifier, IrisDataInterface, IrisData, IrisDataConverter
elif _ModelConfigurations().prediction_runtime == PREDICTION_RUNTIME.SKLEARN.value:
    from app.ml.iris.iris_predictor_sklearn import IrisClassifier, IrisDataInterface, IrisData, IrisDataConverter


logger = logging.getLogger(__name__)
ActiveData = IrisData
ActiveDataInterface = IrisDataInterface
ActiveDataConverter = IrisDataConverter
ActivePredictor = IrisClassifier


class Data(ActiveData):
    pass


class DataInterface(ActiveDataInterface):
    pass


class DataConverter(ActiveDataConverter):
    pass


class Predictor(ActivePredictor):
    pass


DataInterface.input_shape = _ModelConfigurations().io['input_shape']
DataInterface.input_type = _ModelConfigurations().io['input_type']
DataInterface.output_shape = _ModelConfigurations().io['output_shape']
DataInterface.output_type = _ModelConfigurations().io['output_type']
DataInterface.data_type = _ModelConfigurations().io['data_type']

DataConverter.meta_data = DataInterface

active_predictor = Predictor(_ModelConfigurations().model_filepath)
