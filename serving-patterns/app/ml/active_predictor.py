import logging

from app.constants import PREDICTION_RUNTIME
from app.configurations import _ModelConfigurations

if _ModelConfigurations().prediction_runtime == PREDICTION_RUNTIME.ONNX_RUNTIME.value:
    from app.ml.iris.iris_predictor_onnx import IrisClassifier, IrisMetadata, IrisData, IrisDataConverter
elif _ModelConfigurations().prediction_runtime == PREDICTION_RUNTIME.SKLEARN.value:
    from app.ml.iris.iris_predictor_sklearn import IrisClassifier, IrisMetadata, IrisData, IrisDataConverter


logger = logging.getLogger(__name__)
ActiveData = IrisData
ActiveMetadata = IrisMetadata
ActiveDataConverter = IrisDataConverter
ActivePredictor = IrisClassifier


class Data(ActiveData):
    pass


class MetaData(ActiveMetadata):
    pass


class DataConverter(ActiveDataConverter):
    pass


class Predictor(ActivePredictor):
    pass


MetaData.input_shape = _ModelConfigurations().io['input_shape']
MetaData.input_type = _ModelConfigurations().io['input_type']
MetaData.output_shape = _ModelConfigurations().io['output_shape']
MetaData.output_type = _ModelConfigurations().io['output_type']
DataConverter.meta_data = MetaData
active_predictor = Predictor(_ModelConfigurations().model_filepath)
