import logging
import importlib

from src.app.configurations import _ModelConfigurations

runner = importlib.import_module(_ModelConfigurations().runner)


logger = logging.getLogger(__name__)
ActiveData = runner._Data
ActiveDataInterface = runner._DataInterface
ActiveDataConverter = runner._DataConverter
ActivePredictor = runner._Classifier


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

DataConverter.data_interface = DataInterface

active_predictor = Predictor(_ModelConfigurations().model_runners)
