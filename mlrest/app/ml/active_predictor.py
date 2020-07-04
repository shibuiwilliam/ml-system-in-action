from app.ml.iris.iris_predictor import IrisClassifier, IrisData, IrisDataExtension
from app.configurations import configurations
import logging


logger = logging.getLogger(__name__)
ActiveData = IrisData
ActiveDataExtension = IrisDataExtension
ActivePredictor = IrisClassifier


class Data(ActiveData):
    pass


class DataExtension(ActiveDataExtension):
    pass


class Predictor(ActivePredictor):
    pass


predictor = Predictor(configurations.model_filename,
                      configurations.fallback_model_filname)
