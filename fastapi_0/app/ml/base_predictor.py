from ml.iris.iris_predictor import IrisClassifier, IrisData
from configurations import Configurations
import logging


logger = logging.getLogger(__name__)


class Data(IrisData):
    pass


class Classifier(IrisClassifier):
    pass


classifier = Classifier(Configurations.model_filename,
                        Configurations.fallback_model_filname)
