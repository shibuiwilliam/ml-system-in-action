import joblib
from sklearn import datasets

from ml import load_model
from ml.abstract_predictor import Predictor
import logging

logger = logging.getLogger(__name__) 


class IrisClassifier(Predictor):
    def __init__(self, model_filename):
        logger.info(f'Initialized {self.__class__.__name__}')
        self.model_filename = model_filename
        self.classifier = self.load_model()

    def load_model(self):
        logger.info(f'Load model {self.__class__.__name__}')
        model_filename = load_model.get_model_file(self.model_filename)
        return joblib.load(model_filename) if model_filename else None

    def predict(self, data):
        logger.info(f'Predict {self.__class__.__name__}')
        prediction = self.classifier.predict(data)
        logger.info({'prediction': {'input': data, 'prediction': prediction}})
        return prediction
