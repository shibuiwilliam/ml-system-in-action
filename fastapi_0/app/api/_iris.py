from typing import Dict
import joblib
from sklearn import datasets

from ml import load_model
import logging

logger = logging.getLogger(__name__) 
IRIS_MODEL = "iris_svc.pkl"


class IrisClassifier(object):
    def __init__(self, model_filename=IRIS_MODEL):
        self.model_filename = model_filename
        self.classifier = self.get_model()

    def get_model(self):
        model_filename = load_model.get_model_file(self.model_filename)
        return joblib.load(model_filename) if model_filename else None

    def predict(self, data):
        return self.classifier.predict(data)


iris_classifier = IrisClassifier()
sample = datasets.load_iris().data[0].reshape((1, -1))


def test() -> Dict[str, int]:
    y = iris_classifier.predict(sample).tolist()
    logger.info({"input": sample, "prediction": y})
    return {"prediction": y[0]}

