from abc import ABCMeta, abstractmethod


class Predictor(metaclass=ABCMeta):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict_proba(self, data):
        pass

    @abstractmethod
    def predict_proba_from_dict(self, data):
        pass
