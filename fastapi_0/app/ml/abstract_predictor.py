from abc import ABCMeta, abstractmethod


class Predictor(metaclass=ABCMeta):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, data):
        pass
