from abc import ABCMeta, abstractmethod
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np


class BaseData(BaseModel, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @property
    def data_shape(self):
        return self._data_shape

    @data_shape.setter
    def data_shape(self, value):
        if not isinstance(value, Tuple[int]):
            raise TypeError(value)
        self._data_shape = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, List[float]):
            raise TypeError(value)
        self._data = value

    @property
    def np_data(self):
        return self._np_data

    @np_data.setter
    def np_data(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(value)
        self._np_data = value

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, value):
        if not isinstance(value, int):
            raise TypeError(value)
        self._prediction = value

    @property
    def proba_shape(self):
        return self._proba_shape

    @proba_shape.setter
    def proba_shape(self, value):
        if not isinstance(value, Tuple[int]):
            raise TypeError(value)
        self._proba_shape = value

    @property
    def prediction_proba(self):
        return self._prediction_proba

    @prediction_proba.setter
    def prediction_proba(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(value)
        self._prediction_proba = value


class BasePredictor(metaclass=ABCMeta):
    @abstractmethod
    def load_model(self):
        raise NotImplementedError()

    @abstractmethod
    def predict_proba(self, data):
        raise NotImplementedError()

    @abstractmethod
    def predict_proba_from_dict(self, data):
        raise NotImplementedError()
