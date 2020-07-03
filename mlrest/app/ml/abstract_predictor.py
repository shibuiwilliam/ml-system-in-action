from abc import ABCMeta, abstractmethod
from pydantic import BaseModel, Extra
from typing import List, Tuple, Any
import numpy as np


class BaseData(BaseModel):
    data: List[float] = None
    np_data: np.ndarray = None
    data_shape: Tuple[int] = None
    prediction: int = None
    proba_shape: Tuple[int] = None
    prediction_proba: np.ndarray = None

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow


class BasePredictor(metaclass=ABCMeta):
    @abstractmethod
    def load_model(self):
        raise NotImplementedError()

    @abstractmethod
    def predict_proba(self, data) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def predict_proba_from_dict(self, data) -> Any:
        raise NotImplementedError()
