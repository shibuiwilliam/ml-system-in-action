from abc import ABCMeta, abstractmethod
from pydantic import BaseModel, Extra
from typing import List, Tuple, Any
import numpy as np


class BaseData(BaseModel):
    data: List[float] = None
    np_data: np.ndarray = None
    data_shape: Tuple[int] = None
    np_datatype: str = None
    prediction: int = None
    proba_shape: Tuple[int] = None
    prediction_proba: np.ndarray = None

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow


class BaseDataExtension(object):
    def __init__(self, data_object: BaseData):
        self.data_object = data_object

    def convert_data_to_np_data(self):
        self.data_object.np_data = self._astype(self._reshape((np.array(self.data_object.data))))

    def _reshape(self, np_data: np.ndarray) -> np.ndarray:
        if self.data_object.data_shape is None:
            return np_data
        else:
            return np_data.reshape(self.data_object.data_shape)

    def _astype(self, np_data: np.ndarray) -> np.ndarray:
        if self.data_object.np_datatype is None:
            return np_data
        else:
            if self.data_object.np_datatype == 'int':
                return np_data.astype(np.int)
            elif self.data_object.np_datatype == 'int8':
                return np_data.astype(np.int8)
            elif self.data_object.np_datatype == 'int16':
                return np_data.astype(np.int16)
            elif self.data_object.np_datatype == 'int32':
                return np_data.astype(np.int32)
            elif self.data_object.np_datatype == 'int64':
                return np_data.astype(np.int64)
            elif self.data_object.np_datatype == 'float':
                return np_data.astype(np.float)
            elif self.data_object.np_datatype == 'float16':
                return np_data.astype(np.float16)
            elif self.data_object.np_datatype == 'float32':
                return np_data.astype(np.float32)
            elif self.data_object.np_datatype == 'float64':
                return np_data.astype(np.float64)
            else:
                return np_data


class BasePredictor(metaclass=ABCMeta):
    @ abstractmethod
    def load_model(self):
        raise NotImplementedError()

    @ abstractmethod
    def predict_proba(self, data) -> Any:
        raise NotImplementedError()

    @ abstractmethod
    def predict_proba_from_dict(self, data) -> Any:
        raise NotImplementedError()
