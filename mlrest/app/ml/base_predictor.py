from abc import ABCMeta, abstractmethod
from pydantic import BaseModel, Extra
from typing import List, Any, Sequence, Union
import numpy as np

from app.ml.extract_np_type import type_name_to_np_type


class BaseData(BaseModel):
    data: Union[List[float], List[List[float]]] = None
    np_data: np.ndarray = None
    input_shape: Sequence[int] = None
    input_type: str = None
    prediction: Union[List[float], List[List[float]], float] = None
    output: np.ndarray = None
    output_shape: Sequence[int] = None
    output_type: str = None

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow


class BaseDataExtension(metaclass=ABCMeta):
    def __init__(self, data_object: BaseData):
        self.data_object = data_object
        self._input_type = type_name_to_np_type(self.data_object.input_type)
        self._output_type = type_name_to_np_type(self.data_object.output_type)

    def convert_input_data_to_np_data(self):
        self.data_object.np_data = np.array(self.data_object.data)
        self._reshape_input()
        self._astype_input()

    def _reshape_input(self):
        self.data_object.np_data = self.data_object.np_data.reshape(self.data_object.input_shape)

    def _astype_input(self):
        self.data_object.np_data = self.data_object.np_data.astype(self._input_type)

    def convert_output_to_np(self):
        self._reshape_output()
        self._astype_output()

    def _reshape_output(self):
        self.data_object.output = self.data_object.output.reshape(self.data_object.output_shape)

    def _astype_output(self):
        self.data_object.output = self.data_object.output.astype(self._output_type)


class BasePredictor(metaclass=ABCMeta):
    @ abstractmethod
    def load_model(self):
        raise NotImplementedError()

    @ abstractmethod
    def predict_proba(self, data) -> Any:
        raise NotImplementedError()
