from abc import ABCMeta, abstractmethod
from pydantic import BaseModel, Extra
from typing import List, Any, Sequence, Union
import numpy as np

from app.ml.extract_np_type import type_name_to_np_type


class BaseData(BaseModel):
    data: Union[List[float], List[List[float]]] = None
    input_shape: Sequence[int] = None
    input_type: str = None
    prediction: Union[List[float], List[List[float]], float] = None
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

    def convert_input_data_to_np(self) -> np.ndarray:
        np_data = np.array(self.data_object.data).astype(self._input_type).reshape(self.data_object.input_shape)
        return np_data

    def reshape_output(self, output: np.ndarray) -> np.ndarray:
        np_data = output.astype(self._output_type).reshape(self.data_object.output_shape)
        return np_data


class BasePredictor(metaclass=ABCMeta):
    @ abstractmethod
    def load_model(self):
        raise NotImplementedError()

    @ abstractmethod
    def predict(self, data) -> Any:
        raise NotImplementedError()
