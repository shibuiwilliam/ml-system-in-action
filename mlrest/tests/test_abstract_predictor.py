import pytest
from typing import List, Tuple
import numpy as np

from app.ml.abstract_predictor import BaseData, BaseDataExtension, BasePredictor


f_data = [[0.1, 0.9, 1.1]]
i_data = [[1, 2, 3]]


@pytest.mark.parametrize(
    ('data', 'np_data', 'data_shape', 'np_datatype', 'prediction', 'proba_shape', 'prediction_proba'),
    [(f_data, np.array(f_data), (1, 3), 'float', 0, (1, 4), np.array([[0.1, 0.2, 0.3, 0.4]]))]
)
def test_BaseData(mocker, data, np_data, data_shape, np_datatype, prediction, proba_shape, prediction_proba):
    class MockData(BaseData):
        testf_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]

    mockf_data = MockData()
    mockf_data.data = data
    mockf_data.np_data = np_data
    mockf_data.np_datatype = np_datatype
    mockf_data.prediction = prediction
    mockf_data.prediction_proba = prediction_proba
    mockf_data.data_shape = data_shape
    mockf_data.proba_shape = proba_shape


@pytest.mark.parametrize(
    ('data', 'np_data', 'data_shape', 'np_datatype', 'expected_datatype', 'prediction', 'proba_shape', 'prediction_proba'),
    [(f_data, np.array(f_data).astype(np.float), (1, 3), 'float', np.float, 0, (1, 4), np.array([[0.1, 0.2, 0.3, 0.4]])),
     (f_data, np.array(f_data).astype(np.float64), (1, 3), 'float64', np.float64, 0, (1, 4), np.array([[0.1, 0.2, 0.3, 0.4]])),
     (i_data, np.array(i_data).astype(np.int8), (1, 3), 'int8', np.int8, 0, (1, 4), np.array([[0.1, 0.2, 0.3, 0.4]]))]
)
def test_BaseDataExtension(mocker, data, np_data, data_shape, np_datatype, expected_datatype, prediction, proba_shape, prediction_proba):
    class MockData(BaseData):
        testf_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]

    mockf_data = MockData()
    mockf_data.data = data
    mockf_data.np_datatype = np_datatype
    mockf_data.prediction = prediction
    mockf_data.prediction_proba = prediction_proba
    mockf_data.data_shape = data_shape
    mockf_data.proba_shape = proba_shape

    mock_basef_data_extension = BaseDataExtension(data_object=mockf_data)
    mock_basef_data_extension.convert_data_to_np_data()
    assert mockf_data.np_data.shape == mockf_data.data_shape
    assert mockf_data.np_data.dtype == expected_datatype
    np.testing.assert_equal(mockf_data.np_data, np_data)
