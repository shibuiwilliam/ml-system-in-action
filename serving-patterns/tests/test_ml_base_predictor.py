import pytest
from typing import List
import numpy as np

from app.ml.base_predictor import BaseData, BaseDataExtension


f_data = [[0.1, 0.9, 1.1]]
i_data = [[1, 2, 3]]
f_proba = [0.1, 0.2, 0.3, 0.4]


@pytest.mark.parametrize(
    ('data', 'input_shape', 'input_type', 'prediction', 'output_shape'),
    [(f_data, (1, 3), 'float', [f_proba], (1, 4))]
)
def test_BaseData(
        mocker,
        data,
        input_shape,
        input_type,
        prediction,
        output_shape):
    class MockData(BaseData):
        test_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]

    mock_data = MockData()
    mock_data.data = data
    mock_data.input_type = input_type
    mock_data.prediction = prediction
    mock_data.input_shape = input_shape
    mock_data.output_shape = output_shape


@pytest.mark.parametrize(
    ('data_dict'),
    [(
        {'data': f_data,
         'input_shape': (1, 3),
         'input_type': 'float',
         'prediction': [f_proba],
         'output_shape': (1, 4),
         'output_type': 'float'}
    ),
        (
        {'data': f_data[0],
         'input_shape': (1, 3),
         'input_type': 'float',
         'prediction': [f_proba],
         'output_shape': (1, 4),
         'output_type': 'float'}
    )]
)
def test_BaseDataDict(
        mocker,
        data_dict):
    class MockData(BaseData):
        test_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]

    mock_data = MockData(**data_dict)
    assert mock_data.data == data_dict['data']
    assert mock_data.input_type == data_dict['input_type']
    assert mock_data.prediction == data_dict['prediction']
    assert mock_data.input_shape == data_dict['input_shape']
    assert mock_data.output_shape == data_dict['output_shape']
    assert mock_data.output_type == data_dict['output_type']


@pytest.mark.parametrize(
    ('data', 'input_shape', 'input_type', 'expected_input_datatype', 'prediction', 'output_shape', 'output_type', 'expected_output_datatype'),
    [(f_data, (1, 3), 'float', np.float, [f_proba], (1, 4), 'float64', np.float64),
     (f_data, (1, 3), 'float64', np.float64, [f_proba], (1, 4), 'float64', np.float64),
     (f_data[0], (1, 3), 'float32', np.float32, [f_proba], (1, 4), 'float64', np.float64),
     (i_data, (1, 3), 'int8', np.int8, [f_proba], (1, 4), 'float64', np.float64),
     (i_data[0], (1, 3), 'int16', np.int16, [f_proba], (1, 4), 'float64', np.float64)]
)
def test_BaseDataExtension(
        mocker,
        data,
        input_shape,
        input_type,
        expected_input_datatype,
        prediction,
        output_shape,
        output_type,
        expected_output_datatype):
    class MockData(BaseData):
        testf_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]

    mock_data = MockData()
    mock_data.data = data
    mock_data.input_shape = input_shape
    mock_data.input_type = input_type
    mock_data.prediction = prediction
    mock_data.output_shape = output_shape
    mock_data.output_type = output_type

    mock_base_data_extension = BaseDataExtension(data_object=mock_data)
    np_data = mock_base_data_extension.convert_input_data_to_np()
    output = mock_base_data_extension.reshape_output(np.array([f_proba]))
    assert np_data.shape == mock_data.input_shape
    assert np_data.dtype == expected_input_datatype
    assert output.shape == mock_data.output_shape
    assert output.dtype == expected_output_datatype
    # print()
    # print(mock_data.data)
    # print(mock_data.np_data)
    # print(mock_data.np_data.shape)
    # print(mock_data.np_data.dtype)
    # print(mock_data.output.shape)
    # print(mock_data.output.dtype)
