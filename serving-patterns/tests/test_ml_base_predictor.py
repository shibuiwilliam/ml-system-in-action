import pytest
from typing import List
import numpy as np

from app.ml.base_predictor import BaseData, BaseMetaData, BaseDataConverter


f_data = [[0.1, 0.9, 1.1, 1.1]]
i_data = [[1, 2, 3, 4]]
f_proba = [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    ('input_data', 'prediction'),
    [(f_data, [f_proba])]
)
def test_BaseData(
        mocker,
        input_data,
        prediction):
    class MockData(BaseData):
        test_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]

    mock_data = MockData()
    mock_data.input_data = input_data
    mock_data.prediction = prediction


@pytest.mark.parametrize(
    ('data_dict'),
    [({'input_data': f_data, 'prediction': [f_proba]}),
     ({'input_data': f_data[0], 'prediction': [f_proba]})]
)
def test_BaseDataDict(
        mocker,
        data_dict):
    class MockData(BaseData):
        test_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]

    mock_data = MockData(**data_dict)
    assert mock_data.input_data == data_dict['input_data']
    assert mock_data.prediction == data_dict['prediction']


@pytest.mark.parametrize(
    ('input_shape', 'input_type', 'output_shape', 'output_type'),
    [((1, 4), 'float32', (1, 3), 'float32')]
)
def test_BaseMetaData(
        mocker,
        input_shape,
        input_type,
        output_shape,
        output_type):
    class MockMetaData(BaseMetaData):
        pass

    MockMetaData.input_shape = input_shape
    MockMetaData.input_type = input_type
    MockMetaData.output_shape = output_shape
    MockMetaData.output_type = output_type


@pytest.mark.parametrize(
    ('meta_data_dict'),
    [(
        {'input_shape': (1, 4),
         'input_type': 'float32',
         'output_shape': (1, 3),
         'output_type': 'float32'}
    ),
        (
        {'input_shape': (1, 4),
         'input_type': 'float32',
         'output_shape': (1, 3),
         'output_type': 'float32'}
    )]
)
def test_BaseDataDict(
        mocker,
        meta_data_dict):
    class MockMetaData(BaseMetaData):
        pass

    MockMetaData.input_type = meta_data_dict['input_type']
    MockMetaData.input_shape = meta_data_dict['input_shape']
    MockMetaData.output_shape = meta_data_dict['output_shape']
    MockMetaData.output_type = meta_data_dict['output_type']
    assert MockMetaData.input_type == meta_data_dict['input_type']
    assert MockMetaData.input_shape == meta_data_dict['input_shape']
    assert MockMetaData.output_shape == meta_data_dict['output_shape']
    assert MockMetaData.output_type == meta_data_dict['output_type']


@pytest.mark.parametrize(
    ('input_data', 'input_shape', 'input_type', 'expected_input_datatype', 'prediction', 'output_shape', 'output_type', 'expected_output_datatype'),
    [(f_data, (1, 4), 'float', np.float, [f_proba], (1, 3), 'float32', np.float32),
     (f_data, (1, 4), 'float64', np.float64, [f_proba], (1, 3), 'float32', np.float32),
     (f_data[0], (1, 4), 'float32', np.float32, [f_proba], (1, 3), 'float32', np.float32),
     (i_data, (1, 4), 'int8', np.int8, [f_proba], (1, 3), 'float32', np.float32),
     (i_data[0], (1, 4), 'int16', np.int16, [f_proba], (1, 3), 'float32', np.float32)]
)
def test_BaseDataConverter(
        mocker,
        input_data,
        input_shape,
        input_type,
        expected_input_datatype,
        prediction,
        output_shape,
        output_type,
        expected_output_datatype):
    class MockData(BaseData):
        testf_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]

    class MockMetaData(BaseMetaData):
        pass

    mock_data = MockData()
    mock_data.input_data = input_data
    mock_data.prediction = prediction

    MockMetaData.input_shape = input_shape
    MockMetaData.input_type = input_type
    MockMetaData.output_shape = output_shape
    MockMetaData.output_type = output_type

    BaseDataConverter.meta_data = MockMetaData
    np_data = BaseDataConverter.convert_input_data_to_np(mock_data.input_data)
    output = BaseDataConverter.reshape_output(np.array([mock_data.prediction]))
    assert np_data.shape == BaseDataConverter.meta_data.input_shape
    assert np_data.dtype == expected_input_datatype
    assert output.shape == BaseDataConverter.meta_data.output_shape
    assert output.dtype == expected_output_datatype
    # print()
    # print(MockMetaData.__dict__)
    # print(BaseDataConverter.meta_data.__dict__)
    # print(MockMetaData.input_shape)
    # print(MockMetaData.output_shape)
    # print(np_data.shape)
    # print(output.shape)
    # print(np_data.dtype)
    # print(output.dtype)


@pytest.mark.parametrize(
    ('input_data', 'input_shape', 'input_type', 'expected_input_datatype', 'prediction', 'output_shape', 'output_type', 'expected_output_datatype'),
    [(f_data, (1, 4), 'float', np.float, [f_proba], (1, 3), 'float32', np.float32),
     (f_data, (1, 4), 'float64', np.float64, [f_proba], (1, 3), 'float32', np.float32),
     (f_data[0], (1, 4), 'float32', np.float32, [f_proba], (1, 3), 'float32', np.float32),
     (i_data, (1, 4), 'int8', np.int8, [f_proba], (1, 3), 'float32', np.float32),
     (i_data[0], (1, 4), 'int16', np.int16, [f_proba], (1, 3), 'float32', np.float32)]
)
def test_BaseDataConverter2(
        mocker,
        input_data,
        input_shape,
        input_type,
        expected_input_datatype,
        prediction,
        output_shape,
        output_type,
        expected_output_datatype):
    class MockData(BaseData):
        testf_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]

    class MockMetaData(BaseMetaData):
        pass

    class MockDataConverter(BaseDataConverter):
        pass


    mock_data = MockData()
    mock_data.input_data = input_data
    mock_data.prediction = prediction

    MockMetaData.input_shape = input_shape
    MockMetaData.input_type = input_type
    MockMetaData.output_shape = output_shape
    MockMetaData.output_type = output_type

    MockDataConverter.meta_data = MockMetaData
    np_data = MockDataConverter.convert_input_data_to_np(mock_data.input_data)
    output = MockDataConverter.reshape_output(np.array([mock_data.prediction]))
    assert np_data.shape == MockDataConverter.meta_data.input_shape
    assert np_data.dtype == expected_input_datatype
    assert output.shape == MockDataConverter.meta_data.output_shape
    assert output.dtype == expected_output_datatype
    # print()
    # print(MockMetaData.__dict__)
    # print(MockDataConverter.meta_data.__dict__)
    # print(MockMetaData.input_shape)
    # print(MockMetaData.output_shape)
    # print(MockDataConverter.meta_data.input_shape)
    # print(MockDataConverter.meta_data.output_shape)
    # print(np_data.shape)
    # print(output.shape)
    # print(np_data.dtype)
    # print(output.dtype)
