import pytest
from fastapi import BackgroundTasks
from typing import List, Tuple
import numpy as np

from app.constants import PLATFORM_ENUM
from app.ml.base_predictor import BaseData, BaseDataExtension, BasePredictor
import app
from app.api._predict import (
    _save_data_job,
    __predict,
    _predict_from_redis_cache,
    _test,
    _predict,
    _predict_async_post,
    _predict_async_get)


test_uuid = '550e8400-e29b-41d4-a716-446655440000'
job_id = f'{test_uuid}_0'
mock_BackgroundTasks = BackgroundTasks()
f_proba = [0.7, 0.2, 0.1]
f_data = [[5.1, 3.5, 1.4, 0.2]]


class MockPredictor(BasePredictor):
    def load_model(self):
        pass

    def predict(self, data):
        return None


class MockData(BaseData):
    data: List[List[float]] = [[5.1, 3.5, 1.4, 0.2]]
    test_data: List[List[float]] = [[5.1, 3.5, 1.4, 0.2]]
    input_shape: Tuple[int] = (1, 4)
    input_type: str = 'float64'
    output_shape: Tuple[int] = (1, 3)
    output_type: str = 'float64'


class MockDataExtension(BaseDataExtension):
    pass


class MockJob():
    def __call__(self):
        return True


@pytest.mark.parametrize(
    ('_uuid', 'data', 'num', 'expected'),
    [(test_uuid, MockData(), 0, f'{test_uuid}_0'),
     (test_uuid, MockData(), 1, f'{test_uuid}_1'),
     (test_uuid, MockData(), None, f'{test_uuid}_0'),
     (test_uuid, MockData(output=np.array([[0.1, 0.2, 0.7]])), 0, f'{test_uuid}_0')]
)
def test_save_data_job(mocker, _uuid, data, num, expected):
    mock_job = MockJob()
    app.api._predict.PLATFORM = PLATFORM_ENUM.DOCKER_COMPOSE.value
    mocker.patch('app.middleware.redis_client.redis_client.get', return_value=num)
    mocker.patch('uuid.uuid4', return_value=_uuid)
    mocker.patch(
        'app.jobs.store_data_job.SaveDataRedisJob',
        return_value=mock_job)
    job_id = _save_data_job(data, mock_BackgroundTasks)
    assert job_id == expected


@pytest.mark.parametrize(
    ('output', 'expected'),
    [(np.array([[0.8, 0.1, 0.1]]), {'prediction': [[0.8, 0.1, 0.1]]}),
     (np.array([[0.2, 0.1, 0.7]]), {'prediction': [[0.2, 0.1, 0.7]]})]
)
def test__predict(mocker, output, expected):
    mock_data = MockData()
    mock_data_extension = MockDataExtension(mock_data)
    mocker.patch.object(
        mock_data_extension,
        'convert_input_data_to_np',
        return_value=np.array(mock_data.data).astype(np.float64).reshape(mock_data.input_shape))
    mocker.patch.object(
        mock_data_extension,
        'reshape_output',
        return_value=output)
    mocker.patch(
        'app.ml.active_predictor.active_predictor.predict',
        return_value=output)
    __predict(data=mock_data)
    assert mock_data.prediction == expected['prediction']


@pytest.mark.parametrize(
    ('job_id', 'data', 'expected'),
    [(job_id, {'data': f_data}, {'data': f_data, 'prediction': [f_proba]})]
)
def test_predict_from_redis_cache(mocker, job_id, data, expected):
    mock_data = MockData(
        data=data['data'],
        prediction=expected['prediction']
    )

    mocker.patch('app.jobs.store_data_job.load_data_redis', return_value=data)

    mocker.patch(
        'app.ml.active_predictor.active_predictor.predict',
        return_value=np.array(expected['prediction']))
    result = _predict_from_redis_cache(job_id)
    assert expected['data'] == result.data
    assert expected['prediction'] == result.prediction


@pytest.mark.parametrize(
    ('output', 'expected'),
    [(np.array([[0.8, 0.1, 0.1]]), {'prediction': [[0.8, 0.1, 0.1]]}),
     (np.array([[0.2, 0.1, 0.7]]), {'prediction': [[0.2, 0.1, 0.7]]})]
)
def test_test(mocker, output, expected):
    mocker.patch(
        'app.ml.active_predictor.active_predictor.predict',
        return_value=output)
    result = _test(data=MockData())
    assert result['prediction'] == expected['prediction']


@pytest.mark.parametrize(
    ('output', 'expected'),
    [(np.array([[0.8, 0.1, 0.1]]), {'prediction': [[0.8, 0.1, 0.1]]}),
     (np.array([[0.2, 0.1, 0.7]]), {'prediction': [[0.2, 0.1, 0.7]]})]
)
def test_predict(mocker, output, expected):
    mocker.patch(
        'app.ml.active_predictor.active_predictor.predict',
        return_value=output)
    mocker.patch('app.api._predict._save_data_job', return_value=job_id)
    result = _predict(MockData(), mock_BackgroundTasks)
    assert result['prediction'] == expected['prediction']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('job_id'),
    [(job_id)]
)
async def test_predict_async_post(mocker, job_id):
    mocker.patch('app.api._predict._save_data_job', return_value=job_id)
    result = await _predict_async_post(MockData(), mock_BackgroundTasks)
    assert result['job_id'] == job_id


@pytest.mark.parametrize(
    ('job_id', 'data_dict', 'expected'),
    [(job_id,
      {'data': [[5.1, 3.5, 1.4, 0.2]], 'prediction': [[0.8, 0.1, 0.1]]},
      {job_id: {'prediction': [[0.8, 0.1, 0.1]]}})]
)
def test_predict_async_get(mocker, job_id, data_dict, expected):
    app.api._predict.PLATFORM = PLATFORM_ENUM.DOCKER_COMPOSE.value
    mocker.patch(
        'app.jobs.store_data_job.load_data_redis',
        return_value=data_dict)
    result = _predict_async_get(job_id)
    assert result == expected
