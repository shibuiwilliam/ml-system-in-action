import pytest
from fastapi import BackgroundTasks
from typing import List, Tuple
import numpy as np

from app.constants import CONSTANTS, PLATFORM_ENUM
from app.ml.abstract_predictor import BaseData, BasePredictor
import app
from app.api._predict import (
    _save_data_job,
    _predict_job,
    _test,
    _predict,
    _predict_async_post,
    _predict_async_get)


test_uuid = '550e8400-e29b-41d4-a716-446655440000'
job_id = f'{test_uuid}_0'
mock_BackgroundTasks = BackgroundTasks()


class MockPredictor(BasePredictor):
    def load_model(self):
        pass

    def predict_proba(self, data):
        return None

    def predict_proba_from_dict(self, data):
        return None


class MockData(BaseData):
    test_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]
    input_shape: Tuple[int] = (1, 4)
    input_type: str = 'float64'
    output_shape: Tuple[int] = (1, 3)
    output_type: str = 'int64'


class MockJob():
    def __call__(self):
        return True


@pytest.mark.parametrize(
    ('_uuid', 'data', 'num', 'expected'),
    [(test_uuid, MockData(), 0, f'{test_uuid}_0'),
     (test_uuid, MockData(), 1, f'{test_uuid}_1'),
     (test_uuid, MockData(), None, f'{test_uuid}_0'),
     (test_uuid, MockData(prediction_proba=np.array(CONSTANTS.NONE_DEFAULT_LIST)), 0, f'{test_uuid}_0'),
     (test_uuid, MockData(prediction_proba=np.array([[0.1, 0.2, 0.7]])), 0, f'{test_uuid}_0')]
)
def test_save_data_job(mocker, _uuid, data, num, expected):
    mock_job = MockJob()
    app.api._predict.PLATFORM = PLATFORM_ENUM.DOCKER_COMPOSE.value
    mocker.patch('app.middleware.redis.redis_connector.get', return_value=num)
    mocker.patch('uuid.uuid4', return_value=_uuid)
    mocker.patch(
        'app.jobs.save_data_job.SaveDataRedisJob',
        return_value=mock_job)
    job_id = _save_data_job(data, mock_BackgroundTasks)
    assert job_id == expected


@pytest.mark.parametrize(
    ('job_id', 'expected'),
    [(job_id, job_id)]
)
def test_predict_job(mocker, job_id, expected):
    mock_job = MockJob()
    app.api._predict.PLATFORM = PLATFORM_ENUM.DOCKER_COMPOSE.value
    mocker.patch(
        'app.jobs.predict_job.PredictFromRedisJob',
        return_value=mock_job)
    job_id = _predict_job(job_id, mock_BackgroundTasks)
    assert job_id == expected


@pytest.mark.parametrize(
    ('proba', 'expected'),
    [(np.array([[0.9, 0.1]]), {'prediction': 0}),
     (np.array([[0.2, 0.1, 0.7]]), {'prediction': 2})]
)
def test_test(mocker, proba, expected):
    mocker.patch(
        'app.ml.active_predictor.predictor.predict_proba',
        return_value=proba)
    result = _test(data=MockData())
    assert result['prediction'] == expected['prediction']


@pytest.mark.parametrize(
    ('proba', 'expected'),
    [(np.array([[0.9, 0.1]]), {'prediction': 0}),
     (np.array([[0.2, 0.1, 0.7]]), {'prediction': 2})]
)
def test_predict(mocker, proba, expected):
    mocker.patch(
        'app.ml.active_predictor.predictor.predict_proba',
        return_value=proba)
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
    mocker.patch('app.api._predict._predict_job', return_value=job_id)
    result = await _predict_async_post(MockData(), mock_BackgroundTasks)
    assert result['job_id'] == job_id


@pytest.mark.parametrize(
    ('job_id', 'data_dict', 'expected'),
    [(job_id,
      {'data': [1.0, -1.0], 'prediction': 0, 'prediction_proba': np.array([[0.9, 0.1]])},
      {job_id: {'prediction': 0}}),
     (job_id,
      None,
      {job_id: {'prediction': CONSTANTS.NONE_DEFAULT}}),
     (job_id,
      {'data': [1.0, -1.0], 'prediction_proba': np.array([[0.9, 0.1]])},
      {job_id: {'prediction': CONSTANTS.NONE_DEFAULT}})]
)
def test_predict_async_get(mocker, job_id, data_dict, expected):
    app.api._predict.PLATFORM = PLATFORM_ENUM.DOCKER_COMPOSE.value
    mocker.patch(
        'app.middleware.redis.redis_connector.hgetall',
        return_value=data_dict)
    result = _predict_async_get(job_id)
    assert result == expected
