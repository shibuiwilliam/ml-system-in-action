import pytest
import numpy as np
from typing import List, Tuple

from app.ml.base_predictor import BaseData, BaseDataExtension, BasePredictor
from app.backend.prediction_batch import prediction_batch


test_job_id = '550e8400-e29b-41d4-a716-446655440000_0'
f_proba = [0.7, 0.2, 0.1]
f_data = [[5.1, 3.5, 1.4, 0.2]]


class MockData(BaseData):
    data: List[List[float]] = f_data
    input_shape: Tuple[int] = (1, 4)
    input_type: str = 'float64'
    output_shape: Tuple[int] = (1, 3)
    output_type: str = 'float64'


class MockDataExtension(BaseDataExtension):
    pass


@pytest.mark.parametrize(
    ('job_id', 'data'),
    [(test_job_id, MockData())]
)
def test_run_prediction_if_queue(mocker, job_id, data):
    mocker.patch('app.jobs.store_data_job.right_pop_queue', return_value=job_id)
    mocker.patch('app.api._predict._predict_from_redis_cache', return_value=data)
    mocker.patch('app.jobs.store_data_job.save_data_redis_job', return_value=None)
    mocker.patch('app.jobs.store_data_job.left_push_queue', return_value=None)
    mocker.patch('app.jobs.store_data_job.load_data_redis', return_value=data.__dict__)

    prediction_batch._run_prediction_if_queue()


@pytest.mark.parametrize(
    ('job_id', 'data'),
    [(None, None)]
)
def test_run_prediction_if_queue_none(mocker, job_id, data):
    mocker.patch('app.jobs.store_data_job.right_pop_queue', return_value=job_id)
    mocker.patch('app.api._predict._predict_from_redis_cache', return_value=data)
    mocker.patch('app.jobs.store_data_job.save_data_redis_job', return_value=None)
    mocker.patch('app.jobs.store_data_job.left_push_queue', return_value=None)
    mocker.patch('app.jobs.store_data_job.load_data_redis', return_value=data)

    prediction_batch._run_prediction_if_queue()
