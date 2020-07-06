import pytest
import numpy as np
from typing import List, Tuple

from app.jobs import predict_job
from app.ml.base_predictor import BaseData, BaseDataExtension, BasePredictor


test_job_id = '550e8400-e29b-41d4-a716-446655440000_0'
f_proba = [0.9, 0.1]
f_data = [[5.1, 3.5, 1.4, 0.2]]


class MockPredictor(BasePredictor):
    def load_model(self):
        pass

    def predict_proba(self, data):
        return None

    def predict_proba_from_dict(self, data):
        return None


class MockData(BaseData):
    data: List[List[float]] = f_data
    input_shape: Tuple[int] = (1, 4)
    input_type: str = 'float64'
    output_shape: Tuple[int] = (1, 3)
    output_type: str = 'float64'


class MockDataExtension(BaseDataExtension):
    pass


@pytest.mark.parametrize(('job_id',
                          'data',
                          'expected'),
                         [(test_job_id,
                           {'data': f_data},
                           {'data': f_data,
                            'np_data': np.array(f_data),
                            'prediction': [f_proba],
                            'output': np.array([f_proba])}),
                          (test_job_id,
                           {'data': f_data},
                           {'data': f_data,
                            'np_data': np.array(f_data),
                            'prediction': [f_proba],
                            'output': np.array([f_proba])}),
                          (test_job_id,
                           {'data': f_data},
                           {'data': f_data,
                            'np_data': np.array(f_data),
                            'prediction': [f_proba],
                            'output': np.array([f_proba])})])
def test_predict_from_redis_cache(mocker, job_id, data, expected):
    mocker.patch('app.jobs.store_data_job.load_data_redis', return_value=data)

    mock_predictor = MockPredictor()
    mocker.patch.object(
        mock_predictor,
        'predict_proba',
        return_value=expected['output'])

    result = predict_job.predict_from_redis_cache(
        job_id, mock_predictor, MockData, MockDataExtension)

    assert expected['data'] == result.data
    np.testing.assert_equal(expected['np_data'], result.np_data)
    np.testing.assert_equal(expected['output'], result.output)


@pytest.mark.parametrize(
    ('job_id', 'data'),
    [(test_job_id,
      MockData(
          np_data=np.array(f_data),
          output=np.array([f_proba])
      ))]
)
def test_PredictFromRedisJob(mocker, job_id, data):
    mock_predictor = MockPredictor()
    mocker.patch(
        'app.jobs.predict_job.predict_from_redis_cache',
        return_value=data)
    mocker.patch(
        'app.jobs.store_data_job.save_data_redis_job',
        return_value=True)

    predict_from_redis_job = predict_job.PredictFromRedisJob(
        job_id=job_id,
        predictor=mock_predictor,
        baseData=MockData,
        baseDataExtentions=MockDataExtension
    )
    predict_from_redis_job()
    assert predict_from_redis_job.is_completed
