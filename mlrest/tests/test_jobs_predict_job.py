import pytest
import numpy as np
from app.jobs import predict_job
from app.constants import CONSTANTS
from app.ml.base_predictor import BasePredictor


test_job_id = '550e8400-e29b-41d4-a716-446655440000_0'


class MockPredictor(BasePredictor):
    def load_model(self):
        pass

    def predict_proba(self, data):
        return None

    def predict_proba_from_dict(self, data):
        return None


@pytest.mark.parametrize(('job_id',
                          'directory',
                          'data',
                          'expected'),
                         [(test_job_id,
                           '/tmp/',
                           {'data': [1.0, -1.0],
                            'prediction': CONSTANTS.NONE_DEFAULT,
                            'prediction_proba': CONSTANTS.NONE_DEFAULT_LIST},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])}),
                          (test_job_id,
                           '/tmp/',
                           {'data': [1.0, -1.0],
                            'prediction': CONSTANTS.NONE_DEFAULT,
                            'prediction_proba': np.array([0.9, 0.1])},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])}),
                          (test_job_id,
                           '/tmp/',
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([0.9, 0.1])},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])})])
def test_predict_from_file(mocker, tmpdir, job_id, directory, data, expected):
    file_path = tmpdir.mkdir('/tmp/').join('test.json')
    file_path.write('a')
    mocker.patch('os.path.join', return_value=file_path)
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('json.load', return_value=data)

    mock_predictor = MockPredictor()
    mocker.patch.object(
        mock_predictor,
        'predict_proba_from_dict',
        return_value=expected['prediction_proba'])

    data_dict = predict_job.predict_from_file(
        job_id, directory, mock_predictor)
    assert expected['prediction'] == data_dict['prediction']
    np.testing.assert_equal(
        expected['prediction_proba'][0],
        data_dict['prediction_proba'])


@pytest.mark.parametrize(('job_id',
                          'directory',
                          'data',
                          'expected'),
                         [(test_job_id,
                           '/tmp/',
                           {},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])})])
def test_predict_from_file_none(
        mocker,
        tmpdir,
        job_id,
        directory,
        data,
        expected):
    file_path = tmpdir.mkdir('/tmp/').join('test.json')
    file_path.write('a')
    mocker.patch('os.path.join', return_value=file_path)
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('json.load', return_value=data)

    mock_predictor = MockPredictor()
    mocker.patch.object(
        mock_predictor,
        'predict_proba_from_dict',
        return_value=expected['prediction_proba'])

    data_dict = predict_job.predict_from_file(
        job_id, directory, mock_predictor)
    assert data_dict is None


@pytest.mark.parametrize(('job_id',
                          'data',
                          'expected'),
                         [(test_job_id,
                           {'data': 'list_float_1.0;-1.0',
                            'prediction': CONSTANTS.NONE_DEFAULT,
                            'prediction_proba': CONSTANTS.NONE_DEFAULT_LIST_CONVERTED},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])}),
                          (test_job_id,
                           {'data': 'list_float_1.0;-1.0',
                            'prediction': CONSTANTS.NONE_DEFAULT,
                            'prediction_proba': 'list_float_0.9;0.1'},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])}),
                          (test_job_id,
                           {'data': 'list_float_1.0;-1.0',
                            'prediction': 0,
                            'prediction_proba': 'list_float_0.9;0.1'},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])})])
def test_predict_from_redis_cache(mocker, job_id, data, expected):
    mocker.patch('app.middleware.redis.redis_connector.hgetall',
                 return_value=data)

    mock_predictor = MockPredictor()
    mocker.patch.object(
        mock_predictor,
        'predict_proba_from_dict',
        return_value=expected['prediction_proba'])

    data_dict = predict_job.predict_from_redis_cache(job_id, mock_predictor)
    assert expected['prediction'] == data_dict['prediction']
    np.testing.assert_equal(
        expected['prediction_proba'][0],
        data_dict['prediction_proba'])


@pytest.mark.parametrize(
    ('job_id', 'directory', 'data_dict'),
    [(test_job_id, '/tmp/',
      {'data': [1.0, -1.0], 'prediction': 0, 'prediction_proba': np.array([[0.9, 0.1]])})]
)
def test_PredictFromFileJob(mocker, job_id, directory, data_dict):
    mock_predictor = MockPredictor()
    mocker.patch(
        'app.jobs.predict_job.predict_from_file',
        return_value=data_dict)
    mocker.patch(
        'app.jobs.store_data_job.save_data_file_job',
        return_value=True)
    predict_from_file_job = predict_job.PredictFromFileJob(
        job_id=job_id, directory=directory, predictor=mock_predictor)
    predict_from_file_job()
    assert predict_from_file_job.is_completed


@pytest.mark.parametrize(
    ('job_id', 'data_dict'),
    [(test_job_id,
      {'data': [1.0, -1.0], 'prediction': 0, 'prediction_proba': np.array([[0.9, 0.1]])})]
)
def test_PredictFromRedisJob(mocker, job_id, data_dict):
    mock_predictor = MockPredictor()
    mocker.patch(
        'app.jobs.predict_job.predict_from_redis_cache',
        return_value=data_dict)
    mocker.patch(
        'app.jobs.store_data_job.save_data_redis_job',
        return_value=True)
    predict_from_redis_job = predict_job.PredictFromRedisJob(
        job_id=job_id, predictor=mock_predictor)
    predict_from_redis_job()
    assert predict_from_redis_job.is_completed
