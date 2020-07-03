import pytest
import numpy as np
from app.jobs import predict_job
from app.constants import CONSTANTS
from app.ml.abstract_predictor import BasePredictor


class TestPredictor(BasePredictor):
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
                         [('550e8400-e29b-41d4-a716-446655440000_0',
                           '/tmp/',
                           {'data': [1.0, -1.0],
                            'prediction': CONSTANTS.NONE_DEFAULT,
                            'prediction_proba': CONSTANTS.NONE_DEFAULT_LIST},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])}),
                          ('550e8400-e29b-41d4-a716-446655440000_0',
                           '/tmp/',
                           {'data': [1.0, -1.0],
                            'prediction': CONSTANTS.NONE_DEFAULT,
                            'prediction_proba': np.array([0.9, 0.1])},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])}),
                          ('550e8400-e29b-41d4-a716-446655440000_0',
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

    mock_predictor = TestPredictor()
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
                         [('550e8400-e29b-41d4-a716-446655440000_0',
                           '/tmp/',
                           {},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])})])
def test_predict_from_file_none(mocker, tmpdir, job_id, directory, data, expected):
    file_path = tmpdir.mkdir('/tmp/').join('test.json')
    file_path.write('a')
    mocker.patch('os.path.join', return_value=file_path)
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('json.load', return_value=data)

    mock_predictor = TestPredictor()
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
                         [('550e8400-e29b-41d4-a716-446655440000_0',
                           {'data': 'list_float_1.0;-1.0',
                            'prediction': CONSTANTS.NONE_DEFAULT,
                            'prediction_proba': CONSTANTS.NONE_DEFAULT_LIST_CONVERTED},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])}),
                          ('550e8400-e29b-41d4-a716-446655440000_0',
                           {'data': 'list_float_1.0;-1.0',
                            'prediction': CONSTANTS.NONE_DEFAULT,
                            'prediction_proba': 'list_float_0.9;0.1'},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])}),
                          ('550e8400-e29b-41d4-a716-446655440000_0',
                           {'data': 'list_float_1.0;-1.0',
                            'prediction': 0,
                            'prediction_proba': 'list_float_0.9;0.1'},
                           {'data': [1.0, -1.0],
                            'prediction': 0,
                            'prediction_proba': np.array([[0.9, 0.1]])})])
def test_predict_from_redis_cache(mocker, job_id, data, expected):
    mocker.patch('app.middleware.redis.redis_connector.hgetall', return_value=data)

    mock_predictor = TestPredictor()
    mocker.patch.object(
        mock_predictor,
        'predict_proba_from_dict',
        return_value=expected['prediction_proba'])

    data_dict = predict_job.predict_from_redis_cache(job_id, mock_predictor)
    assert expected['prediction'] == data_dict['prediction']
    np.testing.assert_equal(
        expected['prediction_proba'][0],
        data_dict['prediction_proba'])
