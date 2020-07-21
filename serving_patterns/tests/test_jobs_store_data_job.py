import pytest
import json
from typing import List, Tuple

from app.constants import CONSTANTS
from app.jobs import store_data_job
from app.ml.base_predictor import BaseData, BaseDataInterface


test_job_id = '550e8400-e29b-41d4-a716-446655440000_0'


class MockData(BaseData):
    data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]
    test_data: List[List[int]] = [[5.1, 3.5, 1.4, 0.2]]


class MockDataInterface(BaseDataInterface):
    input_shape: Tuple[int] = (1, 4)
    input_type: str = 'float64'
    output_shape: Tuple[int] = (1, 3)
    output_type: str = 'float64'


@pytest.mark.parametrize(
    ('queue_name', 'key', 'expected'),
    [(CONSTANTS.REDIS_QUEUE, 'abc', True)]
)
def test_left_push_queue(mocker, queue_name, key, expected):
    mocker.patch('app.middleware.redis_client.redis_client.lpush', return_value=expected)
    result = store_data_job.left_push_queue(queue_name, key)
    assert result == expected


@pytest.mark.parametrize(
    ('queue_name', 'num', 'key'),
    [(CONSTANTS.REDIS_QUEUE, 1, 'abc')]
)
def test_right_pop_queue(mocker, queue_name, num, key):
    mocker.patch('app.middleware.redis_client.redis_client.llen', return_value=num)
    mocker.patch('app.middleware.redis_client.redis_client.rpop', return_value=key)
    result = store_data_job.right_pop_queue(queue_name)
    assert result == key


@pytest.mark.parametrize(
    ('queue_name', 'num'),
    [(CONSTANTS.REDIS_QUEUE, 0)]
)
def test_right_pop_queue_none(mocker, queue_name, num):
    mocker.patch('app.middleware.redis_client.redis_client.llen', return_value=num)
    mocker.patch('app.middleware.redis_client.redis_client.rpop', return_value=None)
    result = store_data_job.right_pop_queue(queue_name)
    assert result is None


@pytest.mark.parametrize(
    ('key', 'data'),
    [(test_job_id, {'data': [1.0, -1.0], 'prediction': None})]
)
def test_load_data_redis(mocker, key, data):
    mocker.patch('app.middleware.redis_client.redis_client.get', return_value=data)
    mocker.patch('json.loads', return_value=data)
    result = store_data_job.load_data_redis(key)
    assert result['data'] == data['data']
    assert result['prediction'] == data['prediction']


@pytest.mark.parametrize(
    ('job_id', 'directory', 'data'),
    [(test_job_id, '/test', {'data': [1.0, -1.0], 'prediction': None})]
)
def test_save_data_file_job(mocker, tmpdir, job_id, directory, data):
    tmp_file = tmpdir.mkdir(directory).join(f'{job_id}.json')
    mocker.patch('os.path.join', return_value=tmp_file)
    result = store_data_job.save_data_file_job(job_id, directory, data)
    assert result
    with open(tmp_file, 'r') as f:
        assert data == json.load(f)


class Test:
    mock_redis_cache = {}

    @pytest.mark.parametrize(
        ('job_id', 'data'),
        [(test_job_id, MockData())]
    )
    def test_save_data_redis_job(self, mocker, job_id, data) -> None:
        def set(key, value):
            self.mock_redis_cache[key] = value

        def get(key):
            return self.mock_redis_cache.get(key, None)
        mocker.patch(
            'app.middleware.redis_client.redis_client.incr',
            return_value=0)
        mocker.patch(
            'app.middleware.redis_client.redis_client.set').side_effect = set

        result = store_data_job.save_data_redis_job(job_id, data)
        assert result
        assert get(job_id) is not None

    @pytest.mark.parametrize(
        ('job_id', 'data'),
        [(test_job_id, {'data': [1.0, -1.0], 'prediction': None})]
    )
    def test_save_data_dict_redis_job(self, mocker, job_id, data) -> None:
        def set(key, value):
            self.mock_redis_cache[key] = value

        def get(key):
            return self.mock_redis_cache.get(key, None)
        mocker.patch(
            'app.middleware.redis_client.redis_client.incr',
            return_value=0)
        mocker.patch(
            'app.middleware.redis_client.redis_client.set').side_effect = set

        result = store_data_job.save_data_dict_redis_job(job_id, data)
        assert result
        assert get(job_id) is not None


@pytest.mark.parametrize(
    ('job_id', 'directory', 'data'),
    [(test_job_id, '/tmp/', {'data': [1.0, -1.0], 'prediction': None})]
)
def test_SaveDataFileJob(mocker, job_id, directory, data):
    save_data_file_job = store_data_job.SaveDataFileJob(
        job_id=job_id,
        directory=directory,
        data=data
    )
    mocker.patch(
        'app.jobs.store_data_job.save_data_file_job',
        return_value=True)
    save_data_file_job()
    assert save_data_file_job.is_completed


@pytest.mark.parametrize(
    ('job_id', 'data', 'queue_name', 'enqueue'),
    [(test_job_id, MockData(), CONSTANTS.REDIS_QUEUE, True),
     (test_job_id, MockData(), 'aaaaaaaa', False)]
)
def test_SaveDataRedisJob(mocker, job_id, data, queue_name, enqueue):
    save_data_redis_job = store_data_job.SaveDataRedisJob(
        job_id=job_id,
        data=data,
        queue_name=queue_name,
        enqueue=enqueue
    )
    mocker.patch(
        'app.jobs.store_data_job.left_push_queue',
        return_value=True)
    mocker.patch(
        'app.jobs.store_data_job.save_data_redis_job',
        return_value=True)
    save_data_redis_job()
    assert save_data_redis_job.is_completed
