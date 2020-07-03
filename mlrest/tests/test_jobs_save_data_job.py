import pytest
import json

from app.jobs import save_data_job


@pytest.mark.parametrize(
    ('job_id', 'directory', 'data'),
    [('550e8400-e29b-41d4-a716-446655440000_0', '/test', {'data': [1.0, -1.0], 'prediction': None})]
)
def test_save_data_file_job(mocker, tmpdir, job_id, directory, data):
    tmp_file = tmpdir.mkdir(directory).join(f'{job_id}.json')
    mocker.patch('os.path.join', return_value=tmp_file)
    result = save_data_job.save_data_file_job(job_id, directory, data)
    assert result
    with open(tmp_file, 'r') as f:
        assert data == json.load(f)


class Test:
    mock_redis_cache = {}

    @pytest.mark.parametrize(
        ('job_id', 'data'),
        [('550e8400-e29b-41d4-a716-446655440000_0', {'data': [1.0, -1.0], 'prediction': None})]
    )
    def test_save_data_redis_job(self, mocker, job_id, data) -> None:
        def set(key, value):
            self.mock_redis_cache[key] = value

        def get(key):
            return self.mock_redis_cache.get(key, None)
        mocker.patch(
            'app.middleware.redis.redis_connector.incr',
            return_value=0)
        mocker.patch(
            'app.middleware.redis.redis_connector.hmset').side_effect = set

        result = save_data_job.save_data_redis_job(job_id, data)
        assert result
        assert get(job_id) is not None


@pytest.mark.parametrize(
    ('job_id', 'directory', 'data'),
    [('aaaa', '/tmp/', {'data': [1.0, -1.0], 'prediction': None})]
)
def test_SaveDataFileJob(mocker, job_id, directory, data):
    save_data_file_job = save_data_job.SaveDataFileJob(
        job_id=job_id,
        directory=directory,
        data=data
    )
    mocker.patch(
        'app.jobs.save_data_job.save_data_file_job',
        return_value=True)
    save_data_file_job()


@pytest.mark.parametrize(
    ('job_id', 'data'),
    [('aaaa', {'data': [1.0, -1.0], 'prediction': None})]
)
def test_SaveDataRedisJob(mocker, job_id, data):
    save_data_file_job = save_data_job.SaveDataRedisJob(
        job_id=job_id,
        data=data
    )
    mocker.patch(
        'app.jobs.save_data_job.save_data_redis_job',
        return_value=True)
    save_data_file_job()
