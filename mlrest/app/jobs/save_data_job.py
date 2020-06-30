import os
from typing import Dict, Any, List, Tuple
import logging
from pydantic import BaseModel
import json

from app.constants import CONSTANTS
from app.middleware import redis

logger = logging.getLogger(__name__)


def save_data_file_job(job_id: str, directory: str, data: Any) -> bool:
    file_path = os.path.join(directory, f'{job_id}.json')
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return True


def save_data_redis_job(job_id: str, data: Any) -> bool:
    redis.redis_connector.incr(CONSTANTS.REDIS_INCREMENTS)
    logger.info({job_id: data})
    _data = {}
    for k, v in data.items():
        if isinstance(v, List) or isinstance(v, Tuple):
            if isinstance(v[0], int):
                _type = 'int'
            elif isinstance(v[0], float):
                _type = 'float'
            elif isinstance(v[0], str):
                _type = 'str'
            else:
                _type = 'None'
            _data[k] = f'list_{_type}_' + \
                CONSTANTS.SEPARATOR.join([str(_v) for _v in v])
        else:
            _data[k] = v
    redis.redis_connector.hmset(job_id, _data)
    return True


class SaveDataJob(BaseModel):

    def __call__(self):
        pass


class SaveDataFileJob(SaveDataJob):
    job_id: str
    directory: str
    data: Any
    is_completed: bool = False

    def __call__(self):
        save_data_jobs[self.job_id] = self
        logger.info(
            f'registered job: {self.job_id} in {self.__class__.__name__}')
        self.is_completed = save_data_file_job(
            self.job_id, self.directory, self.data)
        logger.info(f'completed save data: {self.job_id}')


class SaveDataRedisJob(SaveDataJob):
    job_id: str
    data: Any
    is_completed: bool = False

    def __call__(self):
        save_data_jobs[self.job_id] = self
        logger.info(
            f'registered job: {self.job_id} in {self.__class__.__name__}')
        self.is_completed = save_data_redis_job(self.job_id, self.data)
        logger.info(f'completed save data: {self.job_id}')


save_data_jobs: Dict[str, SaveDataJob] = {}
