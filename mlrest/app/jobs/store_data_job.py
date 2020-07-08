import os
from typing import Dict, Any
import logging
from pydantic import BaseModel
import json
import numpy as np

from app.constants import CONSTANTS
from app.middleware.redis import redis_connector
from app.ml.base_predictor import BaseData

logger = logging.getLogger(__name__)


def left_push_queue(queue_name: str, key: str):
    redis_connector.lpush(queue_name, key)


def right_pop_queue(queue_name: str) -> Any:
    if redis_connector.llen(queue_name) > 0:
        return redis_connector.rpop(queue_name)
    else:
        return None


def load_data_redis(key: str) -> Dict[str, Any]:
    data_dict = json.loads(redis_connector.get(key))
    return data_dict


def save_data_file_job(job_id: str, directory: str, data: Any) -> bool:
    file_path = os.path.join(directory, f'{job_id}.json')
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return True


def save_data_redis_job(job_id: str, data: BaseData) -> bool:
    data_dict = {}
    for k, v in data.__dict__.items():
        data_dict[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return save_data_dict_redis_job(job_id, data_dict)


def save_data_dict_redis_job(job_id: str, data: Dict[str, Any]) -> bool:
    redis_connector.incr(CONSTANTS.REDIS_INCREMENTS)
    logger.info({job_id: data})
    redis_connector.set(job_id, json.dumps(data))
    return True


class SaveDataJob(BaseModel):
    job_id: str
    data: BaseData
    is_completed: bool = False

    def __call__(self):
        pass


class SaveDataFileJob(SaveDataJob):
    directory: str

    def __call__(self):
        save_data_jobs[self.job_id] = self
        logger.info(
            f'registered job: {self.job_id} in {self.__class__.__name__}')
        self.is_completed = save_data_file_job(
            self.job_id, self.directory, self.data)
        logger.info(f'completed save data: {self.job_id}')


class SaveDataRedisJob(SaveDataJob):

    def __call__(self):
        save_data_jobs[self.job_id] = self
        logger.info(
            f'registered job: {self.job_id} in {self.__class__.__name__}')
        self.is_completed = save_data_redis_job(self.job_id, self.data)
        logger.info(f'completed save data: {self.job_id}')


save_data_jobs: Dict[str, SaveDataJob] = {}
