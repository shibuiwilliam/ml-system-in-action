import os
from typing import Dict, Any
from fastapi import BackgroundTasks
import logging
from pydantic import BaseModel
import json
import numpy as np
import uuid
from PIL import Image

from src.constants import PLATFORM_ENUM, CONSTANTS
from src.configurations import _PlatformConfigurations
from src.middleware.redis_client import redis_client
from src.configurations import _FileConfigurations
from src.configurations import _CacheConfigurations


logger = logging.getLogger(__name__)


def left_push_queue(queue_name: str, key: str) -> bool:
    try:
        redis_client.lpush(queue_name, key)
        return True
    except Exception:
        return False


def right_pop_queue(queue_name: str) -> Any:
    if redis_client.llen(queue_name) > 0:
        return redis_client.rpop(queue_name)
    else:
        return None


def load_data_redis(key: str) -> Dict[str, Any]:
    data_dict = json.loads(redis_client.get(key))
    return data_dict


def save_data_file_job(job_id: str, directory: str, data: Any) -> bool:
    file_path = os.path.join(directory, f'{job_id}.json')
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return True


def save_data_redis_job(job_id: str, data: Any) -> bool:
    return save_data_dict_redis_job(job_id, data.__dict__)


def save_data_dict_redis_job(job_id: str, data: Dict[str, Any]) -> bool:
    data_dict = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            data_dict[k] = v.tolist()
        elif isinstance(v, Image.Image):
            filepath = os.path.join(_FileConfigurations().shared_volume, f'{job_id}.jpg')
            v.save(filepath)
            data_dict[k] = filepath
        else:
            data_dict[k] = v
    logger.info(f'job_id: {job_id}')
    redis_client.set(job_id, json.dumps(data_dict))
    return True


class SaveDataJob(BaseModel):
    job_id: str
    data: Any
    queue_name: str = CONSTANTS.REDIS_QUEUE
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
    enqueue: bool = False

    def __call__(self):
        save_data_jobs[self.job_id] = self
        logger.info(
            f'registered job: {self.job_id} in {self.__class__.__name__}')
        if isinstance(self.data, Dict):
            self.is_completed = save_data_dict_redis_job(self.job_id, self.data)
        else:
            self.is_completed = save_data_redis_job(self.job_id, self.data)
        if self.enqueue:
            self.is_completed = left_push_queue(self.queue_name, self.job_id)
        logger.info(f'completed save data: {self.job_id}')


def _save_data_job(data: Any,
                   job_id: str,
                   background_tasks: BackgroundTasks,
                   enqueue: bool = False) -> str:
    if _PlatformConfigurations().platform == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        task = SaveDataRedisJob(
            job_id=job_id,
            data=data,
            queue_name=_CacheConfigurations().queue_name,
            enqueue=enqueue)

    elif _PlatformConfigurations().platform == PLATFORM_ENUM.KUBERNETES.value:
        task = SaveDataRedisJob(
            job_id=job_id,
            data=data,
            queue_name=_CacheConfigurations().queue_name,
            enqueue=enqueue)

    elif _PlatformConfigurations().platform == PLATFORM_ENUM.TEST.value:
        task = SaveDataRedisJob(
            job_id=job_id,
            data=data,
            queue_name=_CacheConfigurations().queue_name,
            enqueue=enqueue)
    else:
        raise ValueError('platform must be chosen from constants.PLATFORM_ENUM')
    background_tasks.add_task(task)
    return job_id


save_data_jobs: Dict[str, SaveDataJob] = {}
