import os
from typing import Dict, Any
import logging
import numpy as np
from pydantic import BaseModel
import json

from app.constants import CONSTANTS
from app.middleware import redis, redis_utils
from . import save_data_job

logger = logging.getLogger(__name__)


def predict_from_file(job_id: str,
                      directory: str,
                      predictor: Any) -> Dict[str, Any]:
    file_path = os.path.join(directory, f'{job_id}.json')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        data_dict = json.load(f)
    if 'prediction_proba' in data_dict.keys() and \
            data_dict['prediction_proba'] != CONSTANTS.NONE_DEFAULT_LIST:
        data_dict['prediction'] = int(
            np.argmax(
                data_dict['prediction_proba']
                if isinstance(data_dict['prediction_proba'], np.ndarray)
                else np.array(data_dict['prediction_proba'])))
    if 'prediction' in data_dict.keys() and \
            data_dict['prediction'] != CONSTANTS.NONE_DEFAULT:
        return data_dict
    else:
        if 'data' not in data_dict.keys() or data_dict.get('data', None) is None:
            return None
        _proba = predictor.predict_proba_from_dict(data_dict)
        data_dict['prediction'] = int(np.argmax(_proba[0]))
        data_dict['prediction_proba'] = _proba[0].tolist()
        return data_dict


def predict_from_redis_cache(job_id: str, predictor: Any) -> Dict[str, Any]:
    data_dict = redis.redis_connector.hgetall(job_id)
    logger.info(data_dict)
    if data_dict is None:
        return None
    _data_dict = redis_utils.revert_cache(data_dict)
    if 'prediction_proba' in _data_dict.keys() and \
            _data_dict['prediction_proba'] != CONSTANTS.NONE_DEFAULT_LIST:
        _data_dict['prediction'] = int(
            np.argmax(np.array(_data_dict['prediction_proba'])))
    if 'prediction' in _data_dict.keys() and \
            _data_dict['prediction'] != CONSTANTS.NONE_DEFAULT:
        return _data_dict
    else:
        if 'data' not in _data_dict.keys() or _data_dict.get('data', None) is None:
            return None
        logger.info(_data_dict)
        _proba = predictor.predict_proba_from_dict(_data_dict)
        _data_dict['prediction'] = int(np.argmax(_proba[0]))
        _data_dict['prediction_proba'] = _proba[0].tolist()
        return _data_dict


class PredictJob(BaseModel):
    job_id: str
    predictor: Any
    is_completed: bool = False

    def __call__(self):
        pass


class PredictFromFileJob(PredictJob):
    directory: str

    def __call__(self):
        predict_jobs[self.job_id] = self
        logger.info(
            f'registered job: {self.job_id} in {self.__class__.__name__}')
        while True:
            data_dict = predict_from_file(
                self.job_id, self.directory, self.predictor)
            if data_dict is not None:
                break
        save_data_job.save_data_file_job(
            self.job_id, self.directory, data_dict)
        self.is_completed = True
        logger.info(f'completed prediction: {self.job_id}')


class PredictFromRedisJob(PredictJob):

    def __call__(self):
        predict_jobs[self.job_id] = self
        logger.info(
            f'registered job: {self.job_id} in {self.__class__.__name__}')
        while True:
            data_dict = predict_from_redis_cache(self.job_id, self.predictor)
            if data_dict is not None:
                break
        save_data_job.save_data_redis_job(self.job_id, data_dict)
        self.is_completed = True
        logger.info(f'completed prediction: {self.job_id}')


predict_jobs: Dict[str, PredictJob] = {}
