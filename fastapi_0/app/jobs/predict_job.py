import os
from typing import Dict, Any
import logging
import numpy as np
from pydantic import BaseModel
import json

from constants import CONSTANTS
from middleware import redis
from . import save_data_job

logger = logging.getLogger(__name__)


class PredictJob(BaseModel):

    def __call__(self):
        pass


class PredictFromFileJob(PredictJob):
    job_id: str
    directory: str
    predictor: Any
    is_completed: bool = False

    def __call__(self):
        predict_jobs[self.job_id] = self
        logger.info(
            f'registered job: {self.job_id} in {self.__class__.__name__}')
        file_path = os.path.join(self.directory, f'{self.job_id}.json')
        while True:
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'r') as f:
                data_dict = json.load(f)
            if data_dict['prediction'] != CONSTANTS.PREDICTION_DEFAULT:
                break
            _proba = self.predictor.predict_proba_from_dict(data_dict)
            data_dict['prediction'] = int(np.argmax(_proba[0]))
            data_dict['prediction_proba'] = _proba.tolist()
            save_data_job.save_data_file_job(
                self.job_id, self.directory, data_dict)
            self.is_completed = True
            break
        logger.info(f'completed prediction: {self.job_id}')


class PredictFromRedisJob(PredictJob):
    job_id: str
    predictor: Any
    is_completed: bool = False

    def __call__(self):
        predict_jobs[self.job_id] = self
        logger.info(
            f'registered job: {self.job_id} in {self.__class__.__name__}')
        while True:
            data_dict = redis.redis_connector.hgetall(self.job_id)
            logger.info(data_dict)
            if data_dict is None:
                continue
            if 'prediction' in data_dict.keys():
                if int(data_dict['prediction']) !=\
                        CONSTANTS.PREDICTION_DEFAULT:
                    break
            _data_dict = {}
            for k, v in data_dict.items():
                if v.startswith('list_'):
                    _v = v.split('_')
                    _type = _v[1]
                    _value = _v[2]
                    if _type == 'int':
                        _data_dict[k] = [int(n) for n in _value.split(
                            CONSTANTS.SEPARATOR)]
                    elif _type == 'float':
                        _data_dict[k] = [float(n) for n in _value.split(
                            CONSTANTS.SEPARATOR)]
                    elif _type == 'str':
                        _data_dict[k] = _value.split(CONSTANTS.SEPARATOR)
                else:
                    _data_dict[k] = v
            logger.info(_data_dict)
            _proba = self.predictor.predict_proba_from_dict(_data_dict)
            _data_dict['prediction'] = int(np.argmax(_proba[0]))
            _data_dict['prediction_proba'] = _proba[0].tolist()
            save_data_job.save_data_redis_job(self.job_id, _data_dict)
            self.is_completed = True
            break
        logger.info(f'completed prediction: {self.job_id}')


predict_jobs: Dict[str, PredictJob] = {}
