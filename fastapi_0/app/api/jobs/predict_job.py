import os
from typing import Dict, Any
import logging
import numpy as np
from pydantic import BaseModel
import json

from constants import CONSTANTS

logger = logging.getLogger(__name__)


class PredictJob(BaseModel):
    job_id: str
    file_path: str
    predictor: Any
    is_completed: bool = False

    def __call__(self):
        predict_jobs[self.job_id] = self
        logger.info(
            f'registered job: {self.job_id} in {self.__class__.__name__}')
        while True:
            if not os.path.exists(self.file_path):
                continue
            with open(self.file_path, 'r') as f:
                iris_dict = json.load(f)
            if iris_dict['prediction'] != CONSTANTS.PREDICTION_DEFAULT:
                break
            _proba = self.predictor.predict_proba_from_dict(iris_dict)
            iris_dict['prediction'] = int(np.argmax(_proba[0]))
            iris_dict['prediction_proba'] = _proba.tolist()
            with open(self.file_path, 'w') as f:
                json.dump(iris_dict, f)
            self.is_completed = True
            break
        logger.info(f'completed prediction: {self.job_id}')


predict_jobs: Dict[str, PredictJob] = {}
