from typing import Dict, Any
import logging
from pydantic import BaseModel

from app.ml.base_predictor import BaseData
from . import store_data_job

logger = logging.getLogger(__name__)


def predict_from_redis_cache(
        job_id: str,
        predictor: Any,
        baseData: Any,
        baseDataExtentions: Any) -> BaseData:
    data_dict = store_data_job.load_data_redis(job_id)
    logger.info(data_dict)
    if data_dict is None:
        return None
    data = baseData(**data_dict)
    data_extension = baseDataExtentions(data)
    data_extension.convert_input_data_to_np_data()
    data.output = predictor.predict(data)
    return data


class PredictJob(BaseModel):
    job_id: str
    predictor: Any
    baseData: Any
    baseDataExtentions: Any
    is_completed: bool = False

    def __call__(self):
        pass


class PredictFromRedisJob(PredictJob):

    def __call__(self):
        predict_jobs[self.job_id] = self
        logger.info(f'registered job: {self.job_id} in {self.__class__.__name__}')
        while True:
            data = predict_from_redis_cache(
                self.job_id,
                self.predictor,
                self.baseData,
                self.baseDataExtentions)
            if data is not None:
                break
        store_data_job.save_data_redis_job(self.job_id, data)
        self.is_completed = True
        logger.info(f'completed prediction: {self.job_id}')


predict_jobs: Dict[str, PredictJob] = {}
