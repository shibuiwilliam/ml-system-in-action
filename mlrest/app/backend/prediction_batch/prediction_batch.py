import logging
from time import sleep
from typing import Any

from app.constants import CONSTANTS
from app.ml.base_predictor import BaseData
from app.ml.active_predictor import Data, DataExtension, active_predictor
from app.jobs.store_data_job import (
    load_data_redis,
    save_data_redis_job,
    right_pop_queue)


logger = logging.getLogger('prediction_batch')
logger.info('initializings prediction batch')


def _predict_from_redis_cache(
        job_id: str,
        predictor: Any,
        baseData: Any,
        baseDataExtentions: Any) -> BaseData:
    data_dict = load_data_redis(job_id)
    logger.info(data_dict)
    if data_dict is None:
        return None
    data = baseData(**data_dict)
    data_extension = baseDataExtentions(data)
    data_extension.convert_input_data_to_np_data()
    data.output = predictor.predict(data)
    return data


def prediction_loop():
    while True:
        sleep(2)
        job_id = right_pop_queue(CONSTANTS.REDIS_QUEUE)
        logger.info(job_id)
        if job_id is not None:
            data = _predict_from_redis_cache(
                job_id,
                active_predictor,
                Data,
                DataExtension)
            save_data_redis_job(job_id, data)
