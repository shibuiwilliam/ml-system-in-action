import logging
from time import sleep
from typing import Any
import asyncio
from concurrent.futures import ProcessPoolExecutor

from app.constants import CONSTANTS
from app.ml.base_predictor import BaseData
from app.ml.active_predictor import Data, DataExtension, active_predictor
from app.jobs.store_data_job import (
    load_data_redis,
    save_data_redis_job,
    left_push_queue,
    right_pop_queue)


logger = logging.getLogger('prediction_batch')


def _predict_from_redis_cache(
        job_id: str,
        predictor: Any,
        baseData: Any,
        baseDataExtentions: Any) -> BaseData:
    data_dict = load_data_redis(job_id)
    if data_dict is None:
        return None
    data = baseData(**data_dict)
    data_extension = baseDataExtentions(data)
    data_extension.convert_input_data_to_np_data()
    data.output = predictor.predict(data)
    logger.info(f'prediction: {job_id} {data.__dict__}')
    return data


def _run_prediction_if_queue():
    job_id = right_pop_queue(CONSTANTS.REDIS_QUEUE)
    logger.info(f'predict job_id: {job_id}')
    if job_id is not None:
        data = _predict_from_redis_cache(
            job_id,
            active_predictor,
            Data,
            DataExtension)
        if data is not None:
            save_data_redis_job(job_id, data)
        else:
            left_push_queue(CONSTANTS.REDIS_QUEUE, job_id)


def _loop():
    while True:
        sleep(1)
        _run_prediction_if_queue()


def prediction_loop():
    executor = ProcessPoolExecutor(2)
    loop = asyncio.get_event_loop()

    asyncio.ensure_future(loop.run_in_executor(executor, _loop))

    loop.run_forever()
