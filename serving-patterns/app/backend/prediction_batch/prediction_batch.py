import logging
from time import sleep
import asyncio
from concurrent.futures import ProcessPoolExecutor

from app.constants import CONSTANTS
from app.ml.active_predictor import Data, DataExtension, active_predictor
from app.jobs.predict_job import predict_from_redis_cache
from app.jobs import store_data_job

logger = logging.getLogger('prediction_batch')


def _run_prediction_if_queue():
    job_id = store_data_job.right_pop_queue(CONSTANTS.REDIS_QUEUE)
    logger.info(f'predict job_id: {job_id}')
    if job_id is not None:
        data = predict_from_redis_cache(
            job_id,
            active_predictor,
            Data,
            DataExtension)
        if data is not None:
            store_data_job.save_data_redis_job(job_id, data)
        else:
            store_data_job.left_push_queue(CONSTANTS.REDIS_QUEUE, job_id)


def _loop():
    while True:
        sleep(1)
        _run_prediction_if_queue()


def prediction_loop(num_procs: int = 2):
    executor = ProcessPoolExecutor(num_procs)
    loop = asyncio.get_event_loop()

    for _ in range(num_procs):
        asyncio.ensure_future(loop.run_in_executor(executor, _loop))

    loop.run_forever()
