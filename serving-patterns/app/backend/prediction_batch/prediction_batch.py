import logging
from time import sleep
import asyncio
from concurrent.futures import ProcessPoolExecutor

from app.constants import CONSTANTS
from app.api._predict import _predict_from_redis_cache
from app.jobs import store_data_job
from app.middleware.profiler import do_cprofile


logger = logging.getLogger('prediction_batch')


@do_cprofile
def _run_prediction(job_id: str):
    data = _predict_from_redis_cache(job_id)
    if data is not None:
        store_data_job.save_data_redis_job(job_id, data)
    else:
        store_data_job.left_push_queue(CONSTANTS.REDIS_QUEUE, job_id)


def _trigger_prediction_if_queue():
    job_id = store_data_job.right_pop_queue(CONSTANTS.REDIS_QUEUE)
    logger.info(f'predict job_id: {job_id}')
    if job_id is not None:
        _run_prediction(job_id)


def _loop():
    while True:
        sleep(1)
        _trigger_prediction_if_queue()


def prediction_loop(num_procs: int = 2):
    executor = ProcessPoolExecutor(num_procs)
    loop = asyncio.get_event_loop()

    for _ in range(num_procs):
        asyncio.ensure_future(loop.run_in_executor(executor, _loop))

    loop.run_forever()
