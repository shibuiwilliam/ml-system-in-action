from typing import Dict, Any, List
from fastapi import BackgroundTasks
import uuid
import numpy as np
import logging

from src.middleware.profiler import do_cprofile
from src.middleware.redis_client import redis_client
from src.jobs import store_data_job
from src.constants import PLATFORM_ENUM
from src.configurations import _PlatformConfigurations
from src.app.ml.active_predictor import Data, DataInterface, DataConverter, active_predictor
from src.configurations import _CacheConfigurations


logger = logging.getLogger(__name__)


@do_cprofile
def __predict(data: Data):
    input_np = DataConverter.convert_input_data_to_np(data.input_data)
    output_np = active_predictor.predict(input_np)
    reshaped_output_nps = DataConverter.reshape_output(output_np)
    data.prediction = reshaped_output_nps.tolist()
    logger.info(f'prediction: {data.__dict__}')


@do_cprofile
def __predict_label(data: Data,
                    __predict_callable: callable = __predict) -> Dict[str, float]:
    __predict_callable(data)
    argmax = int(np.argmax(np.array(data.prediction)[0]))
    return {data.labels[argmax]: data.prediction[0][argmax]}


def _predict_from_redis_cache(job_id: str,
                              data_class: callable = Data) -> Data:
    data_dict = store_data_job.load_data_redis(job_id)
    if data_dict is None:
        return None
    data = data_class(**data_dict)
    __predict(data)
    return data


def _labels(data_class: callable = Data) -> Dict[str, List[str]]:
    return {'labels': data_class().labels}


def _test(data: Data = Data(),
          __predict_callable: callable = __predict) -> Dict[str, int]:
    data.input_data = data.test_data
    __predict_callable(data)
    return {'prediction': data.prediction}


def _test_label(
        data: Data = Data(),
        __predict_callable: callable = __predict_label) -> Dict[str, Dict[str, float]]:
    data.input_data = data.test_data
    label_proba = __predict_callable(data)
    return {'prediction': label_proba}


def _predict(
        data: Data,
        job_id: str,
        background_tasks: BackgroundTasks) -> Dict[str, List[float]]:
    __predict(data)
    store_data_job._save_data_job(data, job_id, background_tasks, False)
    return {'prediction': data.prediction, 'job_id': job_id}


def _predict_label(
        data: Data,
        job_id: str,
        background_tasks: BackgroundTasks = BackgroundTasks()) -> Dict[str, Dict[str, float]]:
    label_proba = __predict_label(data)
    store_data_job._save_data_job(data, job_id, background_tasks, False)
    return {'prediction': label_proba, 'job_id': job_id}


async def _predict_async_post(
        data: Data,
        job_id: str,
        background_tasks: BackgroundTasks) -> Dict[str, str]:
    store_data_job._save_data_job(data, job_id, background_tasks, True)
    return {'job_id': job_id}


@do_cprofile
def _predict_async_get(job_id: str) -> Dict[str, List[float]]:
    result = {job_id: {'prediction': []}}
    if _PlatformConfigurations().platform == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        data_dict = store_data_job.load_data_redis(job_id)
        result[job_id]['prediction'] = data_dict['prediction']
        return result

    elif _PlatformConfigurations().platform == PLATFORM_ENUM.KUBERNETES.value:
        data_dict = store_data_job.load_data_redis(job_id)
        result[job_id]['prediction'] = data_dict['prediction']
        return result

    elif _PlatformConfigurations().platform == PLATFORM_ENUM.TEST.value:
        data_dict = store_data_job.load_data_redis(job_id)
        result[job_id]['prediction'] = data_dict['prediction']
        return result

    else:
        return result


@do_cprofile
def _predict_async_get_label(job_id: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    result = {job_id: {'prediction': []}}
    if _PlatformConfigurations().platform == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        data_dict = store_data_job.load_data_redis(job_id)
        if result[job_id]['prediction'] is None:
            result[job_id]['prediction'] = data_dict['prediction']
            return result
        argmax = int(np.argmax(np.array(data_dict['prediction'])[0]))
        result[job_id]['prediction'] = {data_dict['labels'][argmax]: data_dict['prediction'][0][argmax]}
        return result

    elif _PlatformConfigurations().platform == PLATFORM_ENUM.KUBERNETES.value:
        data_dict = store_data_job.load_data_redis(job_id)
        if result[job_id]['prediction'] is None:
            result[job_id]['prediction'] = data_dict['prediction']
            return result
        argmax = int(np.argmax(np.array(data_dict['prediction'])[0]))
        result[job_id]['prediction'] = {data_dict['labels'][argmax]: data_dict['prediction'][0][argmax]}
        return result

    elif _PlatformConfigurations().platform == PLATFORM_ENUM.TEST.value:
        data_dict = store_data_job.load_data_redis(job_id)
        if result[job_id]['prediction'] is None:
            result[job_id]['prediction'] = data_dict['prediction']
            return result
        argmax = int(np.argmax(np.array(data_dict['prediction'])[0]))
        result[job_id]['prediction'] = {data_dict['labels'][argmax]: data_dict['prediction'][0][argmax]}
        return result

    else:
        return result
