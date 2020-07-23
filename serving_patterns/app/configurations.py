import os
from app.constants import CONSTANTS, PHYSICAL_SAVE_DATA, PLATFORM_ENUM, DATA_TYPE
from app.ml.extract_interface import extract_interface_yaml
import logging

logger = logging.getLogger(__name__)


class _PlatformConfigurations():
    # can be docker_compose or kubernetes
    platform = os.getenv('PLATFORM', PLATFORM_ENUM.DOCKER_COMPOSE.value)
    platform = platform if platform in (
        PLATFORM_ENUM.DOCKER_COMPOSE.value,
        PLATFORM_ENUM.KUBERNETES.value) else PLATFORM_ENUM.TEST.value


class _CacheConfigurations():
    cache_host = os.getenv('CACHE_HOST', 'redis')
    cache_port = int(os.getenv('CACHE_PORT', 6379))
    queue_name = os.getenv('QUEUE_NAME', 'queue')


class _RedisCacheConfigurations(_CacheConfigurations):
    redis_db = int(os.getenv('REDIS_DB', 0))
    redis_decode_responses = bool(os.getenv('REDIS_DECODE_RESPONSES', True))


class _FileConfigurations():
    shared_volume = ''
    if _PlatformConfigurations().platform == PLATFORM_ENUM.DOCKER_COMPOSE.value:
        shared_volume = '/shared_volume'
    elif _PlatformConfigurations().platform == PLATFORM_ENUM.KUBERNETES.value:
        shared_volume = ''
    else:
        shared_volume = '/tmp'


class _FastAPIConfigurations():
    title = os.getenv('FASTAPI_TITLE', 'ML Rest')
    description = os.getenv('FASTAPI_DESCRIPTION', 'ML rest description')
    version = os.getenv('FASTAPI_VERSION', '0.1')
    app_name = os.getenv('APP_NAME', 'app.apps.app_web_single')


class _ModelConfigurations():
    model_dir = os.getenv('MODEL_DIR', CONSTANTS.MODEL_DIRECTORY)

    interface_filename = os.getenv('MODEL_INTERFACE')
    if interface_filename is None:
        logging.info('Environment variable "MODEL_INTERFACE" must be specified.')
        interface_filename = 'iris_svc_sklearn.yaml'
    interface_filepath = os.path.join(model_dir, interface_filename)

    interface_dict = extract_interface_yaml(interface_filepath)
    model_name = list(interface_dict.keys())[0]
    io = interface_dict[model_name]['data_interface']
    meta = interface_dict[model_name]['meta']

    models = meta['models']
    model_runners = [{os.path.join(os.getenv('MODEL_DIR', CONSTANTS.MODEL_DIRECTORY), k): v for k,v in m.items()} for m in models]
    prediction_type = meta['prediction_type']
    runner = meta['runner']

    options= interface_dict[model_name].get('options', None)

    physical_save_data = os.getenv('PHYSICAL_SAVE_DATA', PHYSICAL_SAVE_DATA.SAVE)


logger.info(f'model configurations: {_ModelConfigurations.__dict__}')
logger.info(f'fastapi configurations: {_FastAPIConfigurations.__dict__}')
logger.info(f'platform configurations: {_PlatformConfigurations.__dict__}')
