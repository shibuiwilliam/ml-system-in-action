import os
from app.constants import CONSTANTS, PHYSICAL_SAVE_DATA, PLATFORM_ENUM
from app.ml import extract_interface
import logging

logger = logging.getLogger(__name__)


class _PlatformConfigurations():
    # can be docker_compose or kubernetes
    platform = os.getenv('PLATFORM', PLATFORM_ENUM.DOCKER_COMPOSE.value)
    platform = platform if platform in (
        PLATFORM_ENUM.DOCKER_COMPOSE.value,
        PLATFORM_ENUM.KUBERNETES.value) else PLATFORM_ENUM.DOCKER_COMPOSE.value


class _FastAPIConfigurations():
    title = os.getenv('FASTAPI_TITLE', 'ML Rest')
    description = os.getenv('FASTAPI_DESCRIPTION', 'ML rest description')
    version = os.getenv('FASTAPI_VERSION', '0.1')
    app_name = os.getenv('APP_NAME', 'app.apps.app_web_single')


class _ModelConfigurations():
    model_dir = os.getenv('MODEL_DIR', CONSTANTS.MODEL_DIRECTORY)

    interface_filename = os.getenv('MODEL_INTERFACE', 'iris_svc_sklearn.yaml')
    interface_filepath = os.path.join(model_dir, interface_filename)

    interface_dict = extract_interface.extract_interface_yaml(interface_filepath)
    model_name = list(interface_dict.keys())[0]
    io = interface_dict[model_name]['interface']
    meta = interface_dict[model_name]['meta']

    model_filename = meta['model_filename']
    model_filepath = os.path.join(model_dir, model_filename)
    prediction_runtime = meta['prediction_runtime']
    prediction_type = meta['prediction_type']

    physical_save_data = os.getenv('PHYSICAL_SAVE_DATA', PHYSICAL_SAVE_DATA.SAVE)


logger.info(f'model configurations: {_ModelConfigurations.__dict__}')
logger.info(f'fastapi configurations: {_FastAPIConfigurations.__dict__}')
logger.info(f'platform configurations: {_PlatformConfigurations.__dict__}')
