import os
from src.configurations.constants import PLATFORM_ENUM
from src.configurations.configurations import _PlatformConfigurations
from src.app.constants import CONSTANTS, PHYSICAL_SAVE_DATA, DATA_TYPE
from src.app.ml.extract_interface import extract_interface_yaml
import logging

logger = logging.getLogger(__name__)



class _FastAPIConfigurations():
    title = os.getenv('FASTAPI_TITLE', 'ServingPattern')
    description = os.getenv('FASTAPI_DESCRIPTION', 'machine learning system serving patterns')
    version = os.getenv('FASTAPI_VERSION', '0.1')
    app_name = os.getenv('APP_NAME', 'src.app.apps.app_web_single:app')


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

    options = interface_dict[model_name].get('options', None)

    physical_save_data = os.getenv('PHYSICAL_SAVE_DATA', PHYSICAL_SAVE_DATA.SAVE)


logger.info(f'model configurations: {_ModelConfigurations.__dict__}')
logger.info(f'fastapi configurations: {_FastAPIConfigurations.__dict__}')
