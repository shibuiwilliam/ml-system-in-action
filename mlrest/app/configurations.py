import os
import yaml
from typing import Dict, Any
from app.constants import CONSTANTS, PHYSICAL_SAVE_DATA
from app.ml import extract_interface
import logging

logger = logging.getLogger(__name__)


class _Configurations():
    model_dir = os.getenv(
        'MODEL_DIR',
        os.path.join(CONSTANTS.MODEL_DIRECTORY, 'iris_svc')
    )

    model_filename = os.getenv('IRIS_MODEL', 'iris_svc.pkl')
    model_filepath = os.path.join(model_dir, model_filename)

    interface_filename = os.getenv('IRIS_INTERFACE', 'iris_svc.yaml')
    interface_filepath = os.path.join(model_dir, interface_filename)
    interface_dict = extract_interface.extract_interface_yaml(interface_filepath)
    model_name = list(interface_dict.keys())[0]
    io_interface = interface_dict[model_name]

    physical_save_data = os.getenv('PHYSICAL_SAVE_DATA', PHYSICAL_SAVE_DATA.SAVE)


logger.info(f'configurations: {_Configurations.__dict__}')
