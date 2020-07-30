import os
from typing import Dict, List
import yaml
import json
import joblib

from src.app.constants import PREDICTION_TYPE, MODEL_RUNTIME, DATA_TYPE


def load_labels(label_filepath: str):
    with open(label_filepath, 'r') as f:
        return json.load(f)


def dump_sklearn(model, name: str):
    joblib.dump(model, name)


def save_interface(model_name: str,
                   interface_filepath: str,
                   input_shape: List[int],
                   input_type: str,
                   output_shape: List[int],
                   output_type: str,
                   data_type: DATA_TYPE,
                   models: List[Dict[str, MODEL_RUNTIME]],
                   prediction_type: PREDICTION_TYPE,
                   runner: str,
                   **kwargs: Dict) -> None:
    if not (interface_filepath.endswith('yaml') or interface_filepath.endswith('yml')):
        interface_filepath = f'{interface_filepath}.yaml'
    _models = [{k:v.value for k,v in m.items()} for m in models]
    with open(interface_filepath, 'w') as f:
        f.write(yaml.dump({
            model_name: {
                'data_interface': {
                    'input_shape': input_shape,
                    'input_type': input_type,
                    'output_shape': output_shape,
                    'output_type': output_type,
                    'data_type': data_type.value
                },
                'meta': {
                    'models': _models,
                    'prediction_type': prediction_type.value,
                    'runner': runner,
                }, 
                'options': kwargs,
            }
        }, default_flow_style=False))