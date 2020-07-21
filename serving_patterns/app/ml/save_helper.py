import os
from typing import Dict, List
import yaml
import json
import joblib

from app.constants import PREDICTION_TYPE, MODEL_RUNTIME, DATA_TYPE


def load_labels(label_filepath: str):
    with open(label_filepath, 'r') as f:
        return json.load(f)


def dump_sklearn(model, name: str):
    joblib.dump(model, name)


def save_interface(model_name: str,
                   model_dir: str,
                   interface_filename: str,
                   input_shape: List[int],
                   input_type: str,
                   output_shape: List[int],
                   output_type: str,
                   data_type: DATA_TYPE,
                   models: List[Dict[str, MODEL_RUNTIME]],
                   prediction_type: PREDICTION_TYPE,
                   runner: str,
                   **kwargs: Dict) -> None:
    os.makedirs(model_dir, exist_ok=True)
    if not (interface_filename.endswith('yaml') or interface_filename.endswith('yml')):
        interface_filename = f'{interface_filename}.yaml'
    filepath = os.path.join(model_dir, interface_filename)
    _models = [{k:v.value for k,v in m.items()} for m in models]
    with open(filepath, 'w') as f:
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