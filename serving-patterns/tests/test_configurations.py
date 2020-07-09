import os
from typing import Dict
from app.configurations import _ModelConfigurations
from app.constants import PREDICTION_TYPE, PREDICTION_RUNTIME


def test_ModelConfigurations():
    prediction_types = [e.value for e in PREDICTION_TYPE]
    prediction_runtimes = [e.value for e in PREDICTION_RUNTIME]

    assert os.path.exists(_ModelConfigurations().model_dir)
    assert os.path.exists(_ModelConfigurations().interface_filepath)
    assert isinstance(_ModelConfigurations().interface_dict, Dict)
    assert isinstance(_ModelConfigurations().model_name, str)
    assert isinstance(_ModelConfigurations().io, Dict)
    assert os.path.exists(_ModelConfigurations().model_filepath)
    assert _ModelConfigurations().prediction_runtime in prediction_runtimes
    assert _ModelConfigurations().prediction_type in prediction_types
