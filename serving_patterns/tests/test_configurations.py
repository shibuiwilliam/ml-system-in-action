import os
from typing import Dict, List
from app.configurations import _ModelConfigurations
from app.constants import PREDICTION_TYPE


def test_ModelConfigurations():
    prediction_types = [e.value for e in PREDICTION_TYPE]

    assert os.path.exists(_ModelConfigurations().model_dir)
    assert os.path.exists(_ModelConfigurations().interface_filepath)
    assert isinstance(_ModelConfigurations().interface_dict, Dict)
    assert isinstance(_ModelConfigurations().model_name, str)
    assert isinstance(_ModelConfigurations().io, Dict)
    assert isinstance(_ModelConfigurations().model_runners, List)
    assert isinstance(_ModelConfigurations().runner, str)
    assert _ModelConfigurations().prediction_type in prediction_types
