import os
from typing import Dict
from app.configurations import _Configurations
from app.constants import PREDICTION_TYPE, PREDICTION_RUNTIME


def test_Configurations():
    prediction_types = [e.value for e in PREDICTION_TYPE]
    prediction_runtimes = [e.value for e in PREDICTION_RUNTIME]

    assert os.path.exists(_Configurations().model_dir)
    assert os.path.exists(_Configurations().interface_filepath)
    assert isinstance(_Configurations().interface_dict, Dict)
    assert isinstance(_Configurations().model_name, str)
    assert isinstance(_Configurations().io, Dict)
    assert os.path.exists(_Configurations().model_filepath)
    assert _Configurations().prediction_runtime in prediction_runtimes
    assert _Configurations().prediction_type in prediction_types
