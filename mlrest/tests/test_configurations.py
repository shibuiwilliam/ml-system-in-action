import os
from typing import Dict
from app.configurations import _Configurations


def test_Configurations():
    assert os.path.exists(_Configurations().model_dir)
    assert os.path.exists(_Configurations().model_filepath)
    assert os.path.exists(_Configurations().interface_filepath)
    assert isinstance(_Configurations().interface_dict, Dict)
    assert isinstance(_Configurations().model_name, str)
    assert isinstance(_Configurations().io_interface, Dict)
