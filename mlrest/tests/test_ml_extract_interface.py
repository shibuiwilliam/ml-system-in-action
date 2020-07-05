import pytest
from app.ml import extract_interface
from typing import Tuple


filepath = 'app/ml/models/iris_svc/iris_svc.yaml'


@pytest.mark.parametrize(
    ('filepath'),
    [(filepath)]
)
def test_extract_interface_yaml(filepath):
    interface = extract_interface.extract_interface_yaml(filepath)
    model_name = list(interface.keys())[0]
    input_shape = interface[model_name]['input_shape']
    input_type = interface[model_name]['input_type']
    output_shape = interface[model_name]['output_shape']
    output_type = interface[model_name]['output_type']
    assert isinstance(input_shape, Tuple)
    assert isinstance(input_type, str)
    assert isinstance(output_shape, Tuple)
    assert isinstance(output_type, str)
