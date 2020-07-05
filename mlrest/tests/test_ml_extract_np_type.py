import pytest
import numpy as np
from app.ml import extract_np_type


@pytest.mark.parametrize(
    ('type_name', 'expected'),
    [('int', np.int), ('float', np.float), ('float64', np.float64), ('object', None)]
)
def test_type_name_to_np_type(type_name, expected):
    result = extract_np_type.type_name_to_np_type(type_name)
    assert result == expected
