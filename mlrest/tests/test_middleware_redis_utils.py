import pytest
from app.middleware import redis_utils


@pytest.mark.parametrize(
    ('data', 'none_default'),
    [({'data': [1.0, -1.0], 'prediction': None}, -1)]
)
def test_convert_dict(data, none_default):
    result = redis_utils.convert_dict(data, none_default)
    assert result['data'] == 'list_float_1.0;-1.0'
    assert result['prediction'] == -1
