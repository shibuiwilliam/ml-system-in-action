import pytest
from app.middleware import redis_utils


@pytest.mark.parametrize(
    ('data', 'none_default', 'expected_data', 'expected_prediction'),
    [({'data': [1.0, -1.0], 'prediction': None}, -1, 'list_float_1.0;-1.0', -1),
     ({'data': [1, -1], 'prediction': 0}, -1, 'list_int_1;-1', 0),
     ({'data': ['a', 'bb'], 'prediction': 'z'}, 'x', 'list_str_a;bb', 'z')]
)
def test_convert_dict(data, none_default, expected_data, expected_prediction):
    result = redis_utils.convert_dict(data, none_default)
    assert result['data'] == expected_data
    assert result['prediction'] == expected_prediction


@pytest.mark.parametrize(
    ('data', 'none_default', 'expected_data'),
    [([1, 2], '', None),
     ('a', '', None),
     (None, '', None)]
)
def test_convert_dict_none(data, none_default, expected_data):
    result = redis_utils.convert_dict(data, none_default)
    assert result == expected_data


@pytest.mark.parametrize(
    ('data', 'expected'),
    [({'data': 'list_float_1.0;-1.0', 'prediction': -1.1}, {'data': [1.0, -1.0], 'prediction': -1.1}),
     ({'data': 'list_int_1;-1', 'prediction': 0}, {'data': [1, -1], 'prediction': 0}),
     ({'data': 'list_str_a;bb', 'prediction': 'z'}, {'data': ['a', 'bb'], 'prediction': 'z'})]
)
def test_revert_cache(data, expected):
    result = redis_utils.revert_cache(data)
    assert result['data'] == expected['data']
    assert result['prediction'] == expected['prediction']


@pytest.mark.parametrize(
    ('data', 'none_default', 'expected_data'),
    [([1, 2], '', None),
     ('a', '', None),
     (None, '', None)]
)
def test_revert_cache(data, none_default, expected_data):
    result = redis_utils.revert_cache(data)
    assert result == expected_data
