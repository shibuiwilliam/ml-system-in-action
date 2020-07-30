from typing import Dict, Any, List, Tuple
from src.app.constants import CONSTANTS


def convert_dict(data: Dict[str, Any], none_default: Any) -> Dict[str, str]:
    if not isinstance(data, Dict):
        return None
    _data = {}
    for k, v in data.items():
        if v is None:
            _data[k] = none_default
        elif isinstance(v, List) or isinstance(v, Tuple):
            if isinstance(v[0], int):
                _type = 'int'
            elif isinstance(v[0], float):
                _type = 'float'
            elif isinstance(v[0], str):
                _type = 'str'
            else:
                _type = 'None'
            _data[k] = f'list_{_type}_' + \
                CONSTANTS.SEPARATOR.join([str(_v) for _v in v])
        else:
            _data[k] = v
    return _data


def revert_cache(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, Dict):
        return None
    _data = {}
    for k, v in data.items():
        if isinstance(v, str) and v.startswith('list_'):
            _v = v.split('_')
            _type = _v[1]
            _value = _v[2]
            if _type == 'int':
                _data[k] = [int(n) for n in _value.split(CONSTANTS.SEPARATOR)]
            elif _type == 'float':
                _data[k] = [
                    float(n) for n in _value.split(
                        CONSTANTS.SEPARATOR)]
            elif _type == 'str':
                _data[k] = _value.split(CONSTANTS.SEPARATOR)
        else:
            _data[k] = v
    return _data
