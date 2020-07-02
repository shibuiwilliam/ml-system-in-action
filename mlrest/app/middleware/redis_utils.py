from typing import Dict, Any, List, Tuple
from app.constants import CONSTANTS


def convert_dict(data: Dict[str, Any], none_default: str) -> Dict[str, str]:
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
