import numpy as np
from typing import Any


def type_name_to_np_type(type_name: str) -> Any:
    if type_name == 'int':
        return np.int
    elif type_name == 'int8':
        return np.int8
    elif type_name == 'int16':
        return np.int16
    elif type_name == 'int32':
        return np.int32
    elif type_name == 'int64':
        return np.int64
    elif type_name == 'float':
        return np.float
    elif type_name == 'float16':
        return np.float16
    elif type_name == 'float32':
        return np.float32
    elif type_name == 'float64':
        return np.float64
    else:
        return None
