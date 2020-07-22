from typing import List, Tuple, Dict, Union
import numpy as np

def floats_almost_equal(X: Union[List[float], Tuple[float], np.ndarray],
                        Y: Union[List[float], Tuple[float], np.ndarray]):
    return all(round(x-y, 5) == 0 for x,y in zip(X, Y))


def nested_floats_almost_equal(X: Union[List[List[float]], Tuple[Tuple[float]], np.ndarray],
                               Y: Union[List[List[float]], Tuple[Tuple[float]], np.ndarray]):
    return all((round(_x-_y, 5) == 0 for _x,_y in zip(x,y)) for x,y in zip(X, Y))

