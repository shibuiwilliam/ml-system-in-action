from sklearn.model_selection import train_test_split
import argparse
import os
import numpy as np
from typing import Dict, List, Any

from src.app.ml.save_helper import load_data


DATA_DIR = './src/app/ml/iris/data/'
DATA_FILEPATH = os.path.join(DATA_DIR, 'iris_data.csv')


def split_dataset(
        data: List[List[Any]],
        target: List[Any],
        test_size=0.3) -> Dict[str, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        target,
        shuffle=True,
        test_size=test_size)
    x_train = np.array(x_train).astype('float32')
    y_train = np.array(y_train).astype('float32')
    x_test = np.array(x_test).astype('float32')
    y_test = np.array(y_test).astype('float32')
    return {'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test}


def main():
    parser = argparse.ArgumentParser(description='iris dataset preparation.')
    parser.add_argument('--test_rate', required=False, type=float, default=float(os.getenv('TEST_RATE', 0.3)))
    args = parser.parse_args()

    _full_data = load_data(DATA_FILEPATH)
    _data = [d[:4] for d in _full_data]
    _target = [d[4] for d in _full_data]
    data = split_dataset(_data, _target, args.test_rate)

    for k, v in data.items():
        np.save(os.path.join(DATA_DIR, f'{k}.npy'), v)


if __name__ == '__main__':
    main()
