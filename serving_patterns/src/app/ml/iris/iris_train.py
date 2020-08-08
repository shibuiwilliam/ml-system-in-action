from sklearn import svm, tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
import numpy as np
import argparse

from src.app.constants import PREDICTION_TYPE, MODEL_RUNTIME, DATA_TYPE
from src.app.ml.save_helper import save_interface, dump_sklearn


MODEL_DIR = './models/'
DATA_DIR = './src/app/ml/iris/data/'
LABEL_FILEPATH = os.path.join(DATA_DIR, 'iris_label.csv')
X_TRAIN_NPY = os.path.join(DATA_DIR, 'x_train.npy')
Y_TRAIN_NPY = os.path.join(DATA_DIR, 'y_train.npy')


def define_svc_pipeline() -> Pipeline:
    steps = [
        ('normalize', StandardScaler()),
        ('svc', svm.SVC(probability=True))
    ]
    pipeline = Pipeline(steps=steps)
    return pipeline


def define_tree_pipeline() -> Pipeline:
    steps = [
        ('normalize', StandardScaler()),
        ('tree', tree.DecisionTreeClassifier())
    ]
    pipeline = Pipeline(steps=steps)
    return pipeline


def train_model(model, x: np.ndarray, y: np.ndarray):
    model.fit(x, y)


def save_onnx(model,
              filepath: str):
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(filepath, 'wb') as f:
        f.write(onx.SerializeToString())


def main():
    parser = argparse.ArgumentParser(description='iris dataset svc trainer.')
    parser.add_argument(
        '--save_model_name',
        required=False,
        type=str,
        default=str(os.getenv('SAVE_MODEL_NAME', 'iris_svc')))
    parser.add_argument(
        '--save_format',
        required=False,
        choices=['sklearn', 'onnx'],
        default=str(os.getenv('SAVE_FORMAT', 'sklearn')))
    parser.add_argument(
        '--ml_model',
        required=False,
        choices=['svc', 'tree'],
        default=str(os.getenv('ML_MODEL', 'svc')))
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    x_train = np.load(X_TRAIN_NPY)
    y_train = np.load(Y_TRAIN_NPY)

    if args.ml_model == 'svc':
        pipeline = define_svc_pipeline()
    elif args.ml_model == 'tree':
        pipeline = define_tree_pipeline()
    else:
        pass

    train_model(pipeline, x_train, y_train)

    modelname = args.save_model_name

    if args.save_format == 'sklearn':
        model_filename = f'{modelname}.pkl'
        sklearn_interface_filename = f'{modelname}_sklearn.yaml'
        dump_sklearn(pipeline, os.path.join(MODEL_DIR, model_filename))
        save_interface(modelname,
                       os.path.join(MODEL_DIR, sklearn_interface_filename),
                       [1, 4],
                       str(x_train.dtype).split('.')[-1],
                       [1, 3],
                       'float32',
                       DATA_TYPE.ARRAY,
                       [{model_filename: MODEL_RUNTIME.SKLEARN}],
                       PREDICTION_TYPE.CLASSIFICATION,
                       'src.app.ml.iris.iris_predictor_sklearn',
                       label_filepath=LABEL_FILEPATH)
    elif args.save_format == 'onnx':
        onnx_filename = f'{modelname}.onnx'
        onnx_interface_filename = f'{modelname}_onnx_runtime.yaml'
        save_onnx(pipeline, os.path.join(MODEL_DIR, onnx_filename))
        save_interface(modelname,
                       os.path.join(MODEL_DIR, onnx_interface_filename),
                       [1, 4],
                       str(x_train.dtype).split('.')[-1],
                       [1, 3],
                       'float32',
                       DATA_TYPE.ARRAY,
                       [{onnx_filename: MODEL_RUNTIME.ONNX_RUNTIME}],
                       PREDICTION_TYPE.CLASSIFICATION,
                       'src.app.ml.iris.iris_predictor_onnx',
                       label_filepath=LABEL_FILEPATH)
    else:
        pass


if __name__ == '__main__':
    main()
