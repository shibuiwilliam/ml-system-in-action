from sklearn import datasets, svm, tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import os
import joblib
import numpy as np
from typing import Dict, List, Union
import yaml

from app.constants import PREDICTION_TYPE, MODEL_RUNTIME, DATA_TYPE
from app.ml.save_helper import save_interface, dump_sklearn


MODEL_DIR = './models/'


def get_data() -> Dict[str, np.ndarray]:
    iris = datasets.load_iris()
    print(
        f'input datatype: {type(iris.data)}, {iris.data.dtype}, {iris.data.shape}'
    )
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        shuffle=True,
        test_size=0.3)
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    return {'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test}


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


def evaluate_model(model, x: np.ndarray, y: np.ndarray):
    p = model.predict(x)
    score = metrics.accuracy_score(y, p)
    print(score)


def train_and_save(model,
                   modelname: str,
                   filename: str,
                   x_train: np.ndarray,
                   y_train: np.ndarray,
                   x_test: np.ndarray,
                   y_test: np.ndarray):
    train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump_sklearn(model, os.path.join(MODEL_DIR, filename))


def save_onnx(model,
              modelname: str,
              filename: str,
              x_test: np.ndarray,
              y_test: np.ndarray):
    os.makedirs(MODEL_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_DIR, filename)

    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(filepath, 'wb') as f:
        f.write(onx.SerializeToString())

    def test_run():
        sess = rt.InferenceSession(filepath)
        inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
        # print("input name='{}' and shape={} and type={}".format(inp.name, inp.shape, inp.type))
        # print("output name='{}' and shape={} and type={}".format(out.name, out.shape, out.type))
        input_name = sess.get_inputs()[0].name
        pred_onx = sess.run(None, {input_name: x_test.astype('float32')})
        # print(pred_sonx)
        score = metrics.accuracy_score(y_test, pred_onx[0])
        print(score)

    test_run()


def main():
    data = get_data()

    svc_pipeline = define_svc_pipeline()
    modelname = 'iris_svc'
    model_filename = f'{modelname}.pkl'
    interface_filename = f'{modelname}_sklearn.yaml'
    train_and_save(svc_pipeline,
                   modelname,
                   model_filename,
                   data['x_train'],
                   data['y_train'],
                   data['x_test'],
                   data['y_test'])
    save_interface(modelname,
                   MODEL_DIR,
                   interface_filename,
                   [1, 4],
                   str(data['x_train'].dtype).split('.')[-1],
                   [1, 3],
                   'float32',
                   DATA_TYPE.ARRAY,
                   [{model_filename: MODEL_RUNTIME.SKLEARN}],
                   PREDICTION_TYPE.CLASSIFICATION,
                   'app.ml.iris.iris_predictor_sklearn')

    onnx_filename = f'{modelname}.onnx'
    interface_filename = f'{modelname}_onnx_runtime.yaml'
    save_onnx(svc_pipeline,
              modelname,
              onnx_filename,
              data['x_test'],
              data['y_test'])
    save_interface(modelname,
                   MODEL_DIR,
                   interface_filename,
                   [1, 4],
                   str(data['x_train'].dtype).split('.')[-1],
                   [1, 3],
                   'float32',
                   DATA_TYPE.ARRAY,
                   [{onnx_filename: MODEL_RUNTIME.ONNX_RUNTIME}],
                   PREDICTION_TYPE.CLASSIFICATION,
                   'app.ml.iris.iris_predictor_onnx')

    tree_pipeline = define_tree_pipeline()
    modelname = 'iris_tree'
    model_filename = f'{modelname}.pkl'
    interface_filename = f'{modelname}_sklearn.yaml'
    train_and_save(tree_pipeline,
                   modelname,
                   model_filename,
                   data['x_train'],
                   data['y_train'],
                   data['x_test'],
                   data['y_test'])
    save_interface(modelname,
                   MODEL_DIR,
                   interface_filename,
                   [1, 4],
                   str(data['x_train'].dtype).split('.')[-1],
                   [1, 3],
                   'float32',
                   DATA_TYPE.ARRAY,
                   [{model_filename: MODEL_RUNTIME.SKLEARN}],
                   PREDICTION_TYPE.CLASSIFICATION,
                   'app.ml.iris.iris_predictor_sklearn')

    onnx_filename = f'{modelname}.onnx'
    interface_filename = f'{modelname}_onnx_runtime.yaml'
    save_onnx(tree_pipeline,
              modelname,
              onnx_filename,
              data['x_test'],
              data['y_test'])
    save_interface(modelname,
                   MODEL_DIR,
                   interface_filename,
                   [1, 4],
                   str(data['x_train'].dtype).split('.')[-1],
                   [1, 3],
                   'float32',
                   DATA_TYPE.ARRAY,
                   [{onnx_filename: MODEL_RUNTIME.ONNX_RUNTIME}],
                   PREDICTION_TYPE.CLASSIFICATION,
                   'app.ml.iris.iris_predictor_onnx')


if __name__ == '__main__':
    main()
