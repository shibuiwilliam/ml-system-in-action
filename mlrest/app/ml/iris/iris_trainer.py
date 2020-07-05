from sklearn import datasets, svm, tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import joblib
import numpy as np
from typing import Dict, List
import yaml


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


def dump_model(model, name: str):
    joblib.dump(model, name)


def train_and_save(model,
                   modelname: str,
                   filename: str,
                   x_train: np.ndarray,
                   y_train: np.ndarray,
                   x_test: np.ndarray,
                   y_test: np.ndarray):
    train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)
    directory = f'./app/ml/models/{modelname}'
    os.makedirs(directory, exist_ok=True)
    dump_model(model, os.path.join(directory, filename))


def save_interface(modelname: str,
                   filename: str,
                   input_shape: List,
                   input_type: str,
                   output_shape: List,
                   output_type: str):
    directory = f'./app/ml/models/{modelname}'
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as f:
        f.write(yaml.dump({
            modelname: {
                'input_shape': input_shape,
                'input_type': input_type,
                'output_shape': output_shape,
                'output_type': output_type
            }
        }, default_flow_style=False))


def main():
    data = get_data()
    svc_pipeline = define_svc_pipeline()
    modelname = 'iris_svc'
    model_filename = f'{modelname}.pkl'
    interface_filename = f'{modelname}.yaml'
    train_and_save(svc_pipeline,
                   modelname,
                   model_filename,
                   data['x_train'],
                   data['y_train'],
                   data['x_test'],
                   data['y_test'])
    y_proba = svc_pipeline.predict_proba(data['x_test'])
    save_interface(modelname,
                   interface_filename,
                   [1, 4],
                   str(data['x_train'].dtype).split('.')[-1],
                   [1, 3],
                   str(y_proba.dtype).split('.')[-1])

    tree_pipeline = define_tree_pipeline()
    modelname = 'iris_tree'
    model_filename = f'{modelname}.pkl'
    interface_filename = f'{modelname}.yaml'
    train_and_save(tree_pipeline,
                   modelname,
                   model_filename,
                   data['x_train'],
                   data['y_train'],
                   data['x_test'],
                   data['y_test'])
    y_proba = tree_pipeline.predict_proba(data['x_test'])
    save_interface(modelname,
                   interface_filename,
                   [1, 4],
                   str(data['x_train'].dtype).split('.')[-1],
                   [1, 3],
                   str(y_proba.dtype).split('.')[-1])


if __name__ == '__main__':
    main()
