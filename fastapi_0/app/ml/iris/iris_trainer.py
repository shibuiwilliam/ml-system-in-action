from sklearn import datasets, svm, tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import joblib
import numpy as np
from typing import Dict


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
                   name: str,
                   x_train: np.ndarray,
                   y_train: np.ndarray,
                   x_test: np.ndarray,
                   y_test: np.ndarray):
    train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)
    dump_model(model, os.path.join('./app/ml/models/', name))


def main():
    data = get_data()
    svc_pipeline = define_svc_pipeline()
    train_and_save(svc_pipeline,
                   'iris_svc.pkl',
                   data['x_train'],
                   data['y_train'],
                   data['x_test'],
                   data['y_test'])
    tree_pipeline = define_tree_pipeline()
    train_and_save(tree_pipeline,
                   'iris_tree.pkl',
                   data['x_train'],
                   data['y_train'],
                   data['x_test'],
                   data['y_test'])


if __name__ == '__main__':
    main()
