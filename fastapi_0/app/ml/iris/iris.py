from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from typing import Dict


def get_data() -> Dict[str, np.ndarray]:
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        shuffle=True,
        test_size=0.3)
    return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test
            }


def define_pipeline() -> Pipeline:
    steps = [
        ('normalize', StandardScaler()),
        ('svc', svm.SVC())
    ]
    pipeline = Pipeline(steps=steps)
    return pipeline


def train_model(model, x: np.ndarray, y: np.ndarray):
    model.fit(x, y)


def evaluate_model(model, x: np.ndarray, y: np.ndarray):
    p = model.predict(x)
    score = metrics.accuracy_score(y, p)
    print(score)


def dump_model(model, name):
    joblib.dump(model, name)


def main():
    data = get_data()
    pipeline = define_pipeline()
    train_model(pipeline, data["x_train"], data["y_train"])
    evaluate_model(pipeline, data["x_test"], data["y_test"])
    dump_model(pipeline, "./iris_svc.pkl")


if __name__ == '__main__':
    main()