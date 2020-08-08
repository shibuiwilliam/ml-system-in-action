from sklearn import metrics
import onnxruntime as rt
import os
import numpy as np
import argparse
import joblib


MODEL_DIR = './models/'
DATA_DIR = './src/app/ml/iris/data/'
X_TEST_NPY = os.path.join(DATA_DIR, 'x_test.npy')
Y_TEST_NPY = os.path.join(DATA_DIR, 'y_test.npy')


def evaluate_sklearn_model(filepath: str, x_test: np.ndarray, y_test: np.ndarray):
    model = joblib.load(filepath)
    p = model.predict(x_test)
    score = metrics.accuracy_score(y_test, p)
    print(f'accuracy: {score}')


def evaluate_onnx_model(
        filepath: str,
        x_test: np.ndarray,
        y_test: np.ndarray):
    sess = rt.InferenceSession(filepath)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    p = sess.run([output_name], {input_name: x_test.astype('float32')})
    score = metrics.accuracy_score(y_test, p[0])
    print(f'accuracy: {score}')


def main():
    parser = argparse.ArgumentParser(description='iris evaluation.')
    parser.add_argument(
        '--evaluation_model_filename',
        required=False,
        type=str,
        default=str(os.getenv('EVALUATION_MODEL_FILENAME', 'iris_svc.pkl')))
    args = parser.parse_args()

    x_test = np.load(X_TEST_NPY)
    y_test = np.load(Y_TEST_NPY)

    model_filename = args.evaluation_model_filename

    if model_filename.endswith('.pkl'):
        evaluate_sklearn_model(os.path.join(MODEL_DIR, model_filename), x_test, y_test)
    if model_filename.endswith('.onnx'):
        evaluate_onnx_model(os.path.join(MODEL_DIR, model_filename), x_test, y_test)


if __name__ == '__main__':
    main()
