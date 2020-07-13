import onnxruntime as rt
import os
import json
import joblib
from PIL import Image
import numpy as np
from typing import Tuple, List, Union
import yaml
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from app.constants import PREDICTION_TYPE, PREDICTION_RUNTIME, DATA_TYPE


MODEL_DIR = './app/ml/models/'
MODEL_FILENAME = 'resnet50v2.onnx'
RESNET50_MODEL = os.path.join(MODEL_DIR, MODEL_FILENAME)
SAMPLE_IMAGE = os.path.join('./app/ml/imagenet_resnet50', 'good_cat.jpg')
LABEL_FILE = os.path.join(MODEL_DIR, 'imagenet_labels.json')


class ImagePreprocessTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            image_size: Tuple[int] = (224, 224),
            prediction_shape: Tuple[int] = (1, 3, 224, 224),
            mean_vec: List[float] = [0.485, 0.456, 0.406],
            stddev_vec: List[float] = [0.229, 0.224, 0.225]):
        self.image_size = image_size
        self.prediction_shape = prediction_shape
        self.mean_vec = mean_vec
        self.stddev_vec = stddev_vec

    def fit(self, X, y=None):
        return self

    def transform(self, X: Image) -> np.ndarray:
        image_data = np.array(X.resize(self.image_size)).transpose(2, 0, 1).astype('float32')

        mean_vec = np.array(self.mean_vec)
        stddev_vec = np.array(self.stddev_vec)
        norm_image_data = np.zeros(image_data.shape).astype('float32')
        for i in range(image_data.shape[0]):
            norm_image_data[i, :, :] = (image_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
        norm_image_data = norm_image_data.reshape(
            self.prediction_shape).astype('float32')
        return norm_image_data


class SoftmaxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        x = X.reshape(-1)
        e_x = np.exp(x - np.max(x))
        result = np.array([e_x / e_x.sum(axis=0)])
        return result


def dump_model(model, name: str):
    joblib.dump(model, name)


def save_interface(modelname: str,
                   filename: str,
                   input_shape: List,
                   input_type: str,
                   output_shape: List,
                   output_type: str,
                   data_type: DATA_TYPE,
                   model_filename: Union[List[str], str],
                   prediction_type: PREDICTION_TYPE,
                   prediction_runtime: PREDICTION_RUNTIME):
    os.makedirs(MODEL_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_DIR, filename)
    model_filename_list = model_filename if isinstance(model_filename, List) else [model_filename]
    with open(filepath, 'w') as f:
        f.write(yaml.dump({
            modelname: {
                'data_interface': {
                    'input_shape': input_shape,
                    'input_type': input_type,
                    'output_shape': output_shape,
                    'output_type': output_type,
                    'data_type': data_type.value
                },
                'meta': {
                    'model_filename': model_filename_list,
                    'prediction_type': prediction_type.value,
                    'prediction_runtime': prediction_runtime.value
                }
            }
        }, default_flow_style=False))


def load_labels():
    with open(LABEL_FILE, 'r') as f:
        return json.load(f)


def main():
    modelname = 'imagenet_resnet50'
    interface_filename = f'{modelname}.yaml'

    labels = load_labels()

    preprocess = ImagePreprocessTransformer()

    image = Image.open(SAMPLE_IMAGE)
    np_image = preprocess.transform(image)
    print(np_image.shape)

    preprocess_name = 'image_preprocess_transformer'
    preprocess_filename = f'{preprocess_name}.pkl'
    dump_model(preprocess, preprocess_filename)

    sess = rt.InferenceSession(RESNET50_MODEL)
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"input name='{inp.name}' shape={inp.shape} type={inp.type}")
    print(f"output name='{out.name}' shape={out.shape} type={out.type}")
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: np_image})

    postprocess = SoftmaxTransformer()
    postprocess_name = 'softmax_transformer'
    postprocess_filename = f'{postprocess_name}.pkl'
    dump_model(postprocess, postprocess_filename)
    prediction = postprocess.transform(np.array(pred_onx))

    print(prediction.shape)
    print(labels[np.argmax(prediction[0])])

    save_interface(modelname,
                   interface_filename,
                   [1, 3, 224, 224],
                   'float32',
                   [1, 1000],
                   'float32',
                   DATA_TYPE.IMAGE,
                   [preprocess_filename, MODEL_FILENAME, postprocess_filename],
                   PREDICTION_TYPE.CLASSIFICATION,
                   PREDICTION_RUNTIME.ONNX_RUNTIME)


if __name__ == '__main__':
    main()
