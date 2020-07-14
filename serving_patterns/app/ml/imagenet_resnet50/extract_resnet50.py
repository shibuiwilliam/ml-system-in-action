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

from app.constants import PREDICTION_TYPE, MODEL_RUNTIME, DATA_TYPE
from app.ml.save_helper import save_interface, load_labels
from app.ml.transformers import ImagePreprocessTransformer, SoftmaxTransformer


MODEL_DIR = './app/ml/models/'
MODEL_FILENAME = 'imagenet_resnet50v2.onnx'
RESNET50_MODEL = os.path.join(MODEL_DIR, MODEL_FILENAME)
SAMPLE_IMAGE = os.path.join('./app/ml/imagenet_resnet50', 'good_cat.jpg')
LABEL_FILE = os.path.join(MODEL_DIR, 'imagenet_labels.json')


def dump_model(model, name: str):
    joblib.dump(model, name)


def main():
    modelname = 'imagenet_resnet50'
    interface_filename = f'{modelname}.yaml'

    labels = load_labels(LABEL_FILE)

    preprocess = ImagePreprocessTransformer()

    image = Image.open(SAMPLE_IMAGE)
    np_image = preprocess.transform(image)
    print(np_image.shape)

    preprocess_name = f'{modelname}_preprocess_transformer'
    preprocess_filename = f'{preprocess_name}.pkl'
    dump_model(preprocess, os.path.join(MODEL_DIR, preprocess_filename))

    sess = rt.InferenceSession(RESNET50_MODEL)
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"input name='{inp.name}' shape={inp.shape} type={inp.type}")
    print(f"output name='{out.name}' shape={out.shape} type={out.type}")
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: np_image})

    postprocess = SoftmaxTransformer()
    postprocess_name = f'{modelname}_softmax_transformer'
    postprocess_filename = f'{postprocess_name}.pkl'
    dump_model(postprocess, os.path.join(MODEL_DIR, postprocess_filename))
    prediction = postprocess.transform(np.array(pred_onx))

    print(prediction.shape)
    print(labels[np.argmax(prediction[0])])

    save_interface(MODEL_DIR,
                   modelname,
                   interface_filename,
                   [1, 3, 224, 224],
                   'float32',
                   [1, 1000],
                   'float32',
                   DATA_TYPE.IMAGE,
                   [{preprocess_filename: MODEL_RUNTIME.SKLEARN},
                    {MODEL_FILENAME: MODEL_RUNTIME.ONNX_RUNTIME},
                    {postprocess_filename: MODEL_RUNTIME.SKLEARN}],
                   PREDICTION_TYPE.CLASSIFICATION,
                   'app.ml.imagenet_resnet50.imagenet_resnet50_predictor',
                   label_filepath=LABEL_FILE)


if __name__ == '__main__':
    main()
