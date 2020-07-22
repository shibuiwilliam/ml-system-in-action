import onnxruntime as rt
import os
import joblib
from PIL import Image
import numpy as np

from app.constants import PREDICTION_TYPE, MODEL_RUNTIME, DATA_TYPE
from app.ml.save_helper import save_interface, load_labels, dump_sklearn
from app.ml.transformers import ONNXImagePreprocessTransformer, SoftmaxTransformer


MODEL_DIR = './models/'
MODEL_FILENAME = 'resnet50v2.onnx'
RESNET50_MODEL = os.path.join(MODEL_DIR, MODEL_FILENAME)
SAMPLE_IMAGE = os.path.join('./app/ml/data', 'good_cat.jpg')
LABEL_FILEPATH = os.path.join(MODEL_DIR, 'imagenet_labels_1000.json')


def main():
    modelname = 'resnet50'
    interface_filename = f'{modelname}.yaml'

    labels = load_labels(LABEL_FILEPATH)

    preprocess = ONNXImagePreprocessTransformer()

    image = Image.open(SAMPLE_IMAGE)
    np_image = preprocess.transform(image)
    print(np_image.shape)

    preprocess_name = f'{modelname}_preprocess_transformer'
    preprocess_filename = f'{preprocess_name}.pkl'
    dump_sklearn(preprocess, os.path.join(MODEL_DIR, preprocess_filename))

    sess = rt.InferenceSession(RESNET50_MODEL)
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"input name='{inp.name}' shape={inp.shape} type={inp.type}")
    print(f"output name='{out.name}' shape={out.shape} type={out.type}")
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: np_image})

    postprocess = SoftmaxTransformer()
    postprocess_name = f'{modelname}_softmax_transformer'
    postprocess_filename = f'{postprocess_name}.pkl'
    dump_sklearn(postprocess, os.path.join(MODEL_DIR, postprocess_filename))
    prediction = postprocess.transform(np.array(pred_onx))

    print(prediction.shape)
    print(labels[np.argmax(prediction[0])])

    save_interface(modelname,
                   os.path.join(MODEL_DIR, interface_filename),
                   [1, 3, 224, 224],
                   'float32',
                   [1, 1000],
                   'float32',
                   DATA_TYPE.IMAGE,
                   [{preprocess_filename: MODEL_RUNTIME.SKLEARN},
                    {MODEL_FILENAME: MODEL_RUNTIME.ONNX_RUNTIME},
                    {postprocess_filename: MODEL_RUNTIME.SKLEARN}],
                   PREDICTION_TYPE.CLASSIFICATION,
                   'app.ml.resnet50.resnet50_predictor',
                   label_filepath=LABEL_FILEPATH)


if __name__ == '__main__':
    main()
