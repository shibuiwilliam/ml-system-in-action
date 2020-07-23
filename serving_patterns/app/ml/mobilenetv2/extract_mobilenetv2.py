import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

from app.constants import PREDICTION_TYPE, MODEL_RUNTIME, DATA_TYPE
from app.ml.save_helper import save_interface, load_labels, dump_sklearn
from app.ml.transformers import TFImagePreprocessTransformer, SoftmaxTransformer
from app.ml.extract_from_tfhub import get_model

MODEL_DIR = './models/'
MODEL_FILE_DIR = 'savedmodel/mobilenetv2/4'
SAVEDMODEL_DIR = os.path.join(MODEL_DIR, MODEL_FILE_DIR)
PB_FILE = os.path.join(SAVEDMODEL_DIR, 'saved_model.pb')
HUB_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4'
SAMPLE_IMAGE = os.path.join('./app/ml/data', 'good_cat.jpg')
LABEL_FILEPATH = os.path.join(MODEL_DIR, 'imagenet_labels_1001.json')
LABELS = load_labels(LABEL_FILEPATH)


def validate(image, preprocess, predictor, postprocess):
    np_image = preprocess.transform(image)
    result = predictor.predict(np_image)
    result_proba = postprocess.transform(result)
    print(result_proba)
    top1_index = np.argmax(result_proba[0], axis=-1)
    print(top1_index)
    print(LABELS[top1_index])


def main():
    os.makedirs(SAVEDMODEL_DIR, exist_ok=True)

    if os.path.exists(PB_FILE):
        print(f'saved model {SAVEDMODEL_DIR} found')
        model = tf.keras.models.load_model(SAVEDMODEL_DIR)
    else:
        print(f'saved model {SAVEDMODEL_DIR} not found')
        model = get_model(HUB_URL, (224, 224, 3))

    preprocess = TFImagePreprocessTransformer(
        image_size=(
            224, 224), prediction_shape=(
            1, 224, 224, 3))
    postprocess = SoftmaxTransformer()

    image = Image.open(SAMPLE_IMAGE)

    validate(image, preprocess, model, postprocess)

    tf.saved_model.save(model, SAVEDMODEL_DIR)

    modelname = 'mobilenetv2'
    interface_filename = f'{modelname}.yaml'
    preprocess_filename = f'{modelname}_preprocess_transformer.pkl'
    postprocess_filename = f'{modelname}_softmax_transformer.pkl'
    dump_sklearn(preprocess, os.path.join(MODEL_DIR, preprocess_filename))
    dump_sklearn(postprocess, os.path.join(MODEL_DIR, postprocess_filename))

    save_interface(modelname,
                   os.path.join(MODEL_DIR, interface_filename),
                   [1, 224, 224, 3],
                   'float32',
                   [1, 1001],
                   'float32',
                   DATA_TYPE.IMAGE,
                   [{preprocess_filename: MODEL_RUNTIME.SKLEARN},
                    {MODEL_FILE_DIR: MODEL_RUNTIME.TF_SERVING},
                    {postprocess_filename: MODEL_RUNTIME.SKLEARN}],
                   PREDICTION_TYPE.CLASSIFICATION,
                   'app.ml.mobilenetv2.mobilenetv2_predictor',
                   label_filepath=LABEL_FILEPATH,
                   model_spec_name='mobilenetv2',
                   model_spec_signature_name='serving_default',
                   input_name='keras_layer_input',
                   output_name='keras_layer')


if __name__ == '__main__':
    main()
