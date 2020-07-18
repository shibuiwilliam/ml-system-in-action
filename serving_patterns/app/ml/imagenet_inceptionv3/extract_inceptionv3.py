import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

from app.ml.transformers import TFImagePreprocessTransformer, SoftmaxTransformer
from app.ml.save_helper import load_labels

MODEL_DIR = './models/'
SAVEDMODEL_DIR = os.path.join(MODEL_DIR, "savedmodel/inceptionv3")
SAMPLE_IMAGE = os.path.join('./app/ml/data', 'good_cat.jpg')
LABEL_FILE = os.path.join(MODEL_DIR, 'imagenet_labels_1000.json')
LABELS = load_labels(LABEL_FILE)

def main():
    os.makedirs(SAVEDMODEL_DIR)

    hub_url = "https://tfhub.dev/google/imagenet/inception_v3/classification/4"
    model = tf.keras.Sequential([hub.KerasLayer(hub_url, input_shape=(299, 299, 3))])

    preprocess = TFImagePreprocessTransformer()

    image = Image.open(SAMPLE_IMAGE)
    np_image = preprocess.transform(image)

    result = model.predict(np_image)

    postprocess = SoftmaxTransformer()
    result_proba = postprocess.transform(result)
    print(result_proba)

    top1_index = np.argmax(result_proba[0], axis=-1)
    print(top1_index)
    print(LABELS[top1_index])

    tf.saved_model.save(model, SAVEDMODEL_DIR)


if __name__ == '__main__':
    main()
