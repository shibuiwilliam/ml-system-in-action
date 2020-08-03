#!/bin/bash

set -eu

target_dir=./models
target_tar=${target_dir}/resnet50v2.tar.gz
target_extracted=${target_dir}/resnet50v2
original_filename=${target_dir}/resnet50v2/resnet50v2.onnx
model_filename=${target_dir}/resnet50v2.onnx
label_file=${target_dir}/imagenet_labels_1000.json

mkdir -p ${target_dir}

if [ ! -f "${model_filename}" ]; then
    echo "pass"
    # [ ! -f "${target_tar}" ] && curl --output ${target_tar} https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz

    # [ ! -d "${target_extracted}" ] && tar zxvf ${target_tar} -C ${target_dir}

    # mv ${original_filename} ${model_filename}
    # rm -f ${target_tar}
    # rm -rf ${target_extracted}
fi

[ ! -f "${label_file}" ] && curl --output ${label_file} https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json


PYTHONPATH=./ python -m src.app.ml.resnet50_onnx.extract_resnet50
