#!/bin/bash

target_dir=./models
target_tar=${target_dir}/resnet50v2.tar.gz
target_extracted=${target_dir}/resnet50v2
original_filename=${target_dir}/resnet50v2/resnet50v2.onnx
model_filename=${target_dir}/imagenet_resnet50v2.onnx
label_file=${target_dir}/imagenet_labels.json

if [ ! -f "${model_filename}" ]; then
    mkdir -p ${target_dir}
    [ ! -f "${target_tar}" ] && curl --output ${target_dir}/resnet50v2.tar.gz https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz

    [ ! -d "${target_extracted}" ] && tar zxvf ${target_tar} -C ${target_dir}

    mv ${original_filename} ${model_filename}
    rm -f ${target_tar}
    rm -rf ${target_extracted}
fi

[ ! -f "${label_file}" ] && curl --output ${target_dir}/imagenet_labels.json https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json
