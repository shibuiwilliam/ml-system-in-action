#!/bin/bash

set -eu

target_dir=./models
label_file=${target_dir}/imagenet_labels_1000.json

mkdir -p ${target_dir}

[ ! -f "${label_file}" ] && curl --output ${label_file} https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json

PYTHONPATH=./ python -m src.app.ml.resnet50_pytorch_caffe2.extract_resnet50
