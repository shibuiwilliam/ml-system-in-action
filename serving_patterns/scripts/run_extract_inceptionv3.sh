#!/bin/bash

set -eu

target_dir=./models
label_file=${target_dir}/imagenet_labels_1001.json

mkdir -p ${target_dir}

if [ ! -f "${label_file}" ]; then
    curl --output ${label_file} https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json
    sed -i 's/"tench"/"background","tench"/' "${label_file}"
fi

PYTHONPATH=./ python -m app.ml.inceptionv3.extract_inceptionv3
