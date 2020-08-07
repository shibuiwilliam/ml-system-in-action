#!/bin/bash

set -eu

target_dir=./models
label_file=${target_dir}/aiy_plants_V1_labelmap.csv

mkdir -p ${target_dir}

if [ ! -f "${label_file}" ]; then
    curl --output ${label_file} https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_plants_V1_labelmap.csv
fi

PYTHONPATH=./ python -m src.app.ml.mobilenetv2_plant.extract_mobilenetv2
