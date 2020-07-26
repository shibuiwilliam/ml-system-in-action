#!/bin/bash

set -eu

TARGET_HOST=localhost
PORT=8893
HEALTH=health
LABELS=labels
PREDICT=predict
LABEL=label
ASYNC=async
JSON=json
IMAGE_PATH=./app/ml/data/good_cat.jpg
JSON_PATH=./app/ml/data/good_cat_base64.json

function which_is_it() {
    echo "******* ${1} *******"
}

function finish() {
    echo -e "\n"
}

function health() {
    endpoint=${TARGET_HOST}:$1/${HEALTH}
    which_is_it "${endpoint}"
    curl -X GET \
        ${endpoint}
    finish
}

function labels(){
    endpoint=${TARGET_HOST}:$1/${PREDICT}/${LABELS}
    which_is_it "${endpoint}"
    curl -X GET \
        ${endpoint}
    finish
}

function test_predict() {
    endpoint=${TARGET_HOST}:$1/${PREDICT}
    which_is_it "${endpoint}"
    curl -X GET \
        ${endpoint}
    finish
}

function test_predict_label() {
    endpoint=${TARGET_HOST}:$1/${PREDICT}/${LABEL}
    which_is_it "${endpoint}"
    curl -X GET \
        ${endpoint}
    finish
}

function predict() {
    endpoint=${TARGET_HOST}:$1/${PREDICT}
    which_is_it "${endpoint}"
    curl -X POST \
        -H "Content-Type: multipart/form-data" \
        -F "file=@${IMAGE_PATH};type=image/jpeg" \
        ${endpoint}
    finish
}

function predict_label() {
    endpoint=${TARGET_HOST}:$1/${PREDICT}/${LABEL}
    which_is_it "${endpoint}"
    curl -X POST \
        -H "Content-Type: multipart/form-data" \
        -F "file=@${IMAGE_PATH};type=image/jpeg" \
        ${endpoint}
    finish
}

function predict_async() {
    endpoint=${TARGET_HOST}:$1/${PREDICT}/${ASYNC}
    which_is_it "${endpoint}"
    curl -X POST \
        -H "Content-Type: multipart/form-data" \
        -F "file=@${IMAGE_PATH};type=image/jpeg" \
        ${endpoint}
    finish
}

function predict_json() {
    endpoint=${TARGET_HOST}:$1/${PREDICT}/${JSON}
    which_is_it "${endpoint}"
    curl -X POST \
        -H "Content-Type: application/json" \
        -d @${JSON_PATH} \
        ${endpoint}
    finish
}

function predict_label_json() {
    endpoint=${TARGET_HOST}:$1/${PREDICT}/${LABEL}/${JSON}
    which_is_it "${endpoint}"
    curl -X POST \
        -H "Content-Type: application/json" \
        -d @${JSON_PATH} \
        ${endpoint}
    finish
}

function predict_async_json() {
    endpoint=${TARGET_HOST}:$1/${PREDICT}/${ASYNC}/${JSON}
    which_is_it "${endpoint}"
    curl -X POST \
        -H "Content-Type: application/json" \
        -d @${JSON_PATH} \
        ${endpoint}
    finish
}

function all() {
    health $1
    labels $1
    test_predict $1
    test_predict_label $1
    predict $1
    predict_label $1
    predict_async $1
    predict_json $1
    predict_label_json $1
    predict_async_json $1
}

all ${PORT}