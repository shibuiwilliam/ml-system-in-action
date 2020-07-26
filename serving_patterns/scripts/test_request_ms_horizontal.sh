#!/bin/bash

set -eu

TARGET_HOST=localhost
PORT=8000
HEALTH=health
HEALTH_ALL=health_all
REDIRECT=redirect
REDIRECT_JSON=redirect_json
LABELS=labels
PREDICT=predict
LABEL=label
ASYNC=async
JSON=json
IMAGE_PATH=./app/ml/data/good_cat.jpg
JSON_PATH=./app/ml/data/good_cat_base64_data.json

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

function health_all() {
    endpoint=${TARGET_HOST}:$1/${HEALTH_ALL}
    which_is_it "${endpoint}"
    curl -X GET \
        ${endpoint}
    finish
}

function labels(){
    endpoint=${TARGET_HOST}:$1/${REDIRECT}/${PREDICT}/${LABELS}
    which_is_it "${endpoint}"
    curl -X GET \
        ${endpoint}
    finish
}

function test_predict() {
    endpoint=${TARGET_HOST}:$1/${REDIRECT}/${PREDICT}
    which_is_it "${endpoint}"
    curl -X GET \
        ${endpoint}
    finish
}

function test_predict_label() {
    endpoint=${TARGET_HOST}:$1/${REDIRECT}/${PREDICT}/${LABEL}
    which_is_it "${endpoint}"
    curl -X GET \
        ${endpoint}
    finish
}

function predict_json() {
    endpoint=${TARGET_HOST}:$1/${REDIRECT_JSON}/${PREDICT}/${JSON}
    which_is_it "${endpoint}"
    curl -X POST \
        -H "Content-Type: application/json" \
        -d @${JSON_PATH} \
        ${endpoint}
    finish
}

function predict_label_json() {
    endpoint=${TARGET_HOST}:$1/${REDIRECT_JSON}/${PREDICT}/${LABEL}/${JSON}
    which_is_it "${endpoint}"
    curl -X POST \
        -H "Content-Type: application/json" \
        -d @${JSON_PATH} \
        ${endpoint}
    finish
}

function predict_async_json() {
    endpoint=${TARGET_HOST}:$1/${REDIRECT_JSON}/${PREDICT}/${ASYNC}/${JSON}
    which_is_it "${endpoint}"
    curl -X POST \
        -H "Content-Type: application/json" \
        -d @${JSON_PATH} \
        ${endpoint}
    finish
}

function all() {
    health $1
    health_all $1
    labels $1
    test_predict $1
    test_predict_label $1
    predict_json $1
    predict_label_json $1
    predict_async_json $1
}

all ${PORT}