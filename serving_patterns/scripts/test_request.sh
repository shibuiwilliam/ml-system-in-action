#!/bin/bash

set -eu

TARGET_HOST=localhost
WEB_SINGLE_PORT=8888
SYNCHRONOUS_PORT=8889
ASYNCHRONOUS_PORT=8890
WEB_SINGLE_IMAGE_PORT=8891
HEALTH=health
PREDICT=predict

function health() {
    curl -X GET \
        ${TARGET_HOST}:$1/${HEALTH}
    echo ""
}

function test_predict() {
    curl -X GET \
        ${TARGET_HOST}:$1/${PREDICT}
    echo ""
}

function predict() {
    curl -X POST \
        -H "Content-Type: application/json" \
        -d '{"input_data": [5.2, 3.1, 0.1, 1.0]}' \
        ${TARGET_HOST}:$1/${PREDICT}
    echo ""
}

function test_predict_image() {
    curl -X GET \
        ${TARGET_HOST}:$1/${PREDICT}/label
    echo ""
}

function predict_image() {
    curl -X POST \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@./app/ml/data/good_cat.jpg;type=image/jpeg" \
        ${TARGET_HOST}:$1/${PREDICT}/label
    echo ""
}

health ${WEB_SINGLE_PORT}
health ${SYNCHRONOUS_PORT}
health ${ASYNCHRONOUS_PORT}
health ${WEB_SINGLE_IMAGE_PORT}

test_predict ${WEB_SINGLE_PORT}
test_predict ${SYNCHRONOUS_PORT}
test_predict ${ASYNCHRONOUS_PORT}
test_predict_image ${WEB_SINGLE_IMAGE_PORT}

predict ${WEB_SINGLE_PORT}
predict ${SYNCHRONOUS_PORT}
predict ${ASYNCHRONOUS_PORT}
predict_image ${WEB_SINGLE_IMAGE_PORT}
