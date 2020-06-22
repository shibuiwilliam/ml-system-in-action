#!/bin/bash

set -eu

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8888}
WORKERS=${WORKER:-4}
UVICORN_WORKER=${UVICORN_WORKER:-"uvicorn.workers.UvicornWorker"}
LOGLEVEL=${LOGLEVEL:-debug}
LOGCONFIG=${LOGCONFIG:-"./logging.conf"}

gunicorn app:app \
    -b ${HOST}:${PORT} \
    -w ${WORKERS} \
    -k ${UVICORN_WORKER}  \
    --log-level ${LOGLEVEL} \
    --log-config ${LOGCONFIG}