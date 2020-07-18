#!/bin/bash

set -eu

BATCH_CODE=${APP_NAME:-"app.backend.prediction_batch"}

PYTHONPATH=./ python -m ${BATCH_CODE}
