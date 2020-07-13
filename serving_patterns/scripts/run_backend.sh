#!/bin/bash

set -eu

NUM_PROCS=${NUM_PROCS:-2}
BATCH_CODE=${APP_NAME:-"app.backend.prediction_batch"}

PYTHONPATH=./ python -m ${BATCH_CODE}
