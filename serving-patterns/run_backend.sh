#!/bin/bash

set -eu

NUM_PROCS=${NUM_PROCS:-4}

PYTHONPATH=./ python -m app.backend.prediction_batch
