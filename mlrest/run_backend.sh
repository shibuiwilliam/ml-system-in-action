#!/bin/bash

set -eu

PYTHONPATH=./ python -m app.backend.prediction_batch
