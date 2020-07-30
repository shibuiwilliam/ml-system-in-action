#!/bin/bash

set -eu

PYTHONPATH=./ python -m src.app.ml.iris.iris_trainer
