#!/bin/bash

set -eu

PYTHONPATH=./ python -m app.ml.imagenet_inceptionv3.extract_inceptionv3
