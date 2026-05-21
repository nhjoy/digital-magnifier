#!/bin/bash

cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONPATH=src

python -m digital_magnifier.main