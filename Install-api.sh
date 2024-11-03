#!/bin/sh
python3.12 -m venv ./venv
source ./venv/bin/activate
pip install --no-cache-dir llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
pip install --no-cache-dir  -r requirements.txt