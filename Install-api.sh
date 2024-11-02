#!/bin/sh
python3.12 -m venv ./venv
source ./venv/bin/activate
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
pip install -r requirements.txt