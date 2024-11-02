#!/bin/bash
source venv/bin/activate
script_file=$(realpath $0)
script_dir=$(dirname $script_file)
cd $script_dir
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME=$script_dir/Cache/
export TTS_HOME=$script_dir/Cache/
export COQUI_TOS_AGREED=1
export SUNO_ENABLE_MPS=true
echo "Starting Brendapi"
python3 Brendapi.py
