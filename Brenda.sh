#!/bin/sh
script_file=$(realpath $0)
script_dir=$(dirname $script_file)
cd $script_dir
source brenda/bin/activate

export SUNO_ENABLE_MPS=true
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME=$script_dir/Cache/
export TTS_HOME=$script_dir/Cache/

streamlit run Brenda.py
deactivate