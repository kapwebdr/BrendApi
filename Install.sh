#!/bin/sh
script_file=$(realpath $0)
script_dir=$(dirname $script_file)
cd $script_dir
#brew install python@3.9
# rm -Rf ./brenda
python3.9 -m venv brenda
source brenda/bin/activate

CMAKE_ARGS="-DLLAMA_METAL=on" 
FORCE_CMAKE=1 
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

# pip install --upgrade pip setuptools wheel

export CT_METAL=1 
pip install ctransformers --no-binary ctransformers
pip install git+https://github.com/huggingface/transformers


pip install -r $script_dir/requirements.txt #--no-cache-dir  --force-reinstall --no-cache-dir 

deactivate