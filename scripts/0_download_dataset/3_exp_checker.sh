#!/bin/bash

script_dir=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$script_dir")")")
echo "PROJECT_ROOT: $PROJECT_ROOT"

# `conda activate` needs conda's shell functions, which are not loaded in a
# non-interactive shell; `conda init bash` alone does not make them available here.
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "hest" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate hest
fi

python ${PROJECT_ROOT}/PEKA/peka/Exp_helper/4_model_required_rec_check.py \
    --model_name "hf-hub:bioptimus/H-optimus-0" \
    --input_size 3 224 224 \
    --lora_r 32 \
    --lora_alpha 16 \
    --target_modules qkv proj \
    --cuda_ram_test \
    --run_lora_test
