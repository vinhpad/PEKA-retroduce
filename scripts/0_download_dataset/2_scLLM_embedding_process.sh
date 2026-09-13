#!/bin/bash

# dynamic get project root, because script location is fixed, but project root is not fixed
script_dir=$(dirname "$(readlink -f "$0")")
export PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$script_dir")")")

# set dataset name
export DATASET_NAME="breast_visium_26k"  # set dataset name
export TISSUE_TYPE="breast"
# Attention over ~10k expressed genes allocates in large uneven blocks; expandable
# segments let the allocator reuse them instead of fragmenting the 24GB card.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# set scLLM embedder name
export SCLLM_EMBEDDER_NAME="scFoundation"

# optional parameters - if not set, will use default values in config file
export CKPT_NAME=""
export MODEL_MODE=""
export TOP_K=""

# run python script
python ${PROJECT_ROOT}/PEKA/peka/Exp_helper/3_scLLM_embedding_process.py
