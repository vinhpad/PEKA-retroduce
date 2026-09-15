#!/bin/bash
set -e

# kd_lora_train.py resolves the project layout from its working directory,
# so always run it from this folder.
script_dir=$(dirname "$(readlink -f "$0")")
cd "$script_dir"

# large uneven activation blocks fragment the allocator; expandable segments
# let it reuse them instead
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Which benchmark to distil. One run per dataset; the paper trains all four.
# Default is breast only, because each run is ~12 GPU hours (paper, one V100).
#   DATASET=kidney bash train_kd_lora.sh
DATASET="${DATASET:-breast}"
case "$DATASET" in
    breast) TISSUE_TYPE="breast";       DATASET_NAME="breast_visium_26k" ;;
    kidney) TISSUE_TYPE="other_cancer"; DATASET_NAME="kidney_visium_74k" ;;
    liver)  TISSUE_TYPE="other_cancer"; DATASET_NAME="liver_visium_37k" ;;
    lung)   TISSUE_TYPE="other_cancer"; DATASET_NAME="lung_visium_65k" ;;
    *) echo "unknown DATASET '$DATASET' (breast|kidney|liver|lung)" >&2; exit 1 ;;
esac
SCLLM="scFoundation"
GEN_LABEL="clustered100"

# Config paths, relative to hydra_zen/Configs/
DATASET_CONFIG="Datasets/${DATASET_NAME}_${SCLLM}_with_${GEN_LABEL}_label.yaml"
# PEKA itself is the Block-Affine (Bone) adapter; the LoRA / AdaLoRA / HRA configs
# are the baselines the paper compares against. Bone needs peft >= 0.14.0.
MODEL_CONFIG="Models/H-optimus-0_Bone_MLP.yaml"
# knowledge distillation needs the CrossEntropyLoss variants, not the regression defaults
OPTIMIZER_CONFIG="Optimizers/kd_lora.yaml"
TRAINER_CONFIG="Trainers/kd_lora.yaml"

# Leave empty to train phase 1 from scratch, or point at a phase 1 classifier.pt to skip it
PHASE1_CKPT=""
PHASE1_EPOCHS=20
PHASE1_LR=1e-4
PHASE1_HIDDEN_DIM=512

# Experiment name
EXP_NAME="KD_LoRA_${TISSUE_TYPE}_${DATASET_NAME}_${SCLLM}_${GEN_LABEL}"

# Run the training script with the provided arguments
cmd="python kd_lora_train.py \
  --dataset_config \"$DATASET_CONFIG\" \
  --model_config \"$MODEL_CONFIG\" \
  --optimizer_config \"$OPTIMIZER_CONFIG\" \
  --trainer_config \"$TRAINER_CONFIG\" \
  --phase1_epochs $PHASE1_EPOCHS \
  --phase1_lr $PHASE1_LR \
  --phase1_hidden_dim $PHASE1_HIDDEN_DIM \
  --exp_name \"$EXP_NAME\""

if [ -n "$PHASE1_CKPT" ]; then
    cmd="$cmd --phase1_ckpt \"$PHASE1_CKPT\""
fi

echo "Executing: $cmd"
eval $cmd
