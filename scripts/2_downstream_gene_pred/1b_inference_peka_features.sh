#!/bin/bash
# Step 2.3 — extract PEKA (distilled student) feature vectors for the paper's benchmarks.
#
# Runs after Phase 1 training. Produces the .npy files that
# step3_task_gene_expr_reg_KFold.py reads for --feature_type peka and image_encoder+peka.
# Datasets without data or without a Phase 1 checkpoint are skipped, so this is safe to
# run after training breast only.
#
#   DATASETS="breast" bash 1b_inference_peka_features.sh
#   IMAGE_ENCODER_NAME=UNI MODEL_CONFIG=Models/UNI_Bone_MLP.yaml bash 1b_inference_peka_features.sh
set -u

script_dir=$(dirname "$(readlink -f "$0")")
cd "$script_dir"
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$script_dir")")")

EMBEDDER_NAME="${EMBEDDER_NAME:-scFoundation}"
# get_dataset_paths() hard-codes checkpoint_name="default_model" and step3 never overrides
# it, so the output folder has to be spelled exactly this way.
SCLLM_CKPT="${SCLLM_CKPT:-default_model}"
GEN_LABEL="${GEN_LABEL:-clustered100}"

# PEKA itself is the Block-Affine (Bone) adapter; LoRA / AdaLoRA / HRA are the baselines.
# Must match the config the checkpoint was trained with.
IMAGE_ENCODER_NAME="${IMAGE_ENCODER_NAME:-H0}"
MODEL_CONFIG="${MODEL_CONFIG:-Models/H-optimus-0_Bone_MLP.yaml}"

# <key>:<tissue_type>:<dataset_name>
BENCHMARKS=(
  "breast:breast:breast_visium_26k"
  "kidney:other_cancer:kidney_visium_74k"
  "liver:other_cancer:liver_visium_37k"
  "lung:other_cancer:lung_visium_65k"
)
DATASETS="${DATASETS:-breast kidney liver lung}"

# lora_r must match the config the checkpoint was trained with — read it from the YAML
LORA_R=$(sed -n 's/^[[:space:]]*lora_r:[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
         "${PROJECT_ROOT}/PEKA/hydra_zen/Configs/${MODEL_CONFIG}" | head -1)
[ -n "$LORA_R" ] || { echo "ERROR: could not read lora_r from ${MODEL_CONFIG}" >&2; exit 1; }

ran=0
skipped=""
for key in $DATASETS; do
    entry=""
    for b in "${BENCHMARKS[@]}"; do
        [ "${b%%:*}" = "$key" ] && entry="$b"
    done
    [ -n "$entry" ] || { echo "!! unknown dataset key '$key'" >&2; exit 1; }
    IFS=':' read -r _ TISSUE_TYPE DATASET_NAME <<< "$entry"

    DATASET_DIR="${PROJECT_ROOT}/PEKA/DATA/${TISSUE_TYPE}/${DATASET_NAME}"
    PAIRED_SEQ_DIR="${DATASET_DIR}/scLLM_embed/${EMBEDDER_NAME}/${SCLLM_CKPT}/paired_seq"
    SCLLM_EMBED_DIR="${DATASET_DIR}/scLLM_embed/${EMBEDDER_NAME}/${SCLLM_CKPT}/embeddings"
    OUTPUT_DIR="${DATASET_DIR}/peka_embed/${IMAGE_ENCODER_NAME}/${EMBEDDER_NAME}/${SCLLM_CKPT}"
    EXP_NAME="KD_LoRA_${TISSUE_TYPE}_${DATASET_NAME}_${EMBEDDER_NAME}_${GEN_LABEL}"

    [ -d "$PAIRED_SEQ_DIR" ] || { skipped="${skipped} ${key}(no paired_seq, run step 0.3)"; continue; }
    [ -d "${DATASET_DIR}/patches" ] || { skipped="${skipped} ${key}(no patches)"; continue; }

    # This is the ModelCheckpoint file, NOT OUTPUT/<exp>_<ts>/phase2/lora/ — step2 reads
    # checkpoint['state_dict'], which the lora/ folder does not have.
    CKPT="${MODEL_CHECKPOINT:-$(ls -t "${PROJECT_ROOT}/Pretrained/${EXP_NAME}"/*.ckpt 2>/dev/null | head -1)}"
    [ -n "$CKPT" ] && [ -f "$CKPT" ] || { skipped="${skipped} ${key}(no ckpt in Pretrained/${EXP_NAME})"; continue; }

    N_ADATA=$(ls "${PAIRED_SEQ_DIR}"/HEST_breast_adata_*.h5ad 2>/dev/null | wc -l | tr -d ' ')
    [ "$N_ADATA" -gt 0 ] || { skipped="${skipped} ${key}(paired_seq empty)"; continue; }

    # target_scllm_dim must equal the scLLM embedding width the student was trained against
    TARGET_SCLLM_DIM=$(python -c "
import glob, numpy as np, sys
f = sorted(glob.glob('${SCLLM_EMBED_DIR}/*.npy'))
if not f: sys.exit('no scLLM embeddings in ${SCLLM_EMBED_DIR}')
print(np.load(f[0], mmap_mode='r').shape[1])
") || { skipped="${skipped} ${key}(no scLLM embeddings)"; continue; }

    echo "=== ${DATASET_NAME} | ${IMAGE_ENCODER_NAME} | lora_r=${LORA_R} | dim=${TARGET_SCLLM_DIM} ==="
    echo "    ckpt:   $CKPT"
    echo "    output: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # --batch_size is a no-op: inference_from_folder hard-codes batch_size=32.
    python step2_inference_feature_vectors.py \
        --project_root "$PROJECT_ROOT" \
        --tissue_type "$TISSUE_TYPE" \
        --dataset_name "$DATASET_NAME" \
        --scllm "$EMBEDDER_NAME" \
        --scllm_ckpt "$SCLLM_CKPT" \
        --target_scllm_dim "$TARGET_SCLLM_DIM" \
        --image_encoder_name "$IMAGE_ENCODER_NAME" \
        --model_config "$MODEL_CONFIG" \
        --model_checkpoint "$CKPT" \
        --lora_r "$LORA_R" \
        --output_dir "$OUTPUT_DIR"

    # step2 swallows save errors and skips indices whose patch .h5 is missing, then still
    # prints "Inference completed successfully" — so count the files instead.
    N_NPY=$(ls "${OUTPUT_DIR}"/HEST_breast_adata_*.npy 2>/dev/null | wc -l | tr -d ' ')
    echo "    wrote ${N_NPY}/${N_ADATA} .npy files"
    if [ "$N_NPY" -ne "$N_ADATA" ]; then
        echo "    WARNING: count mismatch — check the log for 'Image file ... not found, skipping'" >&2
        echo "             or 'Error while saving features'." >&2
    fi
    ran=$((ran + 1))
done

echo
echo "finished ${ran} dataset(s)"
[ -n "$skipped" ] && echo "skipped:${skipped}"
exit 0
