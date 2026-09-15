#!/bin/bash
# Phase 2 sweep for the H0 student: 5-fold gene expression regression over the
# paper's four benchmarks x the feature types it compares.
#
# Defaults follow PEKA (arXiv 2504.07061) sec 3.1/4: Visium human benchmarks, raw
# (log1p-normalised) expression, top-50 HVG, 5 folds. Datasets whose features are not
# on disk are skipped, so this is safe to run after building only breast.
#
# Override from the environment, e.g.
#   DATASETS="breast" bash 2_auto_reg_KFold_auto_H0.sh
#   BINNED_OPTIONS="false true" FEATURE_TYPES="peka" bash 2_auto_reg_KFold_auto_H0.sh
script_dir=$(dirname "$(readlink -f "$0")")
cd "$script_dir"
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$script_dir")")")

IMAGE_ENCODER_NAME="H0"
IMAGE_BACKBONE="H-optimus-0"
EMBEDDER_NAME="scFoundation"
BASE_OUTPUT_ROOT="${PROJECT_ROOT}/OUTPUT"
EPOCHS=300                      # no-op: train_regressor fits PCA+Ridge in closed form
WITH_INDEPENDENT_TEST_SET=false # false = plain 5-fold, each fold trains on 4/5 (the paper's protocol)

# the four benchmarks: <key>:<tissue_type>:<dataset_name>:<gene list json>
BENCHMARKS=(
  "breast:breast:breast_visium_26k:top_50_genes_Visium_Homo_sapien_Breast_Cancer.json"
  "kidney:other_cancer:kidney_visium_74k:top_50_genes_visium_homo_sapien_Kidney_cancer.json"
  "liver:other_cancer:liver_visium_37k:top_50_genes_visium_homo_sapien_Liver_cancer.json"
  "lung:other_cancer:lung_visium_65k:top_50_genes_visium_homo_sapien_Lung_cancer.json"
)
DATASETS="${DATASETS:-breast kidney liver lung}"
# the paper reports raw expression only; set BINNED_OPTIONS="false true" to add the binned variant
BINNED_OPTIONS="${BINNED_OPTIONS:-false}"
FEATURE_TYPES="${FEATURE_TYPES:-image_encoder peka scLLM image_encoder+peka}"

ran=0
skipped=""
for key in $DATASETS; do
    entry=""
    for b in "${BENCHMARKS[@]}"; do
        [ "${b%%:*}" = "$key" ] && entry="$b"
    done
    if [ -z "$entry" ]; then
        echo "!! unknown dataset key '$key' (expected one of: breast kidney liver lung)" >&2
        exit 1
    fi
    IFS=':' read -r _ TISSUE_TYPE DATASET_NAME GENE_LIST <<< "$entry"
    DATASET_DIR="${PROJECT_ROOT}/PEKA/DATA/${TISSUE_TYPE}/${DATASET_NAME}"
    GENE_LIST_JSON="${script_dir}/${GENE_LIST}"

    if [ ! -d "$DATASET_DIR" ]; then
        skipped="${skipped} ${key}(no DATA dir)"
        continue
    fi
    if [ ! -f "$GENE_LIST_JSON" ]; then
        skipped="${skipped} ${key}(no gene list)"
        continue
    fi

    for USE_BINNED in $BINNED_OPTIONS; do
        for FEATURE_TYPE in $FEATURE_TYPES; do
            # every feature type but scLLM needs the encoder-specific embeddings on disk
            case "$FEATURE_TYPE" in
                peka|image_encoder+peka)
                    [ -d "${DATASET_DIR}/peka_embed/${IMAGE_ENCODER_NAME}" ] || {
                        skipped="${skipped} ${key}/${FEATURE_TYPE}(no peka_embed, run step 2.3)"; continue; } ;;
                image_encoder)
                    [ -d "${DATASET_DIR}/patches_embed/${IMAGE_BACKBONE}" ] || {
                        skipped="${skipped} ${key}/${FEATURE_TYPE}(no patches_embed)"; continue; } ;;
            esac

            binned_str=$([ "$USE_BINNED" = true ] && echo "binned" || echo "raw")
            OUTPUT_ROOT="${BASE_OUTPUT_ROOT}/${TISSUE_TYPE}/${DATASET_NAME}/${binned_str}/"
            mkdir -p "$OUTPUT_ROOT"

            echo "=== ${DATASET_NAME} | ${FEATURE_TYPE} | ${binned_str} | ${IMAGE_ENCODER_NAME} ==="

            cmd="python ${script_dir}/step3_task_gene_expr_reg_KFold.py \
            --project_root $PROJECT_ROOT \
            --tissue_type $TISSUE_TYPE \
            --dataset_name $DATASET_NAME \
            --embedder_name $EMBEDDER_NAME \
            --gene_list_json \"$GENE_LIST_JSON\" \
            --output_root $OUTPUT_ROOT \
            --feature_type $FEATURE_TYPE \
            --epochs $EPOCHS \
            --image_encoder_name $IMAGE_ENCODER_NAME \
            --image_backbone \"$IMAGE_BACKBONE\" \
            --mask_zero_values"

            [ "$USE_BINNED" = true ] && cmd="$cmd --use_binned"
            [ "$WITH_INDEPENDENT_TEST_SET" = true ] && cmd="$cmd --with_independent_test_set"

            echo "Executing: $cmd"
            eval $cmd
            ran=$((ran + 1))
        done
    done
done

echo
echo "finished ${ran} run(s)"
[ -n "$skipped" ] && echo "skipped:${skipped}"
exit 0
