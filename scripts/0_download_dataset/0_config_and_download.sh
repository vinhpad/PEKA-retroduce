#!/bin/bash
# dynamic get project root, because script location is fixed, but project root is not fixed
script_dir=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$script_dir")")")
# the python entry points resolve the project layout from their working directory
cd "$script_dir"
echo "PROJECT_ROOT: $PROJECT_ROOT"

CODE_ROOT="${PROJECT_ROOT}/PEKA/"
SRC_ROOT="${PROJECT_ROOT}/PEKA/peka/"
EXTERNAL_MODELS_ROOT="${PROJECT_ROOT}/PEKA/peka/External_models/"

# Which slice of HEST1k to fetch:
#   subset  -> only the WSIs the predefined sub-datasets need (default, ~12% of HEST1k)
#   full    -> mirror the entire HEST1k release (~1TB)
DOWNLOAD_MODE="subset"
# Restrict further to the four benchmarks reported in the paper
# (breast_visium_26k, kidney_in_hest, liver_in_hest, lung_in_hest)
PAPER_ONLY=true
# Set to true to inspect what would be downloaded without fetching anything
DRY_RUN=false

# config project paths
python ${SRC_ROOT}/Exp_helper/0_config_runable.py

# download HEST1k database; already present samples are skipped
cmd="python ${SRC_ROOT}/Exp_helper/1_dataset_downloader.py --mode ${DOWNLOAD_MODE}"
if [ "$PAPER_ONLY" = true ]; then
    cmd="$cmd --paper_only"
fi
if [ "$DRY_RUN" = true ]; then
    cmd="$cmd --dry_run"
fi

echo "Executing: $cmd"
eval $cmd
