#!/bin/bash
# dynamic get project root, because script location is fixed, but project root is not fixed
script_dir=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$script_dir")")")
# the python entry point resolves the project layout from its working directory
cd "$script_dir"
DATABASE_ROOT="${PROJECT_ROOT}/PEKA/DATA/other_cancer/"

DATABASE="peka_other_datasets.csv"
# Which sub-datasets of $DATABASE to build. Leave empty to build every row of the CSV,
# which requires the corresponding WSIs to have been downloaded first.
DATASET_NAMES="kidney_visium_74k liver_visium_37k lung_visium_65k"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "DATABASE_ROOT: $DATABASE_ROOT"
echo "DATASET_NAMES: ${DATASET_NAMES:-<all>}"

cmd="python ${PROJECT_ROOT}/PEKA/peka/Exp_helper/2_peka_dataset_generator.py \
--project_root $PROJECT_ROOT \
--database_root $DATABASE_ROOT \
--datasets_predefine $DATABASE"

if [ -n "$DATASET_NAMES" ]; then
    cmd="$cmd --dataset_names $DATASET_NAMES"
fi

echo "Executing: $cmd"
eval $cmd
