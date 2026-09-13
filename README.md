# PEKA: Parameter Efficient Knowledge Transfer for Accurate Gene Expression Prediction

PEKA is a novel framework that teaches pathology foundation models to accurately predict gene expression using parameter-efficient knowledge transfer techniques. The project combines histopathology image analysis with single-cell genomics through knowledge distillation and Parameter Efficient Fine-tuning (PEFT) to achieve efficient and accurate gene expression prediction from tissue images.

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Step-by-Step Experimental Guide](#step-by-step-experimental-guide)
  - [Prerequisite: scFoundation checkpoint](#prerequisite-scfoundation-checkpoint)
  - [Phase 0: Data Preparation](#phase-0-data-preparation)
  - [Phase 1: Model Training with Dual Encoders](#phase-1-model-training-with-dual-encoders)
  - [Phase 2: Downstream Gene Expression Prediction](#phase-2-downstream-gene-expression-prediction)
  - [Complete Pipeline Execution](#complete-pipeline-execution)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

PEKA addresses the challenge of predicting gene expression patterns from histopathology images by:

1. **Knowledge Distillation**: Transferring knowledge from pre-trained single-cell foundation models to histopathology models
2. **Parameter Efficiency**: Using LoRA (Low-Rank Adaptation),Adaptive LoRA(AdaLoRA),Bone to minimize computational overhead while maintaining performance
3. **Multi-modal Integration**: Combining histopathology images with spatial transcriptomics data
4. **Scalable Pipeline**: Providing an end-to-end workflow from data preprocessing to model evaluation

### Key Features
- Integration with HEST1K dataset for large-scale histopathology analysis
- Support for multiple cancer types (breast, kidney, liver, lung)
- Knowledge distillation framework with dual encoders
- Parameter-efficient fine-tuning using LoRA,Adaptive LoRA(AdaLoRA),Bone
- Comprehensive evaluation pipeline with K-fold cross-validation
- Integration with Weights & Biases for experiment tracking

## Environment Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git with LFS support
- Sufficient storage space (>100GB for full datasets)

### Installation Steps

1. **Clone the Repository**

   The checkout directory must be named `PEKA`: the scripts resolve code and data as
   `<project_root>/PEKA/...`, where `<project_root>` is its parent directory.
   ```bash
   git clone --recursive <repository-url> PEKA
   cd PEKA
   # if you already cloned without --recursive:
   git submodule update --init --recursive
   ```

2. **Set Up Python Environment**
   ```bash
    # follow last step 
    cd /REPO_LOCATION/PEKA/peka/External_models/HEST

    conda create -n "hest" python=3.9
    conda activate hest
    pip install -e .

    pip install     --extra-index-url=https://pypi.nvidia.com     cudf-cu12==24.6.* dask-cudf-cu12==24.6.* cucim-cu12==24.6.*     raft-dask-cu12==24.6.*
    pip install -U 'wandb>=0.12.10'
    pip install hydra_zen peft

    # Optional: Install FAISS-GPU for GPU-accelerated clustering
    pip install faiss-gpu
   ```

3. **Configure Environment Variables**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file with your credentials
   nano .env
   ```
   
   Required environment variables:
   ```
   WANDB_API_KEY=your_wandb_api_key
   HF_TOKEN=your_huggingface_token
   WANDB_ENTITY=your_wandb_entity
   HEST1K_STORAGE_PATH=/path/to/hest1k/storage
   ```

4. **Initialize Project Configuration**
   ```bash
   cd /REPO_LOCATION/PEKA/scripts/0_download_dataset
   bash 0_config_and_download.sh
   ```

   By default this downloads **only the HEST1k samples the benchmarks need** rather than
   the full ~1TB release. The four datasets reported in the paper (breast / kidney /
   liver / lung) come to 107 of the 1,229 WSIs, roughly a tenth of the download. Edit
   `DOWNLOAD_MODE`, `PAPER_ONLY` and `DRY_RUN` at the top of the script, or call the
   downloader directly:
   ```bash
   # see what would be fetched, without downloading
   python ../../peka/Exp_helper/1_dataset_downloader.py --paper_only --dry_run

   # only the four benchmarks from the paper (107 WSI)
   python ../../peka/Exp_helper/1_dataset_downloader.py --paper_only

   # every sub-dataset defined in hydra_zen/Configs/Datasets/peka_*.csv (224 WSI)
   python ../../peka/Exp_helper/1_dataset_downloader.py --mode subset

   # a single benchmark
   python ../../peka/Exp_helper/1_dataset_downloader.py --dataset_names breast_visium_26k

   # the complete HEST1k release (~1TB)
   python ../../peka/Exp_helper/1_dataset_downloader.py --mode full
   ```
   Samples already on disk are skipped, so the command is safe to re-run and to widen
   later (e.g. start with breast, add the other organs afterwards).

### Directory Structure Setup

The initialization script will create the following directory structure:
```
PEKA/
├── DATA/                 # Dataset storage
├── OUTPUT/              # Experiment outputs
├── Pretrained/          # Pre-trained model checkpoints
└── PEKA/               # Source code
```

## Project Structure

```
PEKA/
├── .env.example                    # Environment variables template
├── .gitignore                     # Git ignore rules
├── .gitmodules                    # Git submodules configuration
├── README.md                      # This file
│
├── DATA/                          # Dataset directory (created during setup)
│   └── HEST1K/                   # HEST1K dataset storage
│
├── hydra_zen/                     # Hydra configuration files
│   └── Configs/                  # Model and experiment configurations
│
├── peka/                         # Main source code directory
│   ├── __init__.py              # Package initialization
│   ├── Data/                    # Data processing modules
│   │   ├── dataset_helper.py    # BatchLocalityDataset (image / embedding / label pairing)
│   │   ├── database_helper.py   # Sub-dataset registry and preprocessing status
│   │   ├── hest1k_helper.py     # HEST1K indexing, patch extraction, gene name alignment
│   │   ├── download_helper.py   # Dataset download utilities
│   │   └── utils.py             # HVG selection and expression binning
│   │
│   ├── Model/                   # Model architectures
│   │   ├── LLM/                # scLLM embedders (base.py, scFoundation.py, utils.py)
│   │   ├── base.py             # HistoPath_AlignmentModel and MLPClassifier
│   │   └── utils.py            # Translate module factory
│   │
│   ├── Trainer/                 # Training frameworks
│   │   ├── KD_LoRA.py          # Knowledge distillation with LoRA,Adaptive LoRA(AdaLoRA),Bone,etc.
│   │   ├── pl_basic.py         # Base PyTorch Lightning module
│   │   └── metrics.py          # Metrics factory
│   │
│   ├── External_models/         # External model integrations (git submodules)
│   │   ├── HEST/               # HEST model integration
│   │   └── scFoundation/       # scFoundation model integration
│   │
│   ├── Exp_helper/             # Experiment utilities (numbered pipeline entry points)
│   │   ├── 0_config_runable.py # Environment configuration
│   │   ├── 1_dataset_downloader.py # Dataset download manager
│   │   ├── 2_peka_dataset_generator.py # Patch extraction + gene name alignment
│   │   ├── 3_scLLM_embedding_process.py # scLLM embedding generation
│   │   ├── 5_patch_feature_embeder.py # Frozen image encoder features
│   │   ├── 6_gene_bin_generator.py # HVG selection and binning
│   │   └── 7_generate_cluster_labels_for_KD.py # KD cluster labels
│   │
│   ├── Hydra_helper/           # Hydra configuration helpers
│   │   ├── dataset_part_helpers.py # Dataset/dataloader factory
│   │   ├── model_part_helpers.py   # Model + PEFT factory
│   │   ├── opt_sch_part_helpers.py # Optimizer/scheduler/loss/metrics factory
│   │   ├── trainer_part_helpers.py # pl.Trainer factory
│   │   ├── experiment_helpers.py   # Experiment config snapshot/reload
│   │   └── pl_model_helpers.py     # PyTorch Lightning model helpers
│   │
│   └── DownstreamTasks_helper/ # Downstream task utilities
│       ├── gene_expression_prediction.py # Feature/label loading per feature_type
│       ├── train_and_val_exp.py          # K-fold regression and plots
│       └── inference.py                  # Embedding inference over a dataset folder
│
├── scripts/                      # Experimental pipeline scripts
│   ├── 0_download_dataset/      # Phase 0: Data preparation
│   │   ├── 0_config_and_download.sh        # Initial setup
│   │   ├── 1_generate_peka_datasets_*.sh # Dataset generation
│   │   ├── 2_scLLM_embedding_process.sh     # Embedding processing
│   │   ├── 4_extract_img_features_*.sh      # Image feature extraction
│   │   └── 5_generate_cluster_label_for_KD.sh # Clustering for KD
│   │
│   ├── 1_train_with_2_encoders/ # Phase 1: Model training
│   │   ├── kd_lora_train.py     # Main training script
│   │   ├── kd_lora_train_with_cluster.py # Training with on-the-fly clustering
│   │   ├── kd_lora_inference.py # Inference script
│   │   ├── reproduce_kd_lora_experiment.py # Replay a saved experiment
│   │   ├── simple_train.py      # Simplified training
│   │   └── train_kd_lora.sh     # Training shell script
│   │
│   └── 2_downstream_gene_pred/  # Phase 2: Gene prediction
│       ├── step1_process_hvg.py # Highly variable genes processing
│       ├── step2_inference_feature_vectors.py # Feature extraction
│       ├── step3_task_gene_expr_reg_KFold.py  # K-fold regression
│       ├── 1_generate_labels_*.sh # Label generation
│       ├── 2_auto_reg_KFold_*.sh  # Automated regression
│       └── top_50_genes_*.json    # Gene lists for different cancer types
│
└── support_files/               # Additional support files
    ├── HEST_v1_1_0.csv          # HEST1K index snapshot
    └── scLLM_configs.csv        # scLLM checkpoint registry
```

### Key Components Description

#### Core Modules (`peka/`)
- **Data**: Handles dataset downloading, preprocessing, and loading
- **Model**: Contains model architectures and utilities
- **Trainer**: Implements training frameworks including PEKA
- **External_models**: Integration with external models (HEST, scFoundation)
- **Exp_helper**: Experiment management and configuration utilities
- **Hydra_helper**: Configuration management using Hydra-zen
- **DownstreamTasks_helper**: Utilities for downstream gene prediction tasks

#### Experimental Scripts (`scripts/`)
- **Phase 0** (`0_download_dataset/`): Data preparation and preprocessing
- **Phase 1** (`1_train_with_2_encoders/`): Model training with knowledge distillation
- **Phase 2** (`2_downstream_gene_pred/`): Gene expression prediction and evaluation

## Step-by-Step Experimental Guide

Everything below assumes:

- the checkout is named `PEKA` and `<project_root>` is its **parent** directory
  (the scripts resolve code as `<project_root>/PEKA/...` and data as `<project_root>/PEKA/DATA/...`)
- the `hest` conda environment is active
- `.env` is filled in (see [Environment Variables](#environment-variables))

Every shell script derives `<project_root>` from its own location and `cd`s into its own
directory, so `bash <script>.sh` works from anywhere. The Python entry points do **not**;
run them from the folder documented in each step.

### Prerequisite: scFoundation checkpoint

Step 0.3 loads scFoundation from `<project_root>/PEKA/DATA/<tissue>/Pretrained/scFoundation/`,
where `<tissue>` is `breast` or `other_cancer`:

```
DATA/<tissue>/Pretrained/scFoundation/
├── default_model.ckpt              # pretrained weights      -- download manually
└── OS_scRNA_gene_index.19264.tsv   # gene vocabulary          -- copied automatically
```

The **gene vocabulary is resolved automatically** from the scFoundation submodule, so it
only requires `git submodule update --init --recursive`.

The **weights must be fetched separately** (1.33 GiB). The canonical source is the
[scFoundation SharePoint folder](https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk),
which needs an interactive browser session. On a headless machine, the same file is
mirrored on the Hugging Face Hub:

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p DATA/breast/Pretrained/scFoundation

hf download genbio-ai/scFoundation models.ckpt \
    --local-dir DATA/breast/Pretrained/scFoundation
mv DATA/breast/Pretrained/scFoundation/models.ckpt \
   DATA/breast/Pretrained/scFoundation/default_model.ckpt
```

Verify before use — both known mirrors (`genbio-ai/scFoundation` and the
`jenny143/scFoundation-weight` dataset) resolve to the same bytes:

```bash
sha256sum DATA/breast/Pretrained/scFoundation/default_model.ckpt
# 9f40bf324d3d0084c4b288d06f5af4fddd12206e2a3f022551d12e89e33a0ea9

python -c "
import torch
d = torch.load('DATA/breast/Pretrained/scFoundation/default_model.ckpt',
               map_location='cpu', weights_only=False)
print('modes:', sorted(d.keys()))"   # expects 'gene' among them
```

The checkpoint holds several models keyed by mode (`gene`, `cell`, `rde`); PEKA selects
`gene` via `model_mode="gene"` in `support_files/scLLM_configs.csv`, matching
`--output_type gene` in scFoundation's own `get_embedding.py`.

Because the path is per tissue, symlink rather than copy the multi-GB file when you run
more than one organ:

```bash
mkdir -p DATA/other_cancer/Pretrained/scFoundation
ln -s ../../../breast/Pretrained/scFoundation/default_model.ckpt \
      DATA/other_cancer/Pretrained/scFoundation/default_model.ckpt
```

---

### Phase 0: Data Preparation

#### Step 0.1: Initial Configuration and Dataset Download

```bash
cd scripts/0_download_dataset
bash 0_config_and_download.sh
```

This:
- configures project paths, creates `DATA/`, `OUTPUT/`, `Pretrained/` next to the checkout
- verifies `.env`, `HF_TOKEN` and `HEST1K_STORAGE_PATH`
- downloads the HEST1k samples the benchmarks need

**Download scope.** The script defaults to `PAPER_ONLY=true`, i.e. only the four datasets
reported in the paper. Sizes measured against the 1,229-WSI HEST1k index:

| selection | flag | WSI |
|---|---|---|
| the four paper benchmarks | `--paper_only` | 107 |
| every sub-dataset in `peka_*_datasets.csv` | `--mode subset` (default) | 224 |
| the complete HEST1k release (~1TB) | `--mode full` | 1,229 |

Edit `DOWNLOAD_MODE` / `PAPER_ONLY` / `DRY_RUN` at the top of the script, or call the
downloader directly (from `scripts/0_download_dataset`):

```bash
python ../../peka/Exp_helper/1_dataset_downloader.py --paper_only --dry_run
python ../../peka/Exp_helper/1_dataset_downloader.py --paper_only
python ../../peka/Exp_helper/1_dataset_downloader.py --dataset_names breast_visium_26k
python ../../peka/Exp_helper/1_dataset_downloader.py --mode full
```

Samples already on disk are skipped, so re-running is cheap and the selection can be
widened later (start with breast, add the other organs afterwards).

> **Note on `platform`.** The paper describes the benchmarks as Visium ST only, but
> `peka_other_datasets.csv` declares `platform = Xenium Visium Spatial_Transcriptomics`
> for liver/lung/kidney. `breast_visium_26k` (30,543 spots) and `kidney_in_hest`
> (74,220 spots) already match the paper's 30,414 / 73,813 counts; liver and lung do not,
> because the Xenium slides are pulled in as well. Set `platform` to `Visium` in that CSV
> to reproduce the paper's selection — the downloader reads the same file, so the download
> shrinks to 83 WSI accordingly.

#### Step 0.2: Generate PEKA Datasets

Builds each sub-dataset: extracts `patch_224_0.5_<idx>.h5` tiles from the WSIs and aligns
gene names against Ensembl.

```bash
bash 1_generate_peka_datasets_breast.sh   # breast_visium_26k
bash 1_generate_peka_datasets_other.sh    # kidney_in_hest, liver_in_hest, lung_in_hest
```

`DATASET_NAMES` at the top of each script picks which rows of the predefine CSV to build;
it defaults to the paper benchmarks. Leave it empty to build every row — that requires the
corresponding WSIs to have been downloaded first (`--mode subset`, not `--paper_only`).

Produces, per dataset: `aligned_gene_name/`, `aligned_adata/`, `patches/`, plus the index
`<dataset_name>.csv` and a `dataset_config.csv` tracking preprocessing state.

#### Step 0.3: Process scLLM Embeddings

Runs scFoundation over the aligned AnnData to produce the teacher embeddings.

```bash
bash 2_scLLM_embedding_process.sh
```

Configured by the exported variables at the top of the script, **one dataset per run**:

```bash
export TISSUE_TYPE="breast"            # breast | other_cancer
export DATASET_NAME="breast_visium_26k"
export SCLLM_EMBEDDER_NAME="scFoundation"
```

Produces `scLLM_embed/scFoundation/default_model/{paired_seq,embeddings}/` and a
`meta.csv` recording the embedding dimension and sample count. Repeat for every dataset
(`kidney_in_hest`, `liver_in_hest`, `lung_in_hest` with `TISSUE_TYPE=other_cancer`).

#### Step 0.4: Extract Image Features

Runs the frozen pathology encoders (H-optimus-0, 1536-d and UNI, 1024-d) over the patches.
These features are the `image_encoder` baseline in Phase 2; they are **not** used for KD
training.

```bash
bash 4_extract_img_features_breast.sh
bash 4_extract_img_features_kidney.sh
bash 4_extract_img_features_liver.sh
bash 4_extract_img_features_lung.sh
```

Produces `patches_embed/H-optimus-0/` and `patches_embed/UNI/`.

#### Step 0.5: Generate Cluster Labels for Knowledge Distillation

K-means over the scLLM embeddings; the cluster id becomes the classification target that
teacher and student are both scored on.

```bash
bash 5_generate_cluster_label_for_KD.sh
```

Set `TISSUE_NAME`, `DATASET_NAME`, `SCLLM_EMBEDDER_NAME`, `CKPT` and `N_CLUSTERS` (default
100) at the top of the script; run once per dataset. The label is written back into the
`paired_seq` AnnData as `obs["gen_clustered_label_<N_CLUSTERS>"]`, which is what
`label_name` in the dataset configs refers to.

Skip this step if you train with `kd_lora_train_with_cluster.py`, which clusters on the fly.

#### Step 0.6: Verify Environment and Data Preparation

```bash
bash 3_exp_checker.sh
```

Probes GPU memory and a LoRA forward/backward pass for the configured encoder. It is a
feasibility check, not a data validator — the per-stage consistency checks run
automatically inside `get_preprocess_status` whenever a dataset is loaded.

#### Phase 0 output layout

```
DATA/<tissue>/                       # breast | other_cancer
  dataset_config.csv
  <dataset_name>/                    # e.g. breast_visium_26k
    <dataset_name>.csv               # sample index
    aligned_gene_name/  aligned_adata/
    patches/                         # patch_224_0.5_<idx>.h5 / .png     (0.2)
    scLLM_embed/scFoundation/default_model/
      paired_seq/                    # AnnData + filter_flag + cluster label  (0.3, 0.5)
      embeddings/                    # <...>.npy teacher embeddings           (0.3)
    patches_embed/<backbone>/        # frozen image features                  (0.4)
```

---

### Phase 1: Model Training with Dual Encoders

Training has two phases inside one run:

1. an MLP classifier is trained on the **precomputed scLLM embeddings** to predict the
   cluster labels — this becomes the frozen teacher head;
2. the student (PEFT-adapted image encoder + translate head) maps patches into the same
   space, and the same frozen head scores both sides. The loss is
   `alpha * KL(soft, T=temperature) + (1 - alpha) * CrossEntropy(hard labels)`.

Run from `scripts/1_train_with_2_encoders`. Config paths are relative to
`hydra_zen/Configs/`, and hyperparameters come from those YAMLs — the scripts do not
override them.

#### Step 1.1: Knowledge Distillation with PEFT Training

**Automated (recommended):** edit the variables at the top, then

```bash
bash train_kd_lora.sh
```

**Manual:**
```bash
cd scripts/1_train_with_2_encoders
python kd_lora_train.py \
    --dataset_config Datasets/breast_visium_26k_scFoundation_with_clustered100_label.yaml \
    --model_config Models/H-optimus-0_LoRA_MLP.yaml \
    --optimizer_config Optimizers/kd_lora.yaml \
    --trainer_config Trainers/kd_lora.yaml \
    --exp_name "breast_kd_lora_exp1"
```

Add `--phase1_ckpt <run>/phase1/classifier.pt` to reuse a teacher head and skip phase 1.
Other options: `--phase1_epochs`, `--phase1_lr`, `--phase1_hidden_dim`.

The optimizer config **must** use `CrossEntropyLoss` (`Optimizers/kd_lora.yaml`);
`Optimizers/default.yaml` is the MSE regression variant used by `simple_train.py` and is
rejected by the KD path.

**Swapping the PEFT method.** Pick a different `--model_config`; each declares its method
via `translate_additional_params.peft_method`:

| method | H-optimus-0 | UNI |
|---|---|---|
| LoRA | `Models/H-optimus-0_LoRA_MLP.yaml` | `Models/UNI_LoRA_MLP.yaml` |
| AdaLoRA | `Models/H-optimus-0_AdaLoRA_MLP.yaml` | `Models/UNI_AdaLoRA_MLP.yaml` |
| HRA | `Models/H-optimus-0_HRA_MLP.yaml` | `Models/UNI_HRA_MLP.yaml` |
| Bone | `Models/H-optimus-0_Bone_MLP.yaml` | `Models/UNI_Bone_MLP.yaml` |

**Training with on-the-fly clustering.** Computes the cluster labels during the run, so
Step 0.5 is not required — use a dataset config **without** `_with_clustered100_label`:

```bash
python kd_lora_train_with_cluster.py \
    --dataset_config Datasets/breast_visium_26k_scFoundation.yaml \
    --model_config Models/H-optimus-0_LoRA_MLP.yaml \
    --optimizer_config Optimizers/kd_lora.yaml \
    --trainer_config Trainers/kd_lora.yaml \
    --exp_name "breast_kd_lora_cluster_exp1"
```

**What a run writes:**
```
OUTPUT/kd_lora_<dataset>_<timestamp>/
├── configs/{dataset,model,optimizer,trainer,pl_model}/   # snapshot, used by Step 1.3
├── phase1/classifier.pt                                  # frozen teacher head
└── phase2/lora/                                          # adapter weights + translate_model.pth
Pretrained/<exp_name>/<epoch>-<metric>.ckpt               # best Lightning checkpoint
```

Note that phase 2 saves **adapter weights via `save_pretrained` plus `translate_model.pth`**
rather than a plain Lightning checkpoint whenever `lora_save_path` is set.

#### Step 1.2: Model Inference

```bash
python kd_lora_inference.py \
    --dataset_config Datasets/breast_visium_26k_scFoundation_with_clustered100_label.yaml \
    --model_config Models/H-optimus-0_LoRA_MLP.yaml \
    --trainer_config Trainers/kd_lora.yaml \
    --checkpoint_path ../../Pretrained/<exp_name>/<best>.ckpt \
    --output_dir ../../OUTPUT/inference_results
```

#### Step 1.3: Reproduce Experiments

Replays a previous run from the config snapshot it saved.

```bash
python reproduce_kd_lora_experiment.py --latest --prefix kd_lora_breast
# or an explicit run
python reproduce_kd_lora_experiment.py \
    --experiment_dir ../../OUTPUT/kd_lora_breast_visium_20250101_120000
```

---

### Phase 2: Downstream Gene Expression Prediction

Trains a per-gene regressor on top of frozen embeddings and compares feature sources with
5-fold cross-validation. Run from `scripts/2_downstream_gene_pred`.

`--project_root` is `<project_root>` as defined at the top of this guide — the folder that
*contains* the `PEKA` checkout. `--feature_type` selects the embeddings to regress from:

| `--feature_type` | source on disk |
|---|---|
| `scLLM` | `scLLM_embed/<embedder>/<ckpt>/embeddings/` — the teacher, an upper bound |
| `image_encoder` | `patches_embed/<image_backbone>/` — the frozen baseline |
| `peka` | `peka_embed/<image_encoder_name>/<embedder>/<ckpt>/` — the distilled student |
| `image_encoder+peka` | both of the above, concatenated |

#### Step 2.1: Select the Gene List

The repo ships `top_50_genes_*.json` for each cancer type, so this step is optional.
To recompute the highly variable genes for a dataset:

```bash
cd scripts/2_downstream_gene_pred
python step1_process_hvg.py \
    --input-dir ../../DATA/breast/breast_visium_26k/aligned_adata \
    --output-dir ../../OUTPUT/hvg_analysis \
    --output-filename top_50_genes_breast.json \
    --batch-key sample \
    --n-top-hvg 50
```

#### Step 2.2: Generate Binned Expression Labels

Only needed for the `--use_binned` regression variant; the `raw` variant reads
`aligned_adata` directly.

```bash
bash 1_generate_labels_breast.sh
```

Reads `config_gene_bin_breast.txt` (gene list, `BIN_NB`, HVG options) and writes
`binned_adata/` next to `aligned_adata/`.

#### Step 2.3: Extract PEKA Feature Vectors

Runs the trained student over the patches. Required for `--feature_type peka` and
`image_encoder+peka`; the output path must match the layout the loader expects, i.e.
`peka_embed/<image_encoder_name>/<embedder>/<ckpt>`.

```bash
python step2_inference_feature_vectors.py \
    --project_root /path/to/workspace \
    --tissue_type breast \
    --dataset_name breast_visium_26k \
    --scllm scFoundation \
    --scllm_ckpt default_model \
    --image_encoder_name H0 \
    --model_config Models/H-optimus-0_LoRA_MLP.yaml \
    --model_checkpoint ../../Pretrained/<exp_name>/<best>.ckpt \
    --lora_r 256 \
    --output_dir ../../DATA/breast/breast_visium_26k/peka_embed/H0/scFoundation/default_model
```

`--lora_r` must match the `lora_r` of the model config the checkpoint was trained with.

#### Step 2.4: Gene Expression Regression with K-Fold Validation

**Automated** — sweeps `{raw, binned} × {image_encoder, peka, scLLM, image_encoder+peka}`:

```bash
bash 2_auto_reg_KFold_auto_H0.sh    # H-optimus-0 student
bash 2_auto_reg_KFold_auto_UNI.sh   # UNI student
```

**Manual — a single combination:**
```bash
python step3_task_gene_expr_reg_KFold.py \
    --project_root /path/to/workspace \
    --tissue_type breast \
    --dataset_name breast_visium_26k \
    --embedder_name scFoundation \
    --image_encoder_name H0 \
    --image_backbone H-optimus-0 \
    --feature_type peka \
    --gene_list_json top_50_genes_Visium_Homo_sapien_Breast_Cancer.json \
    --output_root ../../OUTPUT/breast/breast_visium_26k/raw \
    --epochs 300 \
    --mask_zero_values
```

`--image_backbone` names the folder under `patches_embed/` (`H-optimus-0` or `UNI`), while
`--image_encoder_name` (`H0` / `UNI`) names the folder under `peka_embed/`. Add
`--use_binned` to regress against Step 2.2's bins. Genes with fewer than 322 usable spots
are skipped.

Writes `<output_root>/<image_encoder_name>/<feature_type>_gene_level_<raw|binned>_regression_<embedder>/`
containing `gene_regression_results.csv`, `ckpt/` and `plots/`.

#### Step 2.5: Analysis and Visualization

**Compare mutual information** between the teacher embeddings and each model's embeddings:
```bash
python 3_compare_mutual_information.py \
    --project_root /path/to/workspace \
    --tissue_type breast \
    --dataset_name breast_visium_26k \
    --embedder_name scFoundation \
    --output_dir ../../OUTPUT/analysis
```

**Plot gene correlations** — reads the `gene_regression_results.csv` files from Step 2.4,
so `--results_dir` is the `--output_root` used there:
```bash
python 4_plot_gene_correlation.py \
    --results_dir ../../OUTPUT/breast/breast_visium_26k/raw \
    --output_dir ../../OUTPUT/visualizations \
    --tissue Breast \
    --image_encoder_name H0
```

---

### Complete Pipeline Execution

End-to-end for breast, using the paper's subset:

```bash
# --- Phase 0: Data Preparation ---
cd scripts/0_download_dataset
bash 0_config_and_download.sh          # paths + HEST1k subset download
bash 1_generate_peka_datasets_breast.sh
# place the scFoundation checkpoints before the next step (see Prerequisite above)
bash 2_scLLM_embedding_process.sh
bash 4_extract_img_features_breast.sh
bash 5_generate_cluster_label_for_KD.sh
bash 3_exp_checker.sh                  # optional GPU / PEFT feasibility probe

# --- Phase 1: Model Training ---
cd ../1_train_with_2_encoders
bash train_kd_lora.sh

# --- Phase 2: Gene Prediction ---
cd ../2_downstream_gene_pred
bash 1_generate_labels_breast.sh       # only needed for the binned variant
python step2_inference_feature_vectors.py \
    --project_root /path/to/workspace --tissue_type breast \
    --dataset_name breast_visium_26k --scllm scFoundation --scllm_ckpt default_model \
    --image_encoder_name H0 --model_config Models/H-optimus-0_LoRA_MLP.yaml \
    --model_checkpoint ../../Pretrained/<exp_name>/<best>.ckpt --lora_r 256 \
    --output_dir ../../DATA/breast/breast_visium_26k/peka_embed/H0/scFoundation/default_model
bash 2_auto_reg_KFold_auto_H0.sh
python 3_compare_mutual_information.py \
    --project_root /path/to/workspace --tissue_type breast \
    --dataset_name breast_visium_26k --output_dir ../../OUTPUT/analysis
python 4_plot_gene_correlation.py \
    --results_dir ../../OUTPUT/breast/breast_visium_26k/raw \
    --output_dir ../../OUTPUT/visualizations --tissue Breast --image_encoder_name H0
```

For the other cancer types: use `1_generate_peka_datasets_other.sh`,
`4_extract_img_features_{kidney,liver,lung}.sh`, set `TISSUE_TYPE=other_cancer` and
`DATASET_NAME=<kidney|liver|lung>_in_hest` in Steps 0.3 and 0.5, and use the matching
`Datasets/<dataset>_scFoundation*.yaml` config in Phase 1.

## Configuration

### Environment Variables

Ensure your `.env` file contains:
```
WANDB_API_KEY=your_wandb_api_key_here
HF_TOKEN=your_huggingface_token_here
WANDB_ENTITY=your_wandb_entity_name
HEST1K_STORAGE_PATH=/path/to/hest1k/dataset
REPO_PATH=/path/to/peka/repository
DATA_PATH=/path/to/data/directory
PROJECT_PATH=/path/to/project/root
```

### Model Configuration

Configuration is hydra-zen based: each YAML under `hydra_zen/Configs/` has a `_target_`
pointing at a *factory function* in `peka/Hydra_helper/`, and the training scripts
`load_from_yaml` then `instantiate` it with the few values only known at runtime. The
YAML is the single source of truth for hyperparameters — the scripts do not override it.

| Group | Factory | Produces | Pick it with |
|---|---|---|---|
| `Datasets/` | `dataset_part_helpers.dataset_generator` | `(train_loader, val_loader, embedding_dim)` | `--dataset_config` |
| `Models/` | `model_part_helpers.model_config` | encoder + PEFT adapter + translate head | `--model_config` |
| `Optimizers/` | `opt_sch_part_helpers.opt_sch_config` | optimizer, scheduler, loss, metrics | `--optimizer_config` |
| `Trainers/` | `trainer_part_helpers.trainer_config` | a configured `pl.Trainer` | `--trainer_config` |
| `PL_Model/` | plain dict | KD parameters (`num_classes`, `temperature`, `alpha`, ...) | loaded automatically |

Things worth knowing when editing these:

- `target_dim` is **not** set in the model YAML; it is taken from the dataset's embedding
  dimension at runtime, which is what ties the student output to the scLLM space.
- `Models/*.yaml` select the PEFT method through `translate_additional_params.peft_method`
  (`lora`, `adalora`, `hra`, `bone`); method-specific knobs (`target_r`, `init_r`,
  `apply_GS`, `init_weights`) live in the same block.
- `target_modules` in a PEFT config is ignored: the model applies the adapter to
  `blocks.<i>.attn.qkv` of every transformer block.
- `ckpt_para.monitor` in a trainer config must name a metric the run actually logs
  (`Trainers/kd_lora.yaml` monitors `val_CosineSimilarity`, `Trainers/default.yaml`
  monitors `MSE_val`).
- Dataset configs named `*_with_clustered100_label.yaml` set `label_name` to the cluster
  label produced by Step 0.5; the plain ones leave it null.
- `hydra_zen/Configs/Datasets/peka_{breast,other}_datasets.csv` define which HEST1k slices
  exist. They drive the download, the dataset build, and the binning step alike.

### Experiment Tracking

PEKA uses Weights & Biases for experiment tracking. Ensure you have:
1. A WANDB account
2. Proper API key in your `.env` file
3. Correct entity name configured

## Troubleshooting

### Common Issues

1. **Paths ending in `/PEKA/...` do not exist**
   - The checkout directory must be named exactly `PEKA`; every script resolves code and
     data as `<parent>/PEKA/...`. Rename the folder rather than patching the scripts.

2. **Import Errors (`No module named 'hest'` / `'scFoundation'` / `'peka'`)**
   - Initialize the submodules: `git submodule update --init --recursive`, then
     `pip install -e peka/External_models/HEST`
   - Activate the `hest` environment
   - Run the Python entry points from the folder named in their step; only the shell
     wrappers are location independent

3. **`FileNotFoundError` on `default_model.ckpt`**
   - Step 0.3 needs the scFoundation weights placed manually — see
     [Prerequisite: scFoundation checkpoint](#prerequisite-scfoundation-checkpoint).
     The error message prints the download link and the exact target path.
   - If the message is about `OS_scRNA_gene_index.19264.tsv` instead, the submodule is not
     initialized: `git submodule update --init --recursive`

4. **Dataset Download Issues**
   - Check internet connection and `HEST1K_STORAGE_PATH` permissions
   - Ensure sufficient disk space — use `--paper_only` or `--dataset_names` to fetch only
     the benchmarks you actually run instead of the full ~1TB release
   - `--dry_run` lists the sample IDs that would be downloaded without fetching them

5. **Step 0.2 fails on a sub-dataset you never downloaded**
   - `1_generate_peka_datasets_*.sh` builds whatever `DATASET_NAMES` lists, defaulting to
     the paper benchmarks. Either widen the download (`--mode subset`) or narrow
     `DATASET_NAMES` to what is on disk.

6. **`AssertionError: task <name> not in predefined datasets`**
   - The dataset config's `task` must match a `dataset_name` row in
     `DATA/<tissue>/dataset_config.csv`, i.e. Step 0.2 must have built that sub-dataset.

7. **`ValueError: KD_LoRA model requires CrossEntropyLoss`**
   - Pass `--optimizer_config Optimizers/kd_lora.yaml`; `Optimizers/default.yaml` is the
     MSE regression variant.

8. **Gene name alignment or patch counts reported as inconsistent**
   - `get_preprocess_status` compares file counts per stage. Delete the affected
     `aligned_adata/` or `patches/` folder and re-run Step 0.2.

9. **CUDA Out of Memory**
   - Reduce `batch_size` in the dataset config
   - Lower `lora_r` in the model config, or use a smaller `patch_size`
   - Run `bash 3_exp_checker.sh` to probe what the GPU can hold

10. **HuggingFace Authentication**
    - Verify `HF_TOKEN` is valid and has accepted the gated model terms for
      `bioptimus/H-optimus-0` and `MahmoodLab/UNI`
    - Try `huggingface-cli login`

11. **WANDB Issues**
    - Verify `WANDB_API_KEY` and `WANDB_ENTITY` in `.env`
    - Try `wandb login`, or set `with_logger` to something other than `wandb` in the
      trainer config to disable logging

### Performance Optimization

1. **Memory Optimization**
   - Use gradient checkpointing
   - Implement data loading optimizations
   - Consider model parallelism for large models

2. **Speed Optimization**
   - Use mixed precision training
   - Optimize data loading with multiple workers
   - Consider distributed training for multiple GPUs

### Getting Help

- Check the issue tracker for known problems
- Review configuration files for parameter settings
- Verify environment setup using the configuration script
- Check WANDB logs for detailed training information

---

**Note**: This project requires significant computational resources and storage space. Ensure your system meets the hardware requirements before beginning the experimental pipeline.
