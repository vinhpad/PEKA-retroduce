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
    - [Where the 5-fold actually is](#where-the-5-fold-actually-is)
    - [Reading the 5-fold numbers](#reading-the-5-fold-numbers)
    - [Flags that do not do what they look like](#flags-that-do-not-do-what-they-look-like)
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

3. **Pin the fragile dependencies**

   Almost nothing in this stack declares an upper version bound, so a plain
   `pip install -U <anything>` can silently walk numpy, numba, pyarrow or peft out of the
   range the pipeline needs — and the breakage shows up at runtime, far from the install.
   `constraints.txt` at the repo root records every such bound:

   ```bash
   pip install -c constraints.txt -U <package>     # always cap installs with it
   pip check                                       # must stay silent
   ```

   | pin | why |
   |---|---|
   | `numpy==1.26.4` | numba 0.60 wants <1.27; cudf-cu12 24.6 wants <2.0 |
   | `numba==0.60.0`, `llvmlite==0.43.0` | newer numba mistypes numpy 1.26 ufuncs inside scanpy |
   | `pyarrow==16.1.0`, `datasets<5` | cudf-cu12 24.6 caps pyarrow <16.2; datasets 5 needs >=21 |
   | `peft>=0.14.0,<0.19` | `BoneConfig` — the paper's adapter — exists only in this range |
   | `transformers<5` | HEST is written against the 4.x API |
   | `opencv-python<=4.11.0.86` | 4.12 requires numpy>=2 |

4. **Configure Environment Variables**
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

5. **Initialize Project Configuration**
   ```bash
   cd /REPO_LOCATION/PEKA/scripts/0_download_dataset
   bash 0_config_and_download.sh
   ```

   By default this downloads **only the HEST1k samples the benchmarks need** rather than
   the full ~1TB release. The four datasets reported in the paper (breast / kidney /
   liver / lung) come to 83 of the 1,229 WSIs, well under a tenth of the download. Edit
   `DOWNLOAD_MODE`, `PAPER_ONLY` and `DRY_RUN` at the top of the script, or call the
   downloader directly:
   ```bash
   # see what would be fetched, without downloading
   python ../../peka/Exp_helper/1_dataset_downloader.py --paper_only --dry_run

   # only the four benchmarks from the paper (83 WSI)
   python ../../peka/Exp_helper/1_dataset_downloader.py --paper_only

   # every sub-dataset defined in hydra_zen/Configs/Datasets/peka_datasets.csv (200 WSI)
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
| the four paper benchmarks | `--paper_only` | 83 |
| every sub-dataset in `peka_datasets.csv` | `--mode subset` (default) | 200 |
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

> **Note on `platform`.** All four benchmarks are **Visium, Homo sapiens**, so
> `peka_datasets.csv` declares `platform = Visium` for every one of them. This is what
> reproduces the paper's sample counts; the four rows resolve to 83 WSI against the
> 1,229-WSI HEST1k index:
>
> | dataset | HEST selection | WSI | paper n |
> |---|---|---|---|
> | `breast_visium_26k` | Breast + Visium | 8 | 30,414 |
> | `kidney_visium_74k` | Kidney + Visium + `oncotree_code=SCCRCC` | 24 | 73,813 |
> | `liver_visium_37k` | Liver + Visium | 14 | 37,168 |
> | `lung_visium_65k` | Lung + Visium | 37 | 64,728 |
>
> Earlier revisions declared `Xenium Visium Spatial_Transcriptomics` for liver/lung/kidney,
> which pulled the Xenium slides in as well (107 WSI). Kidney was unaffected — its SCCRCC
> samples are all Visium — which is why only liver and lung counts moved.
>
> **`liver_visium_37k` and `lung_visium_65k` are not cancer-only.** Section 3.1 of the paper
> describes all four as cancer ("Visium ST data of Homo Sapiens with breast cancer
> (n=30,414 ...), kidney cancer (n=73,813), liver cancer (n=37,168), and lung cancer
> (n=64,728)"), but HEST1k cannot supply that. Counting *every* human sample with
> `disease_state = Cancer`, on every platform — an absolute ceiling — liver has 3 WSI /
> 8,252 spots and lung 4 WSI / 17,558. The paper needs 4.5x and 3.7x those ceilings, so no
> oncological filter reproduces its counts; only the organ-level Visium slices do.
>
> The authors' own definitions agree with the numbers rather than with the prose: in the
> definition CSV as first committed (`peka_other_datasets.csv`, since merged into
> `peka_datasets.csv`), `oncotree_code` is empty for liver and lung and
> set only for kidney (`SCCRCC`). Breast and kidney *are* cancer-only, but as a consequence
> of their definitions rather than a disease filter — every human Visium breast sample in
> HEST1k is IDC or ILC. So describe these two benchmarks as human Visium liver / lung with
> mixed disease state (liver: 2 Cancer, 9 Diseased, 3 Healthy; lung: 2 Cancer, 23 Treated,
> 1 Diseased, 11 Healthy), not as cancer cohorts.
>
> There is no `disease_state` filter in `select_hest_ids`; add `oncotree_code` per row if
> you want a strictly oncological subset — and expect 2 WSI per organ, which is too small
> to train on and is not what the paper ran.
>
> **Index version.** Every count above is against `support_files/HEST_v1_1_0.csv`, the index
> bundled with this repo and the newest one published on the HEST GitHub. HuggingFace also
> ships `HEST_v1_1_1`, `HEST_v1_2_0`, `HEST_v1_2_1` and `HEST_v1_3_0`, which add samples —
> so the same definitions resolve to *more* WSI there. `get_hest_db_index` picks the
> highest-numbered `HEST_*.csv` present in `HEST1K_STORAGE_PATH` and only falls back to the
> bundled copy when the folder has none, so dropping a newer index in is enough to switch.
> To see what that changes before committing to it:
>
> ```bash
> # the HEST repo is gated — accept the terms once, then use your HF_TOKEN
> curl -L -H "Authorization: Bearer $HF_TOKEN" \
>   -o $HEST1K_STORAGE_PATH/HEST_v1_3_0.csv \
>   https://huggingface.co/datasets/MahmoodLab/hest/resolve/main/HEST_v1_3_0.csv
>
> python peka/Exp_helper/8_dataset_definition_checker.py \
>   --index support_files/HEST_v1_1_0.csv \
>   --index $HEST1K_STORAGE_PATH/HEST_v1_3_0.csv
> ```
>
> It prints WSI + spot counts per definition side by side and lists which sample IDs each
> version adds or drops. Switching index mid-project changes the sample set a dataset
> resolves to, so rebuild any dataset folder built against the old one.

#### Step 0.2: Generate PEKA Datasets

Builds each sub-dataset: extracts `patch_224_0.5_<idx>.h5` tiles from the WSIs and aligns
gene names against Ensembl.

```bash
bash 1_generate_peka_datasets_breast.sh   # breast_visium_26k
bash 1_generate_peka_datasets_other.sh    # kidney_visium_74k, liver_visium_37k, lung_visium_65k
```

`DATASET_NAMES` at the top of each script picks which rows of `peka_datasets.csv` to build;
it defaults to the paper benchmarks. That file holds every tissue while `DATABASE_ROOT`
points at one of them (`DATA/breast/` or `DATA/other_cancer/`), so keep the list to rows of
that tissue — leaving it empty builds all 7 rows into whichever folder the script targets.
Building the extra breast sub-datasets also requires their WSIs to be on disk
(`--mode subset`, not `--paper_only`).

Produces, per dataset: `aligned_gene_name/`, `aligned_adata/`, `patches/`, plus the index
`<dataset_name>.csv` and a `dataset_config.csv` tracking preprocessing state.

**The Ensembl gene map is cached, not fetched every run.** `gene_name_alignment` needs a
symbol <-> Ensembl-ID mapping; `support_files/ensembl_gene_map.tsv` ships one (Ensembl
release 116, 49,131 IDs / 41,859 symbols), so alignment is deterministic and works offline.
Delete that file to refresh it against the current release — `fetch_ensembl_gene_map`
then queries `martservice` directly on `www` / `useast` / `asia`, validates the payload and
rewrites the cache.

This replaced the `biomart` package, which made the step fail in a way worth knowing about:
a mirror that is down answers **HTTP 200 with an HTML "Service unavailable" page**, and
`www.ensembl.org` 308-redirects to the current archive host during release transitions, so
a down mirror surfaces as

```
xml.etree.ElementTree.ParseError: mismatched tag: line 85, column 2
```

from deep inside `biomart/dataset.py`. Status codes are not a usable health signal here;
`fetch_ensembl_gene_map` checks whether the body is markup instead, moves on to the next
mirror, and reports every mirror's failure reason if none work.

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
(`kidney_visium_74k`, `liver_visium_37k`, `lung_visium_65k` with `TISSUE_TYPE=other_cancer`).

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

K-means over the scLLM embeddings writes the initial cluster-label column required by the
dataset loader. With the default prototype loss enabled, Phase 1 does **not** train on these
full-dataset assignments: it refits K-means on the training split and assigns validation spots
to the frozen train centroids before training the classifier.

```bash
bash 5_generate_cluster_label_for_KD.sh
```

Set `TISSUE_NAME`, `DATASET_NAME`, `SCLLM_EMBEDDER_NAME`, `CKPT` and `N_CLUSTERS` (default
100) at the top of the script; run once per dataset. The label is written back into the
`paired_seq` AnnData as `obs["gen_clustered_label_<N_CLUSTERS>"]`, which is what
`label_name` in the dataset configs refers to.

You may skip this step only when using `kd_lora_train_with_cluster.py --dataset_folder ...`;
that entry point creates the initial column before loading the labeled dataset config.

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
   space, and the same frozen head scores both sides. The default loss is
   `alpha * KL(soft, T=temperature) + (1 - alpha) * CrossEntropy(hard labels)
   + prototype_loss_weight * KL(student prototype distribution || teacher prototype distribution)`.

For leakage-free clustering, the 80/20 split is created first. K-means and teacher prototypes
are fit only on normalized **training** embeddings; validation labels are assigned by the fixed
train centroids. After `prototype_warmup_epochs`, student prototypes update once per epoch by
EMA and are anchored back to the fixed teacher prototypes. Validation never contributes to
K-means fitting or EMA updates.

Run from `scripts/1_train_with_2_encoders`. Config paths are relative to
`hydra_zen/Configs/`, and hyperparameters come from those YAMLs — the scripts do not
override them.

This phase uses a **single 80/20 train/val split** (`split_seed=42`), not cross-validation.
The paper's 5-fold evaluation belongs to [Phase 2](#phase-2-downstream-gene-expression-prediction).

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
    --model_config Models/H-optimus-0_Bone_MLP.yaml \
    --optimizer_config Optimizers/kd_lora.yaml \
    --trainer_config Trainers/kd_lora.yaml \
    --exp_name "breast_kd_lora_exp1"
```

Select another benchmark with, for example:

```bash
DATASET=kidney bash train_kd_lora.sh   # breast | kidney | liver | lung
```

Do not pass an old `--phase1_ckpt` while prototype loss is enabled: every run refits train-only
clusters, so an old classifier's class IDs need not match the new centroids. To reuse the legacy
fixed-label classifier path, first set `prototype_loss_weight: 0.0` in
`hydra_zen/Configs/PL_Model/kd_lora.yaml`. Other options are `--phase1_epochs`, `--phase1_lr`,
and `--phase1_hidden_dim`.

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

**Bootstrap labels during training.** This avoids running Step 0.5 separately. Pass the dataset
folder so the script creates the initial `gen_clustered_label_100` column, but still use the
labeled dataset config; the prototype stage then replaces those initial assignments with the
train-only clusters used by Phase 1 and Phase 2:

```bash
python kd_lora_train_with_cluster.py \
    --dataset_folder ../../DATA/breast/breast_visium_26k \
    --dataset_config Datasets/breast_visium_26k_scFoundation_with_clustered100_label.yaml \
    --model_config Models/H-optimus-0_Bone_MLP.yaml \
    --optimizer_config Optimizers/kd_lora.yaml \
    --trainer_config Trainers/kd_lora.yaml \
    --exp_name "breast_kd_lora_cluster_exp1"
```

`kd_lora_train_with_cluster.py` currently bootstraps scFoundation/default_model with 100
clusters. For other values, generate labels explicitly in Step 0.5 and keep `num_classes`, the
dataset config's `label_name`, and `N_CLUSTERS` identical.

**What a run writes:**
```
OUTPUT/kd_lora_<dataset>_<timestamp>/
├── configs/{dataset,model,optimizer,trainer,pl_model}/   # snapshot, used by Step 1.3
├── phase1/classifier.pt                                  # frozen teacher head
└── phase2/lora/                                          # adapter weights + translate_model.pth
Pretrained/<exp_name>/<epoch>-<metric>.ckpt               # best Lightning checkpoint
```

Both are written. `on_save_checkpoint` is overridden so that, when `lora_save_path` is set,
phase 2 *also* dumps **adapter weights via `save_pretrained` plus `translate_model.pth`**
into `phase2/lora/`; the hook runs inside Lightning's own save, so `ModelCheckpoint` still
writes the full `.ckpt` under `Pretrained/`. Step 2.3 needs that `.ckpt` — it reads
`checkpoint['state_dict']`, which `phase2/lora/` does not contain.

#### Step 1.2: Model Inference

```bash
python kd_lora_inference.py \
    --dataset_config Datasets/breast_visium_26k_scFoundation_with_clustered100_label.yaml \
    --model_config Models/H-optimus-0_Bone_MLP.yaml \
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

This is where the paper's **5-fold cross-validation** happens. Run from
`scripts/2_downstream_gene_pred`.

#### Where the 5-fold actually is

A frequent misreading: the folds are **not** part of KD training. The two phases split
data differently, and only Phase 2 cross-validates.

| | Phase 1 — `kd_lora_train.py` | Phase 2 — `step3_task_gene_expr_reg_KFold.py` |
|---|---|---|
| split | a **single** 80/20 split, `split_seed=42` | `KFold(n_splits=5, shuffle=True, random_state=2025)` |
| trains | image encoder + PEFT adapter + translate head | one regressor **per gene** |
| model | H-optimus-0 / UNI + LoRA·AdaLoRA·HRA·Bone | PCA(256) + Ridge |
| runs | once | 5 folds × N genes |

The encoder is distilled once; the folds apply only to the regressor sitting on top of
frozen embeddings. This follows the HEST1k benchmark protocol (see the docstring of
`peka/DownstreamTasks_helper/train_and_val_exp.py`).

Per fold, with `N` usable spots for a gene:

```
N spots
├── test   20%   ← held out, scored. The 5 test folds are disjoint and cover all N.
└── rest   80%
    ├── train  64%   ← what the regressor actually sees
    └── val    16%   ← carved out but unused (Ridge has no early stopping)
```

Note the file also contains `train_and_val_step()`, a plain single 80/20 split with no
folds. It is imported but never called — `step3_task_gene_expr_reg_KFold.py` always uses
the K-fold variant.

#### What each feature type needs

`--project_root` is `<project_root>` as defined at the top of this guide — the folder that
*contains* the `PEKA` checkout. `--feature_type` selects the embeddings to regress from.

**Every** feature type reads `scLLM_embed/<embedder>/<ckpt>/paired_seq/` — that is where
the gene expression labels and the `filter_flag` QC mask live — plus `patches/` for the
barcode intersection. So Step 0.3 is required even for the image-only baseline; there is
no shortcut around the scFoundation checkpoint.

| `--feature_type` | prerequisite steps | embedding source |
|---|---|---|
| `scLLM` | 0.2 + 0.3 | `scLLM_embed/<embedder>/<ckpt>/embeddings/` — the teacher, an upper bound |
| `image_encoder` | 0.2 + 0.3 + 0.4 | `patches_embed/<image_backbone>/` — the frozen baseline |
| `peka` | + Phase 1 + Step 2.3 | `peka_embed/<image_encoder_name>/<embedder>/<ckpt>/` — the distilled student |
| `image_encoder+peka` | all of the above | both, concatenated |

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
cd scripts/2_downstream_gene_pred
bash 1b_inference_peka_features.sh                 # every benchmark that has a checkpoint
DATASETS="breast" bash 1b_inference_peka_features.sh
IMAGE_ENCODER_NAME=UNI MODEL_CONFIG=Models/UNI_Bone_MLP.yaml \
    bash 1b_inference_peka_features.sh             # the UNI student
```

The wrapper picks the newest `.ckpt` in `Pretrained/KD_LoRA_<tissue>_<dataset>_scFoundation_clustered100/`,
reads `lora_r` out of the model config and `target_scllm_dim` out of an existing scLLM
embedding, writes to the exact path the loader expects, and then **counts the `.npy` files
it produced** — `step2_inference_feature_vectors.py` swallows save errors and skips indices
whose patch `.h5` is missing, yet still prints `Inference completed successfully`.

Datasets without data or without a checkpoint are skipped with a reason, so running it
after training breast alone is safe. To call the entry point directly instead:

```bash
python step2_inference_feature_vectors.py \
    --project_root /path/to/workspace \
    --tissue_type breast --dataset_name breast_visium_26k \
    --scllm scFoundation --scllm_ckpt default_model \
    --image_encoder_name H0 \
    --model_config Models/H-optimus-0_Bone_MLP.yaml \
    --model_checkpoint ../../Pretrained/<exp_name>/<best>.ckpt \
    --lora_r 256 --target_scllm_dim <scLLM embedding width> \
    --output_dir ../../DATA/breast/breast_visium_26k/peka_embed/H0/scFoundation/default_model
```

`--model_config` must be the config the checkpoint was trained with (Bone for PEKA itself),
and `--lora_r` must match that config's `lora_r`. `--model_checkpoint` is the Lightning
`.ckpt` under `Pretrained/`, **not** `OUTPUT/<exp>_<ts>/phase2/lora/` — the loader reads
`checkpoint['state_dict']`, which the adapter folder does not have.

#### Step 2.4: Gene Expression Regression with K-Fold Validation

The 5 folds run automatically — there is no flag to enable them.

**Build up in stages** rather than waiting for a full pipeline; each rung produces real
5-fold numbers and validates the machinery before the next one costs GPU time.

*Rung 1 — right after Step 0.3, no training needed.* Regressing from the teacher
embeddings gives the upper bound and proves Phase 2 works end to end:

```bash
cd scripts/2_downstream_gene_pred
python step3_task_gene_expr_reg_KFold.py \
    --project_root /path/to/workspace \
    --tissue_type breast \
    --dataset_name breast_visium_26k \
    --embedder_name scFoundation \
    --image_encoder_name H0 \
    --feature_type scLLM \
    --gene_list_json top_50_genes_Visium_Homo_sapien_Breast_Cancer.json \
    --output_root ../../OUTPUT/breast/breast_visium_26k/raw \
    --mask_zero_values
```

*Rung 2 — after Step 0.4.* The frozen image-encoder baseline:

```bash
python step3_task_gene_expr_reg_KFold.py \
    --project_root /path/to/workspace \
    --tissue_type breast --dataset_name breast_visium_26k \
    --embedder_name scFoundation --image_encoder_name H0 \
    --image_backbone H-optimus-0 \
    --feature_type image_encoder \
    --gene_list_json top_50_genes_Visium_Homo_sapien_Breast_Cancer.json \
    --output_root ../../OUTPUT/breast/breast_visium_26k/raw \
    --mask_zero_values
```

*Rung 3 — after Phase 1 and Step 2.3.* The paper's headline result; swap
`--feature_type peka` or `image_encoder+peka` into the command above, or sweep everything:

```bash
bash 2_auto_reg_KFold_auto_H0.sh    # H-optimus-0 student
bash 2_auto_reg_KFold_auto_UNI.sh   # UNI student
```

Each script sweeps `{breast, kidney, liver, lung} × {image_encoder, peka, scLLM,
image_encoder+peka}` on raw expression — the paper reports no binned variant. A dataset
with no `DATA/` folder, and a feature type whose embeddings are not on disk, are skipped
with a reason rather than crashing, so this is safe to run after building breast only.
Override from the environment:

```bash
DATASETS="breast" bash 2_auto_reg_KFold_auto_H0.sh
FEATURE_TYPES="peka" bash 2_auto_reg_KFold_auto_H0.sh
BINNED_OPTIONS="false true" bash 2_auto_reg_KFold_auto_H0.sh   # adds the binned half (needs Step 2.2)
```

`--image_backbone` names the folder under `patches_embed/` (`H-optimus-0` or `UNI`), while
`--image_encoder_name` (`H0` / `UNI`) names the folder under `peka_embed/`.

#### Reading the 5-fold numbers

Results land in
`<output_root>/<image_encoder_name>/<feature_type>_gene_level_<raw|binned>_regression_<embedder>/`
as `gene_regression_results.csv` (plus `ckpt/` and `plots/`). One row per gene; for each of
`mse`, `pearson_correlation`, `cosine_similarity`, `kl_divergence` there is a `_mean` and a
`_std` **across the 5 folds**. The number to compare against the paper is
`pearson_correlation_mean`.

```bash
python -c "
import pandas as pd, glob
for f in sorted(glob.glob('../../OUTPUT/breast/breast_visium_26k/raw/*/*/gene_regression_results.csv')):
    d = pd.read_csv(f)
    print(f.split('/raw/')[1])
    print('  PCC 5-fold: %.4f ± %.4f  (%d genes)' % (
        d.pearson_correlation_mean.mean(), d.pearson_correlation_std.mean(), len(d)))
"
```

#### Flags that do not do what they look like

- **`--epochs` is a no-op.** `train_regressor()` fits PCA + Ridge in closed form; there is
  no training loop, and `epochs`/`patience` are dead variables. The `--epochs 300` in the
  sweep scripts changes nothing.
- **`--with_independent_test_set` now works, and defaults to off.** It used to be dropped
  on the floor — `step3` stored it in its `config` dict and never passed it on, so the
  callee's own default (`True`) always won and every fold trained on 0.8 x 0.8 = 64% of the
  data instead of the 4/5 that "5-fold cross-validation" means. It is forwarded now and the
  default is plain 5-fold, which is the paper's protocol. Passing the flag restores the
  extra 20% validation carve-out, but nothing consumes it: `train_regressor` fits PCA+Ridge
  in closed form and returns that model unconditionally, so it only shrinks training data.
  **Numbers produced before this fix are not comparable with numbers produced after it.**
- **`Ksplit=5` is hard-coded** in `step3_task_gene_expr_reg_KFold.py`; there is no CLI flag
  for a different number of folds.
- **Genes with 321 or fewer usable spots are skipped entirely** and never appear in the
  CSV. With `--mask_zero_values`, spots where the gene reads zero are dropped first, so the
  usable count is per gene — this is why a 50-gene list can yield fewer than 50 rows.

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

### Reproducing the paper

Settings from [PEKA, arXiv:2504.07061](https://arxiv.org/html/2504.07061v1) mapped onto
this repo. Values marked ✅ are already the shipped defaults.

| Paper | Where | Value to use |
|---|---|---|
| PEKA adapter = Block-Affine (**Bone**) | `--model_config` | `Models/H-optimus-0_Bone_MLP.yaml` |
| LoRA / AdaLoRA are **baselines**, not PEKA | `--model_config` | `Models/H-optimus-0_{LoRA,AdaLoRA}_MLP.yaml` |
| r=256, α=32, dropout=0.1 | `Models/*.yaml` | ✅ `lora_r: 256`, `lora_alpha: 32`, `lora_dropout: 0.1` |
| Adam, lr 1e-4 | `Optimizers/kd_lora.yaml` | ✅ |
| 50 epochs | `Trainers/kd_lora.yaml` | ✅ `max_epochs: 50` |
| λ₁=λ₂=0.5 (KD vs structure loss) | `PL_Model/kd_lora.yaml` | ✅ `alpha: 0.5` |
| PCA(256) + Ridge, top-50 HVG, 5-fold | Phase 2 | ✅ |
| scFoundation teacher | Step 0.3 | ✅ |
| Visium ST only | `Datasets/peka_datasets.csv` | ✅ `platform: Visium` on all four (see Step 0.1 note) |

**Bone requires `peft>=0.14.0`** — `BoneConfig` does not exist before that, and the error
message from `model_part_helpers` says so. `pip install -U "peft>=0.14.0"`.

Not stated in the paper: KD temperature (`temperature: 2.0`), number of cluster labels
(`num_classes: 100`; the paper says "k-nearest neighbor clustering" without giving k), and
the prototype-loss/EMA settings below. Treat prototype mode as an additional experiment and
set `prototype_loss_weight: 0.0` when reproducing the previous fixed-label objective exactly.
The paper also does not state batch size; `batch_size: 8` × `accumulate_grad_batches: 4` here
gives an effective batch of 32 and fits a 24GB card.

The paper reports ~5% of backbone parameters as trainable; `kd_lora_train.py` prints the
trainable count at startup, so you can check that directly.

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
bash 5_generate_cluster_label_for_KD.sh  # bootstrap column; Phase 1 refits train-only clusters
bash 3_exp_checker.sh                  # optional GPU / PEFT feasibility probe

# --- Phase 1: Model Training ---
cd ../1_train_with_2_encoders
bash train_kd_lora.sh

# --- Phase 2: Gene Prediction ---
cd ../2_downstream_gene_pred
bash 1_generate_labels_breast.sh       # only needed for the binned variant
bash 1b_inference_peka_features.sh      # step 2.3, every benchmark with a checkpoint
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
`DATASET_NAME` to `kidney_visium_74k` / `liver_visium_37k` / `lung_visium_65k` in
Steps 0.3 and 0.5, and use the matching
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
  column produced by Step 0.5. With prototype loss enabled, this column lets the loader start,
  then its in-memory train/validation labels are replaced from train-fitted centroids before
  Phase 1. Plain configs leave `label_name` null and cannot feed the KD classifier.
- `hydra_zen/Configs/Datasets/peka_{breast,other}_datasets.csv` define which HEST1k slices
  exist. They drive the download, the dataset build, and the binning step alike.

Prototype clustering is configured in `hydra_zen/Configs/PL_Model/kd_lora.yaml`:

| Setting | Default | Meaning |
|---|---:|---|
| `prototype_loss_weight` | `0.1` | Weight added to the existing KD objective; set `0.0` for the legacy fixed-label path |
| `prototype_temperature` | `0.1` | Temperature for teacher/student cosine-to-prototype distributions |
| `prototype_warmup_epochs` | `5` | Keep teacher prototypes fixed before EMA adaptation starts |
| `prototype_ema_momentum` | `0.95` | Momentum applied to student centroids between epochs |
| `teacher_anchor_weight` | `0.5` | Fraction of the fixed teacher centroid retained after each EMA update |
| `cluster_diagnostic_sample_size` | `2000` | Train-only sample used for clustering diagnostics; `0` disables score computation |

At startup the run reports and logs train-only Silhouette, Calinski-Harabasz,
Davies-Bouldin, and min/max cluster sizes. During Phase 2, W&B receives
`{train,val}_prototype_loss` and `{train,val}_prototype_accuracy`. Dynamic prototypes update
from training batches only; distributed runs all-reduce prototype sums/counts before EMA.

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

8. **`--phase1_ckpt cannot be reused when prototype loss refits train-only clusters`**
   - This is intentional: the saved classifier's class IDs belong to a previous set of
     centroids. Remove `--phase1_ckpt` and retrain Phase 1. For a legacy fixed-label run,
     set `prototype_loss_weight: 0.0` before reusing the classifier.

9. **Gene name alignment or patch counts reported as inconsistent**
   - `get_preprocess_status` compares file counts per stage. Delete the affected
     `aligned_adata/` or `patches/` folder and re-run Step 0.2.

10. **Dataset generation writes to `nan/<dataset>` and then reports a missing index**
    - `dataset_config.csv` contains an empty/stale `dataset_storage_folder`. Re-run the
      relevant `1_generate_peka_datasets_*.sh`; the launcher now refreshes runtime storage and
      HEST paths in the existing CSV. A stray `nan/` folder from the failed run is unused and
      may be removed after confirming the real dataset directory is correct.

11. **CUDA Out of Memory**
   - Reduce `batch_size` in the dataset config
   - Lower `lora_r` in the model config, or use a smaller `patch_size`
   - Run `bash 3_exp_checker.sh` to probe what the GPU can hold

12. **HuggingFace Authentication**
    - Verify `HF_TOKEN` is valid and has accepted the gated model terms for
      `bioptimus/H-optimus-0` and `MahmoodLab/UNI`
    - Try `huggingface-cli login`

13. **WANDB Issues**
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
