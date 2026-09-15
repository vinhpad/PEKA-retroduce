# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

PEKA distills knowledge from a single-cell foundation model (scFoundation) into a pathology
image encoder (H-optimus-0 / UNI) using PEFT adapters, so that histology patches can be mapped
into the scLLM gene-expression embedding space. The distilled embeddings are then evaluated on a
downstream gene-expression regression task.

There is no test suite, no `setup.py`/`pyproject.toml`, and no lint config. The repo is a
research pipeline driven by shell scripts + argparse entry points, so "running it" means running
a pipeline stage, not `pytest`.

## Environment

```bash
git submodule update --init --recursive      # HEST and scFoundation are submodules and ship empty
conda create -n hest python=3.9 && conda activate hest
cd peka/External_models/HEST && pip install -e .
pip install hydra_zen peft wandb python-dotenv timm pytorch-lightning torchmetrics scanpy
cp .env.example .env                          # then fill it in
```

`.env` lives at the repo root and must define `WANDB_API_KEY`, `WANDB_ENTITY`, `HF_TOKEN`,
`HEST1K_STORAGE_PATH`, `REPO_PATH`, `DATA_PATH`, `PROJECT_PATH`. Every entry point loads it with
`dotenv` from a path derived from `PROJECT_ROOT`, not from the CWD.

## The directory-name contract (read this before debugging any path error)

Nearly every script computes `PROJECT_ROOT` as the **parent** of the repo directory and then
hard-codes `${PROJECT_ROOT}/PEKA/...` for code and `${PROJECT_ROOT}/PEKA/DATA/...` for data.
Consequences:

- The checkout directory **must be named `PEKA`**. This clone is `PEKA-retroduce`, so scripts will
  resolve paths that do not exist until it is renamed or the paths are adjusted.
- Shell scripts derive `PROJECT_ROOT` from `$(readlink -f "$0")`, but a few (`2_scLLM_embedding_process.sh`,
  `5_generate_cluster_label_for_KD.sh`) derive it from `$(pwd)` — those **must be run from their own
  directory**.
- `scripts/1_train_with_2_encoders/*.py` derive roots from `os.getcwd()`, so run them from inside
  `scripts/1_train_with_2_encoders/`.
- `0_config_runable.py` creates `DATA/`, `OUTPUT/`, `Pretrained/` **one level above** the repo
  (siblings of `PEKA/`), while training writes datasets under `PEKA/DATA/`. Both roots are in use.

## Pipeline

Three ordered phases under `scripts/`. Each stage writes into the dataset folder and the next
stage asserts on what the previous one produced (`peka.Data.database_helper.get_preprocess_status`
reconciles counts and rewrites `dataset_config.csv`).

```bash
# Phase 0 — data prep (run each from scripts/0_download_dataset/)
bash 0_config_and_download.sh              # path config + HEST1K download (subset by default)
bash 1_generate_peka_datasets_breast.sh    # or ..._other.sh; gene-name alignment + patch extraction
bash 2_scLLM_embedding_process.sh          # scFoundation embeddings; configured via exported env vars, not flags
bash 4_extract_img_features_breast.sh      # frozen H-optimus-0 (1536-d) and UNI (1024-d) patch features
bash 5_generate_cluster_label_for_KD.sh    # k-means (default 100 clusters) over scLLM embeddings -> KD labels
bash 3_exp_checker.sh                      # GPU/LoRA feasibility probe

# Phase 1 — distillation training (run from scripts/1_train_with_2_encoders/)
python kd_lora_train.py \
  --dataset_config Datasets/breast_visium_26k_scFoundation_with_clustered100_label.yaml \
  --model_config Models/H-optimus-0_LoRA_MLP.yaml \
  --optimizer_config Optimizers/kd_lora.yaml \
  --trainer_config Trainers/kd_lora.yaml \
  --exp_name my_run                        # add --phase1_ckpt to skip phase 1

# Phase 2 — downstream gene expression regression (run from scripts/2_downstream_gene_pred/)
bash 1_generate_labels_breast.sh           # HVG selection + expression binning
python step2_inference_feature_vectors.py --project_root ... --model_checkpoint ...
python step3_task_gene_expr_reg_KFold.py --feature_type peka --gene_list_json top_50_genes_*.json ...
```

Config paths passed to the Phase 1 scripts are **relative to `hydra_zen/Configs/`**.

`kd_lora_train_with_cluster.py` is the variant that computes cluster labels on the fly instead of
requiring step 5; `reproduce_kd_lora_experiment.py` re-runs a previous run by reading the config
snapshot that `save_experiment_configs` copied into `OUTPUT/<exp>_<timestamp>/configs/`.

## Architecture

**Config-as-code via hydra-zen.** `hydra_zen/Configs/` holds four groups — `Datasets/`, `Models/`,
`Optimizers/`, `Trainers/` (plus a plain-dict `PL_Model/`). Each YAML's `_target_` points at a
*factory function* in `peka/Hydra_helper/`, not at a class. Entry points do
`load_from_yaml(...)` then `instantiate(cfg, <runtime overrides>)`:

| Config group | Factory | Produces |
|---|---|---|
| `Datasets/` | `dataset_part_helpers.dataset_generator` | `(train_loader, val_loader, embedding_dim)` |
| `Models/` | `model_part_helpers.model_config` | `HistoPath_AlignmentModel` with a PEFT-wrapped encoder |
| `Optimizers/` | `opt_sch_part_helpers.opt_sch_config` | `(optimizer partials, scheduler partials, MetricsFactory, loss partial)` |
| `Trainers/` | `trainer_part_helpers.trainer_config` | a configured `pl.Trainer` (wandb logger + checkpoint callback) |

Optimizers and losses are returned as `functools.partial`, bound to parameters later in
`pl_basic.configure_optimizers`. `target_dim` is *not* in the YAML — it comes from the dataset's
embedding dimension at runtime, which is what couples the student output to the scLLM space.

**Model.** `peka/Model/base.py::HistoPath_AlignmentModel` = timm encoder + a `translate_model`
head (`Model/utils.py::get_module`, `MLP` or `Transformer`). PEFT is applied in the constructor,
which **overwrites `lora_paras.target_modules`** with `blocks.{i}.attn.qkv` for every block —
`target_modules` in a config is therefore ignored. `model_part_helpers` selects between
`LoraConfig` / `AdaLoraConfig` / `HRAConfig` / `BoneConfig` via
`translate_additional_params.peft_method` (default `lora`), which is also where AdaLoRA/HRA/Bone
specific knobs (`target_r`, `init_r`, `apply_GS`, `init_weights`, …) are read from.

**Two-phase training** (`peka/Trainer/KD_LoRA.py`):
1. `pl_KD_LoRA.train_phase1` — a plain PyTorch loop (no Lightning) trains an `MLPClassifier` on the
   *precomputed scLLM embeddings* to predict cluster labels. This becomes the frozen teacher head.
2. `pl_KD_LoRA` (Lightning) — the student maps images through the PEFT encoder + translate head into
   the same space; the same frozen classifier scores both student features and teacher embeddings.
   Loss = `alpha * KL(soft, T=temperature) + (1-alpha) * CrossEntropy(hard labels)`.
   Both phases require the loss to be `CrossEntropyLoss` (`pl_model_helpers.create_pl_model` asserts this).

`on_save_checkpoint` is overridden: when `lora_save_path` is set it *additionally* saves
**adapter weights via `save_pretrained` plus `translate_model.pth`** into `OUTPUT/<exp>_<ts>/phase2/lora/`.
The hook runs inside Lightning's own save, so `ModelCheckpoint` still writes the full `.ckpt`
under `Pretrained/<exp_name>/` — and that `.ckpt` is what Step 2.3 loads
(`step2_inference_feature_vectors.py` reads `checkpoint['state_dict']`; the `lora/` folder has no such key).

**Dataset** (`peka/Data/dataset_helper.py::BatchLocalityDataset`). Samples are keyed by barcode
across three stores: `.h5ad` paired sequences, `.npy` embeddings, `.h5` patches. It keeps one file
open at a time and rotates every `batch_switch_interval` batches — hence "batch locality" — so
shuffling is only within the current file. The train/val split is computed by the train instance
and handed to the val instance via `train_val_split_dict`; never construct the val dataset
independently. Batches are `(img, emb, label)`.

## On-disk dataset layout

```
DATA/<tissue>/                       # breast | other_cancer
  dataset_config.csv                 # per-dataset preprocessing state, rewritten by get_preprocess_status
  <dataset_name>/                    # e.g. breast_visium_26k
    <dataset_name>.csv               # sample index
    aligned_gene_name/ aligned_adata/
    patches/                         # patch_{patch_size}_{pixel_size}*.h5 / .png
    patches_embed/<backbone>/        # frozen image-encoder features
    binned_adata/                    # expression bins for downstream regression
    scLLM_embed/<embedder>/<ckpt>/{paired_seq,embeddings}/
    peka_embed/<image_encoder>/<embedder>/<ckpt>/   # distilled student features
```

`feature_type` in Phase 2 selects among these: `scLLM`, `image_encoder`, `peka`,
`image_encoder+peka` (see `DownstreamTasks_helper/gene_expression_prediction.py::get_dataset_paths`).

## HEST1K download scope

`1_dataset_downloader.py` defaults to `--mode subset`: it reads the sub-dataset definitions
in `hydra_zen/Configs/Datasets/peka_*.csv`, resolves them to HEST sample IDs through
`hest1k_helper.collect_required_hest_ids`, and `snapshot_download`s only those samples.
That helper shares `select_hest_ids` with `construct_sub_dataset_index`, so the download
set and the set the pipeline later indexes cannot drift apart — if you change a
definition CSV, both follow.

All four paper benchmarks are declared `platform = Visium`, `species = Homo sapiens`,
which is what reproduces the paper's sample counts (breast 30,414 / kidney 73,813 /
liver 37,168 / lung 64,728 tiles). Note that only breast and kidney are cancer-only —
kidney via `oncotree_code = SCCRCC`, breast because every human Visium breast sample in
HEST1k is IDC/ILC. Liver and lung include healthy/diseased/treated samples; restricting
them to cancer would leave 2 WSI each and would not match the paper.

Sizes against the 1,229-WSI index: `--paper_only` (breast_visium_26k 8, kidney_visium_74k 24,
liver_visium_37k 14, lung_visium_65k 37) = 83 WSI; all defined sub-datasets = 200 WSI;
`--mode full` = the whole ~1TB release. Samples are skipped when `st/<id>.h5ad` already exists, so
widening the selection later only fetches the difference.

## Gotchas that are still live

- **The checkout must be named `PEKA`** (see above). Nothing else in the repo works around this.
- The shell launchers now `cd` into their own directory because the Python entry points read
  `os.getcwd()`; if you invoke a `scripts/**/*.py` directly, run it from its own folder.
- `gene_name_alignment` writes every tissue's aligned AnnData as `HEST_breast_adata_<idx>.h5ad`,
  and `scLLM_QC_preprocess.run`, `BatchLocalityDataset` and the downstream loaders all hard-code
  that prefix. It is not breast-specific despite the name — don't "fix" one side alone.
- `HistoPath_AlignmentModel.__init__` overwrites `lora_paras.target_modules` with
  `blocks.{i}.attn.qkv`, so `target_modules` in a PEFT config is ignored by design.
- A trainer config's `ckpt_para.monitor` must name a metric the run actually logs
  (`Trainers/kd_lora.yaml` monitors `val_CosineSimilarity`, `Trainers/default.yaml` monitors
  `MSE_val`); a mismatch surfaces only once Lightning tries to checkpoint.
- `model_part_helpers.model_config` calls `.get()` on `translate_additional_params`, so that
  field must be a mapping in YAML. `get_module` additionally accepts the `ListConfig`
  list-of-dicts form, but `model_config` would fail on it first.
- Phase 2 runs **plain 5-fold**: `step3_task_gene_expr_reg_KFold.py` now forwards
  `--with_independent_test_set` (it used to drop it, letting the callee's `True` default
  shrink every fold's training set from 4/5 to 0.8x0.8 = 64%). Default is off, matching the
  paper. Results produced before this fix are not comparable with results after it.
- The Phase 2 sweeps (`2_auto_reg_KFold_auto_{H0,UNI}.sh`) and Step 2.3
  (`1b_inference_peka_features.sh`) iterate all four benchmarks and **skip** datasets whose
  data or checkpoint is missing, printing a reason. `DATASETS`, `FEATURE_TYPES` and
  `BINNED_OPTIONS` override from the environment; binned is off by default since the paper
  reports raw expression only.
- There is still no test suite; verification means running a pipeline stage.
