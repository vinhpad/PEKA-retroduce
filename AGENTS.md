# AGENTS.md

## Repository Contract

- This is a GPU research pipeline, not an installable/tested Python package: there is no root
  `pyproject.toml`, test suite, lint config, or CI. Verify changes with the narrowest affected
  argparse entry point or pipeline stage; do not claim `pytest`/lint coverage.
- Initialize both submodules before debugging imports:
  `git submodule update --init --recursive`. Install `peka/External_models/HEST` editable.
- Use Python 3.9 and constrain every pip operation with the root file, for example
  `pip install -c constraints.txt <packages>` followed by `pip check`. The pins are functional:
  notably NumPy 1.26.4, `datasets==4.0.0`, `transformers<5`, and `peft>=0.14,<0.19`
  (Bone was removed in PEFT 0.19).
- Copy `.env.example` to root `.env`. Current entry points actively read `WANDB_API_KEY`,
  `WANDB_ENTITY`, `HF_TOKEN`, and `HEST1K_STORAGE_PATH`; the three `*_PATH` template fields are
  currently unused because paths are derived in code.

## Path Contract

- The checkout must be named `PEKA`. Scripts derive its parent as `PROJECT_ROOT`, then hard-code
  `${PROJECT_ROOT}/PEKA`; this checkout is currently named `PEKA-retroduce`, so full pipeline runs
  will resolve nonexistent paths unless it is renamed or the path code is deliberately changed.
- Shell launchers derive paths from their own location and generally `cd` where required, so they
  can be launched from anywhere. Direct `scripts/1_train_with_2_encoders/*.py` calls are different:
  they derive roots from `os.getcwd()` and must run inside that directory.
- Two storage roots intentionally coexist. Dataset content is under `PEKA/DATA/`, while
  `0_config_runable.py`, training outputs, Lightning checkpoints, and downstream results use
  sibling directories `${PROJECT_ROOT}/{DATA,OUTPUT,Pretrained}`. Do not normalize these paths
  without tracing every consumer.

## Pipeline And Focused Commands

- Stages are ordered and mutate dataset state; `get_preprocess_status` also reconciles counts and
  rewrites `DATA/<tissue>/dataset_config.csv`.
- Data prep launchers are in `scripts/0_download_dataset/`:
  `0_config_and_download.sh` -> `1_generate_peka_datasets_{breast,other}.sh` ->
  `2_scLLM_embedding_process.sh` -> relevant `4_extract_img_features_*.sh` ->
  `5_generate_cluster_label_for_KD.sh`. Edit/export each launcher's dataset variables; the scLLM
  and cluster-label scripts process one dataset per run.
- Before a large download, run from `scripts/0_download_dataset/`:
  `python ../../peka/Exp_helper/1_dataset_downloader.py --paper_only --dry_run`.
  `--paper_only` is 83 benchmark WSIs; default subset definitions live in
  `hydra_zen/Configs/Datasets/peka_datasets.csv`; `--mode full` is roughly 1 TB.
- Train from `scripts/1_train_with_2_encoders/` with `bash train_kd_lora.sh`; select a benchmark via
  `DATASET=kidney bash train_kd_lora.sh` (also `breast`, `liver`, `lung`). Config arguments passed
  to Python are relative to `hydra_zen/Configs/`.
- Downstream, run `scripts/2_downstream_gene_pred/1b_inference_peka_features.sh` before
  `2_auto_reg_KFold_auto_{H0,UNI}.sh`. Narrow expensive runs with `DATASETS="breast"`,
  `FEATURE_TYPES="peka"`, and `BINNED_OPTIONS="false"`; launchers skip missing artifacts.
- There is no cheap end-to-end smoke test. For Python-only edits, at minimum use
  `python -m compileall <changed paths>`; for behavior, use a launcher's dry-run/argument path or
  the smallest affected stage with existing data and report unavailable GPU/data prerequisites.

## Architecture Constraints

- Hydra YAML `_target_` values point to factories in `peka/Hydra_helper/`, not model classes.
  Dataset factory returns `(train_loader, val_loader, embedding_dim)`; that runtime dimension is
  passed to the model as `target_dim`. Optimizer/loss factories return `functools.partial` objects.
- KD has two phases in `peka/Trainer/KD_LoRA.py`: a plain-PyTorch classifier over precomputed
  scLLM embeddings, then Lightning training of the image student against the frozen classifier.
  KD configs must use `CrossEntropyLoss`; `create_pl_model` rejects other losses.
- With prototype loss enabled, training refits clusters on the train split, assigns validation by
  those centroids, and updates student prototypes by teacher-anchored EMA. Do not reuse an old
  `--phase1_ckpt`: its classifier class IDs belong to different centroids.
- `HistoPath_AlignmentModel` overwrites PEFT `target_modules` with every
  `blocks.{i}.attn.qkv`; a YAML `target_modules` setting is ignored. Select LoRA/AdaLoRA/HRA/Bone
  through the mapping `translate_additional_params.peft_method`.
- `translate_additional_params` must remain a mapping because `model_part_helpers.model_config`
  calls `.get()` on it. Trainer `ckpt_para.monitor` must match a logged metric; KD currently logs
  and monitors `val_CosineSimilarity`.
- `BatchLocalityDataset` joins `.h5ad`, `.npy`, and patch `.h5` records by barcode. Construct the
  validation dataset with the train instance's `train_val_split_dict`; an independent validation
  instance computes a different split.
- The misleading `HEST_breast_adata_<idx>` prefix is a cross-pipeline storage contract for every
  tissue. Writers, scFoundation processing, datasets, inference, and downstream loaders all use it;
  never rename one side alone.

## Artifacts And Evaluation

- Lightning writes full `.ckpt` files under `${PROJECT_ROOT}/Pretrained/<exp_name>/` and also saves
  adapter files plus `translate_model.pth` under `${PROJECT_ROOT}/OUTPUT/<run>/phase2/lora/`.
  Downstream feature inference requires the full `.ckpt` because it reads `checkpoint['state_dict']`.
- `1b_inference_peka_features.sh` checks output file counts because the Python inference path can
  skip missing patch files or swallow save errors while still printing success. Preserve that check.
- Phase 2 defaults to plain 5-fold CV. Do not add `--with_independent_test_set` unless intentionally
  requesting an extra holdout; old results made with the formerly unforwarded/default-true behavior
  are not comparable. The sweep's `--epochs` value is currently a no-op because regression is
  closed-form PCA + Ridge.
- The shipped Ensembl map `support_files/ensembl_gene_map.tsv` makes alignment deterministic.
  Deleting it triggers BioMart mirror requests; mirrors can return an HTML error with HTTP 200, so
  validate response content rather than status alone.
