"""
This file is used to download the HEST1k dataset.

By default only the samples that the predefined sub-datasets actually need are
downloaded (`--mode subset`). The full HEST1k release is around 1TB, while the four
benchmarks reported in the paper (breast / kidney / liver / lung) need roughly a tenth
of that. Use `--mode full` to mirror the whole repository.

for following steps:
    (1) config project path
    (2) config python import path add peka and external models
    (3) resolve which HEST1k samples are required
    (4) check .env file which save key environment variables
    (5) check HF_TOKEN
    (6) check HEST1K_STORAGE_PATH
"""

print("🤖 running 1_dataset_downloader.py")
# config runable path
import sys
import os
proj_path = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
sys.path.append(proj_path)
print(f" ⭐️ proj_path: {proj_path}")

# add python import path
code_path = str(proj_path) + "/PEKA/"
sys.path.append(code_path)
print(f" ⭐️ code_path: {code_path}")
# add external models path
external_module_path = str(proj_path) + "/PEKA/peka/External_models/"
sys.path.append(external_module_path)
print(f" ⭐️ external_module_path: {external_module_path}")
sys.path.append(external_module_path + "/HEST/src/")

# import peka
from peka import logger
from peka.Data.download_helper import download_hest1k, download_hest1k_subset, hest_ids_on_disk
from peka.Data.hest1k_helper import collect_required_hest_ids
import dotenv

import argparse

# the four benchmarks reported in the paper
PAPER_DATASETS = ["breast_visium_26k", "kidney_in_hest", "liver_in_hest", "lung_in_hest"]


def get_args():
    parser = argparse.ArgumentParser(description="Download the HEST1k dataset.")
    parser.add_argument('--mode', choices=['subset', 'full'], default='subset',
                        help="'subset' downloads only the samples the predefined "
                             "sub-datasets need (default); 'full' mirrors all of HEST1k (~1TB).")
    parser.add_argument('--datasets_predefine', nargs='+',
                        default=["peka_breast_datasets.csv", "peka_other_datasets.csv"],
                        help='Predefine CSVs under hydra_zen/Configs/Datasets/ that '
                             'describe which sub-datasets exist.')
    parser.add_argument('--dataset_names', nargs='*', default=None,
                        help='Only download these sub-datasets, e.g. '
                             f'{" ".join(PAPER_DATASETS)}')
    parser.add_argument('--paper_only', action='store_true',
                        help=f'Shorthand for --dataset_names {" ".join(PAPER_DATASETS)}')
    parser.add_argument('--force_download', action='store_true',
                        help='Re-download samples that are already present.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Report what would be downloaded and exit.')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    env_file_path = str(proj_path) + "/PEKA/.env"
    print(f"⭐️ loading env file from {env_file_path}")
    dotenv.load_dotenv(dotenv_path=env_file_path)

    hest_storage_path = os.getenv("HEST1K_STORAGE_PATH")
    hf_token = os.getenv("HF_TOKEN")
    if not hest_storage_path:
        raise ValueError("HEST1K_STORAGE_PATH is not set in .env")
    os.makedirs(hest_storage_path, exist_ok=True)

    if args.mode == "full":
        print("🚨 downloading the complete HEST1k database (~1TB).")
        if args.dry_run:
            print("📃 dry run: stopping before download.")
            sys.exit(0)
        download_hest1k(hest_storage_path, hf_token)
        sys.exit(0)

    dataset_names = PAPER_DATASETS if args.paper_only else args.dataset_names
    required_ids, per_dataset = collect_required_hest_ids(
        hest_loc=hest_storage_path,
        dataset_predefine_list=args.datasets_predefine,
        dataset_names=dataset_names,
    )

    present, missing = hest_ids_on_disk(hest_storage_path, required_ids)
    to_download = required_ids if args.force_download else missing

    print("📃 required sub-datasets:")
    for name, count in per_dataset.items():
        print(f"    - {name}: {count} WSI")
    print(f"📃 {len(required_ids)} WSI required | {len(present)} already local | "
          f"{len(to_download)} to download")

    if args.dry_run:
        print("📃 dry run: stopping before download.")
        print(f"📃 ids: {to_download}")
        sys.exit(0)

    if not to_download:
        print(f"📃 HEST1k subset already complete in: {hest_storage_path}")
    else:
        download_hest1k_subset(hest_storage_path, to_download, hf_token)
        _, still_missing = hest_ids_on_disk(hest_storage_path, required_ids)
        if still_missing:
            logger.warning(f"🚨 {len(still_missing)} samples still missing after download: "
                           f"{still_missing}")
        else:
            print(f"📃 HEST1k subset complete in: {hest_storage_path}")
