"""
Resolve the sub-dataset definitions against one or more HEST1k index versions.

Answers two questions:
  * what does each row of hydra_zen/Configs/Datasets/peka_*.csv actually select?
  * does a newer HEST1k index change that selection?

Mirrors `peka.Data.hest1k_helper.select_hest_ids` (organ + species + optional
oncotree_code + optional platform list). Deliberately stdlib-only, so it runs
outside the conda env — `hest1k_helper` pulls in hest/torch/scanpy.

    python 8_dataset_definition_checker.py
    python 8_dataset_definition_checker.py --index ../../support_files/HEST_v1_1_0.csv \
                                           --index /path/to/HEST_v1_3_0.csv
"""
import argparse
import csv
import os

PAPER_DATASETS = ["breast_visium_26k", "kidney_visium_74k", "liver_visium_37k", "lung_visium_65k"]
# tile counts reported in PEKA (arXiv 2504.07061), for reference
PAPER_N = {"breast_visium_26k": 30414, "kidney_visium_74k": 73813,
           "liver_visium_37k": 37168, "lung_visium_65k": 64728}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_index(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def select(index_rows, organ, species, platform_list=None, oncotree_code=None):
    """Same predicate as hest1k_helper.select_hest_ids."""
    out = []
    for r in index_rows:
        if r["species"] != species or r["organ"] != organ:
            continue
        if oncotree_code and r["oncotree_code"] != oncotree_code:
            continue
        if platform_list and r["st_technology"] not in platform_list:
            continue
        out.append(r)
    return out


def spots(rows):
    total = 0
    for r in rows:
        try:
            total += int(float(r["spots_under_tissue"]))
        except (TypeError, ValueError):
            pass
    return total


def definitions(predefines):
    for predefine in predefines:
        path = predefine if os.path.isabs(predefine) else \
            os.path.join(REPO, "hydra_zen/Configs/Datasets", predefine)
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                platform = str(row["platform"]).strip()
                yield row["dataset_name"], dict(
                    organ=row["organ"],
                    species=row["species"],
                    platform_list=[p.replace("_", " ") for p in platform.split()] or None,
                    oncotree_code=row["oncotree_code"] or None,
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="append", default=None,
                    help="HEST index CSV; repeat to compare versions (default: support_files/HEST_v1_1_0.csv)")
    ap.add_argument("--predefine", action="append",
                    default=["peka_datasets.csv"])
    ap.add_argument("--paper_only", action="store_true", help="only the four paper benchmarks")
    args = ap.parse_args()

    indexes = args.index or [os.path.join(REPO, "support_files/HEST_v1_1_0.csv")]
    loaded = [(os.path.basename(p), read_index(p)) for p in indexes]
    for name, rows in loaded:
        print(f"index {name}: {len(rows)} WSI")
    print()

    header = f"{'dataset':<24}" + "".join(f"{n:>26}" for n, _ in loaded) + f"{'paper n':>10}"
    print(header)
    print("-" * len(header))

    unions = [set() for _ in loaded]
    paper_unions = [set() for _ in loaded]
    for name, sel in definitions(args.predefine):
        if args.paper_only and name not in PAPER_DATASETS:
            continue
        cells = []
        for i, (_, rows) in enumerate(loaded):
            hit = select(rows, **sel)
            ids = {r["id"] for r in hit}
            unions[i] |= ids
            if name in PAPER_DATASETS:
                paper_unions[i] |= ids
            cells.append(f"{len(hit):>6} WSI {spots(hit):>10,}")
        mark = " *" if len({c.split()[0] for c in cells}) > 1 else ""
        paper = f"{PAPER_N[name]:,}" if name in PAPER_N else "-"
        print(f"{name:<24}" + "".join(f"{c:>26}" for c in cells) + f"{paper:>10}{mark}")

    print()
    print(f"{'--paper_only union':<24}" + "".join(f"{len(u):>20} WSI" for u in paper_unions))
    print(f"{'--mode subset union':<24}" + "".join(f"{len(u):>20} WSI" for u in unions))

    if len(loaded) > 1:
        print("\nper-dataset sample churn vs first index:")
        base_name, base_rows = loaded[0]
        for name, sel in definitions(args.predefine):
            if args.paper_only and name not in PAPER_DATASETS:
                continue
            base = {r["id"] for r in select(base_rows, **sel)}
            for other_name, other_rows in loaded[1:]:
                new = {r["id"] for r in select(other_rows, **sel)}
                added, removed = sorted(new - base), sorted(base - new)
                if added or removed:
                    print(f"  {name} ({base_name} -> {other_name}): "
                          f"+{len(added)} -{len(removed)}")
                    if added:
                        print(f"      added:   {', '.join(added[:12])}{' ...' if len(added) > 12 else ''}")
                    if removed:
                        print(f"      removed: {', '.join(removed[:12])}{' ...' if len(removed) > 12 else ''}")


if __name__ == "__main__":
    main()
