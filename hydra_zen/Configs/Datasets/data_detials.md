# Sub-dataset definitions

Rows of `peka_breast_datasets.csv` / `peka_other_datasets.csv`, resolved against
`support_files/HEST_v1_1_0.csv` (1,229 WSI). The four marked ★ are the paper's benchmarks.

| dataset_name | organ | oncotree_code | platform | WSI | spots (index) |
|---|---|---|---|---|---|
| `breast_in_heast` | Breast |  | Xenium Visium Spatial_Transcriptomics | 125 | 180,694 |
| `breast_xenium_100k` | Breast |  | Xenium | 9 | 104,845 |
| `breast_visium_26k` ★ | Breast |  | Visium | 8 | 30,543 |
| `breast_visium_st_70k` | Breast |  | Visium Spatial_Transcriptomics | 116 | 75,849 |
| `kidney_visium_74k` ★ | Kidney | SCCRCC | Visium | 24 | 74,220 |
| `liver_visium_37k` ★ | Liver |  | Visium | 14 | 64,152 |
| `lung_visium_65k` ★ | Lung |  | Visium | 37 | 69,643 |

Paper tile counts: breast 30,414 · kidney 73,813 · liver 37,168 · lung 64,728 (206,123 total).
Breast and kidney match the index spot counts to within 0.5%. Liver and lung do not, because
HEST records `spots_under_tissue = 4992` — the full Visium array — for most samples of those
two organs, so the index column is an upper bound there, not a tile count.

## Disease state of each benchmark

| dataset | Cancer | Treated | Diseased | Healthy |
|---|---|---|---|---|
| `breast_visium_26k` | 8 | 0 | 0 | 0 |
| `kidney_visium_74k` | 24 | 0 | 0 | 0 |
| `liver_visium_37k` | 2 | 0 | 9 | 3 |
| `lung_visium_65k` | 2 | 23 | 1 | 11 |

Only breast and kidney are cancer-only. `liver_visium_37k` and `lung_visium_65k` are
organ-level Visium slices — that is what reproduces the paper's counts, even though the
paper calls all four cancer cohorts. Every human `disease_state = Cancer` sample in HEST1k,
on every platform, comes to 3 WSI / 8,252 spots for liver and 4 WSI / 17,558 for lung —
a ceiling well below the paper's 37,168 / 64,728. The strictly oncological Visium subsets
are smaller still:

- liver, `oncotree_code=HCC`: 2 WSI / 4,248 spots
- lung, `oncotree_code=LUSC`: 1 WSI / 3,858 spots; `LNET`: 1 WSI / 6,195 spots

`select_hest_ids` matches a single `oncotree_code`, so a multi-code selection needs one CSV
row per code.
