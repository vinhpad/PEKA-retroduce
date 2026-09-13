"""
Factory for the scLLM embedders used in Phase 0 (2_scLLM_embedding_process.sh).

Every embedder subclasses `peka.Model.LLM.base.scLLM_QC_preprocess` and exposes the
same `run()` / `valid_check()` interface, so the pipeline only needs a name to pick one.
"""
from typing import Any, Dict, Type

from peka import logger
from peka.Model.LLM.base import scLLM_QC_preprocess


def _get_embedder_registry() -> Dict[str, Type[scLLM_QC_preprocess]]:
    """Import embedders lazily: each one pulls in heavy external model code."""
    registry: Dict[str, Type[scLLM_QC_preprocess]] = {}

    from peka.Model.LLM.scFoundation import scFoundation_embedder

    registry["scFoundation"] = scFoundation_embedder
    return registry


def get_scLLM_embedder(data_root: str,
                       dataset_name: str,
                       scLLM_embedder_name: str,
                       **kwargs: Any) -> scLLM_QC_preprocess:
    """Instantiate the scLLM embedder registered under `scLLM_embedder_name`.

    Args:
        data_root: tissue folder, e.g. `<project>/PEKA/DATA/breast`
        dataset_name: sub dataset name, e.g. `breast_visium_26k`
        scLLM_embedder_name: key from `support_files/scLLM_configs.csv` (e.g. `scFoundation`)
        **kwargs: embedder specific parameters, e.g. `ckpt_name`, `model_mode`

    Returns:
        An embedder instance exposing `run()` and `valid_check()`.
    """
    registry = _get_embedder_registry()
    if scLLM_embedder_name not in registry:
        raise ValueError(
            f"Unknown scLLM embedder: {scLLM_embedder_name}. "
            f"Available: {sorted(registry.keys())}"
        )

    embedder_cls = registry[scLLM_embedder_name]
    logger.info(f" 🤖 building {scLLM_embedder_name} embedder for {dataset_name} with {kwargs}")
    return embedder_cls(
        data_root=data_root,
        dataset_name=dataset_name,
        scLLM_embedder_name=scLLM_embedder_name,
        **kwargs,
    )
