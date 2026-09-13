import logging
import os
from pathlib import Path

log_level = logging.DEBUG
logger = logging.getLogger("PEKA")
# check if logger has been initialized
if not logger.hasHandlers():
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# get current location
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = str(Path(REPO_DIR).parent)

# set environment variables
# NOTE: these must be absolute, otherwise they silently depend on the CWD and the
# external model imports fail with ModuleNotFoundError.
import sys

EXTERNAL_MODELS_DIR = os.path.join(REPO_DIR, "External_models")
for _external_path in (
    EXTERNAL_MODELS_DIR,
    os.path.join(EXTERNAL_MODELS_DIR, "HEST", "src"),
    os.path.join(EXTERNAL_MODELS_DIR, "scFoundation"),
):
    if _external_path not in sys.path:
        sys.path.append(_external_path)
