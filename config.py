"""Paths and the switch between a full run and a reviewer run.

The Allen Visual Behavior Neuropixels dataset is not distributed with this
project because of its size. To reproduce preprocessing, download it from

    https://portal.brain-map.org/circuits-behavior/visual-behavior-neuropixels

and point ALLEN_RAW_DATA_DIR at the local cache (the directory that contains
``manifests/``).

RUN_FROM_RAW
    False (default, and the Code Ocean setting): do not preprocess and do not
    rebuild per-session tables. Analyses use the derived tables in DATA_DIR.
    True: full reproduction. Preprocess the raw cache into
    ALLEN_PREPROCESSED_DIR, build the derived tables, then run the analyses.

Environment variables override the defaults: ALLEN_RAW_DATA_DIR,
ALLEN_PREPROCESSED_DIR, ALLEN_CACHE_PARENT, DATA_DIR, RESULTS_DIR, RUN_FROM_RAW
(``1`` / ``0``).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _env_path(name, default):
    value = os.environ.get(name)
    if value:
        return Path(value)
    return Path(default)


# Directory that contains manifests/ for the AllenSDK local cache.
ALLEN_RAW_DATA_DIR = _env_path("ALLEN_RAW_DATA_DIR", "/data/AllenNPXVisualBehavior")

# Per-session tables written by code_gendata/preprocess_data.py.
ALLEN_PREPROCESSED_DIR = _env_path(
    "ALLEN_PREPROCESSED_DIR", "/data/allen_engagement_data"
)

# AllenSDK wants a sibling folder named visual-behavior-neuropixels. Keep that
# symlink out of DATA_DIR so the derived tables can be copied on their own.
ALLEN_CACHE_PARENT = _env_path("ALLEN_CACHE_PARENT", ROOT / ".allen_sdk_cache")

DATA_DIR = _env_path("DATA_DIR", ROOT / "data")
RESULTS_DIR = _env_path("RESULTS_DIR", ROOT / "results")

RUN_FROM_RAW = os.environ.get("RUN_FROM_RAW", "0") == "1"


def results_dir(script_name):
    """Folder where one script writes its figures and tables."""
    path = RESULTS_DIR / script_name
    path.mkdir(parents=True, exist_ok=True)
    return path
