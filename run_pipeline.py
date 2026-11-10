#!/usr/bin/env python
"""Run preprocessing, table generation, and figure scripts in order.

The Allen Visual Behavior Neuropixels dataset is not distributed with this
project because of its size. Download it from

    https://portal.brain-map.org/circuits-behavior/visual-behavior-neuropixels

and set ALLEN_RAW_DATA_DIR to reproduce the full pipeline. With the default
RUN_FROM_RAW=0, preprocessing and per-session generation are skipped and each
analysis runs only when its derived tables are already in DATA_DIR.

Missing inputs are reported and skipped. A script that starts with its inputs
present and then exits non-zero stops the pipeline.

    python run_pipeline.py            # reviewer / Code Ocean mode
    RUN_FROM_RAW=1 python run_pipeline.py
    python run_pipeline.py --dry-run  # print RUN/SKIP only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent


def _existing(paths):
    missing = []
    for path in paths:
        if not Path(path).exists():
            missing.append(Path(path))
    return missing


class Runner:
    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.ran = []
        self.skipped = []
        self.failed = []

    def step(self, relative, required=()):
        missing = _existing(required)
        if missing:
            print(f"SKIP {relative}")
            for path in missing:
                print(f"  missing {path}")
            self.skipped.append(relative)
            return False

        print(f"RUN  {relative}")
        if self.dry_run:
            self.ran.append(relative)
            return True

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["RUN_FROM_RAW"] = "1" if config.RUN_FROM_RAW else "0"
        env["ALLEN_RAW_DATA_DIR"] = str(config.ALLEN_RAW_DATA_DIR)
        env["ALLEN_PREPROCESSED_DIR"] = str(config.ALLEN_PREPROCESSED_DIR)
        env["ALLEN_CACHE_PARENT"] = str(config.ALLEN_CACHE_PARENT)
        env["DATA_DIR"] = str(config.DATA_DIR)
        env["RESULTS_DIR"] = str(config.RESULTS_DIR)

        completed = subprocess.run(
            [sys.executable, str(ROOT / relative)],
            cwd=str(ROOT),
            env=env,
        )
        if completed.returncode != 0:
            print(f"FAIL {relative} (exit {completed.returncode})")
            self.failed.append(relative)
            return False
        print(f"OK   {relative}")
        self.ran.append(relative)
        return True

    def abort_if_failed(self):
        if not self.failed:
            return False
        print()
        print(
            f"Ran {len(self.ran)}, skipped {len(self.skipped)}, "
            f"failed {len(self.failed)}."
        )
        return True

    def step_if_any(self, relative, options):
        """Run when at least one path in options exists."""
        if any(Path(path).exists() for path in options):
            return self.step(relative)
        print(f"SKIP {relative}")
        for path in options:
            print(f"  missing {path}")
        self.skipped.append(relative)
        return False


def _data(name):
    return config.DATA_DIR / name


def _sessions():
    return config.ALLEN_PREPROCESSED_DIR / "df_sessions.pk"


def _raw_manifests():
    return config.ALLEN_RAW_DATA_DIR / "manifests"


GEOMETRY_OUTPUTS = [
    "df_basis_vectors.pk",
    "df_max_cosine_active_passive.pk",
    "df_max_cosine_within_condition.pk",
    "df_subspace_overlap_pc.pk",
    "df_subspace_overlap_svd.pk",
]


def run(dry_run):
    os.environ.setdefault("MPLBACKEND", "Agg")
    runner = Runner(dry_run)

    print("Allen Visual Behavior Neuropixels is not included (dataset size).")
    print("https://portal.brain-map.org/circuits-behavior/visual-behavior-neuropixels")
    print(f"RUN_FROM_RAW={config.RUN_FROM_RAW}")
    print(f"DATA_DIR={config.DATA_DIR}")
    print(f"RESULTS_DIR={config.RESULTS_DIR}")
    if dry_run:
        print("Dry run: decisions use files present now, and nothing is executed.")
    print()

    activity_ok = False
    if config.RUN_FROM_RAW:
        runner.step(
            "code_gendata/preprocess_data.py",
            required=[_raw_manifests()],
        )
        if runner.abort_if_failed():
            return 1
        for relative in (
            "code_gendata/generate_VIS2MID_activity.py",
            "code_gendata/generate_decoding.py",
            "code_gendata/generate_poisson_stats.py",
            "code_gendata/generate_covar_stats.py",
            "code_gendata/VISp_cloud_shape.py",
        ):
            ok = runner.step(relative, required=[_sessions()])
            if runner.abort_if_failed():
                return 1
            if relative.endswith("generate_VIS2MID_activity.py"):
                activity_ok = ok
    else:
        print("SKIP preprocessing and per-session generation (RUN_FROM_RAW=0).")
        print("  Set RUN_FROM_RAW=1 when the raw Allen cache is available.")
        print()

    geometry_missing = any(not _data(name).exists() for name in GEOMETRY_OUTPUTS)
    if (config.RUN_FROM_RAW and activity_ok) or (
        not config.RUN_FROM_RAW and geometry_missing
    ):
        runner.step(
            "code_gendata/generate_VIS2MID_geometry.py",
            required=[_data("df_cortex2mid.pk")],
        )
        if runner.abort_if_failed():
            return 1
    else:
        print("SKIP code_gendata/generate_VIS2MID_geometry.py")
        if config.RUN_FROM_RAW:
            print("  activity table was not rebuilt")
        else:
            print("  geometry tables already present")
        runner.skipped.append("code_gendata/generate_VIS2MID_geometry.py")

    if config.RUN_FROM_RAW:
        behavior_required = [_sessions()]
    else:
        behavior_required = [_data("df_behavior.pk")]
    runner.step(
        "code_supplementary/suppl_behavior_session.py",
        required=behavior_required,
    )
    if runner.abort_if_failed():
        return 1
    if config.RUN_FROM_RAW:
        runner.step(
            "code_supplementary/generate_covstats_behavior.py",
            required=[_sessions(), _data("df_behavior.pk")],
        )
        if runner.abort_if_failed():
            return 1

    analyses = [
        (
            "code_analysis/poisson_metrics.py",
            [_data("df_poisson_stats.pk"), _data("df_poisson_stats_subsampled.pk")],
        ),
        ("code_analysis/vis_circuit_metrics.py", [_data("df_stats_area.pk")]),
        (
            "code_analysis/vis_primary_metrics.py",
            [_data("df_stats_area.pk"), _data("visp_cloud_shape.pk")],
        ),
        ("code_analysis/multiple_linear_regression.py", [_data("df_stats_area.pk")]),
        (
            "code_analysis/alignment_VIS2MID_channel.py",
            [_data(name) for name in GEOMETRY_OUTPUTS],
        ),
        (
            "code_supplementary/suppl_alignment_VISpl_SCMRN.py",
            [
                _data("df_basis_vectors.pk"),
                _data("df_max_cosine_active_passive.pk"),
                _data("df_subspace_overlap_pc.pk"),
                _data("df_subspace_overlap_svd.pk"),
            ],
        ),
        ("code_supplementary/suppl_PR_extrapolated.py", [_data("df_stats_area.pk")]),
        (
            "code_supplementary/suppl_analysis_behavior.py",
            [
                _data("suppl_stats_unengagement.pk"),
                _data("suppl_stats_speed_terciles.pk"),
                _data("suppl_stats_pupil_terciles.pk"),
                _data("df_stats_area.pk"),
            ],
        ),
        ("code_analysis/cross_area_VIS2MID_channel.py", [_data("df_cortex2mid.pk")]),
        (
            "code_supplementary/suppl_PR_timewindows.py",
            None,  # cache or per-session tables
        ),
        ("code_supplementary/suppl_channel_mlr_onset.py", None),
        ("code_theory/theory_plots.py", []),
        ("code_gendata/circuit_cloud.py", [_sessions()]),
        ("code_gendata/cortex_cloud.py", [_sessions()]),
        ("code_supplementary/unit_counts_by_area.py", [_sessions()]),
        ("code_supplementary/change_aligned_licks_rewards.py", [_sessions()]),
    ]

    for relative, required in analyses:
        if relative.endswith("suppl_PR_timewindows.py"):
            runner.step_if_any(
                relative,
                [_data("suppl_stats_PR_timewindows.pk"), _sessions()],
            )
        elif relative.endswith("suppl_channel_mlr_onset.py"):
            runner.step_if_any(
                relative,
                [_data("suppl_stats_onset_window.pk"), _sessions()],
            )
        else:
            runner.step(relative, required=required)
        if runner.abort_if_failed():
            return 1

    runner.step_if_any(
        "code_analysis/decoding_analysis.py",
        [_data("df_decoding_repeat.pk"), _data("df_decoding.pk")],
    )
    if runner.abort_if_failed():
        return 1
    runner.step_if_any(
        "code_analysis/image_decoding_analysis.py",
        [_data("df_image_decoding_repeat.pk"), _data("df_decoding.pk")],
    )
    if runner.abort_if_failed():
        return 1

    print()
    print(f"Ran {len(runner.ran)}, skipped {len(runner.skipped)}, failed {len(runner.failed)}.")
    if runner.failed:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print RUN/SKIP decisions and do not execute scripts.",
    )
    args = parser.parse_args()
    sys.exit(run(args.dry_run))


if __name__ == "__main__":
    main()
