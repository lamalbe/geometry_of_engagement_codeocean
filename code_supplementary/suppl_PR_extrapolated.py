# %%
"""
Extrapolated participation ratio vs hypothetical population size.

Aim
---
Empirical PR is computed from the recorded population size, so areas (and
sessions) cannot be compared at a common N. If the covariance is summarized by
four moments — mean and spread of on- and off-diagonal entries (mii, mij, sii,
sij) — the participation ratio has a closed form in N:

    PR(N) = N / (1 + (sii/mii)^2 + (N-1) * ((sij/mii)^2 + (mij/mii)^2))

This notebook plugs session–image mean moments into that formula and traces
PR(N) from 1 to 10^5. The curves ask whether engagement (active vs passive)
changes how dimensionality would scale if more neurons were recorded, and
whether that scaling differs across visual cortex, LGd, and midbrain.

Method
------
Uses data/df_stats_area.pk (from generate_covar_stats.py).
No new pickle is written.

Cohort (same as the other analysis notebooks):
  session_number == 1, experience_level == Familiar, image sets G and H,
  time_from_stimulus >= 0.1 (already applied at generation), viewing,
  non-change, non-novel, non-omitted, N >= 30.

For each (session_id, image_name, area, active), average mii, mij, sii, sij
over time-chunk subsample_id first, then evaluate PR(N). Mean ± SE of those
curves is taken across session–image pairs (not across sessions after
collapsing images). Moments are averaged before the nonlinear PR formula;
we do not average PR and then plot.

Figures: results/suppl_PR_extrapolated/.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

import config

DATA_DIR = config.DATA_DIR
RESULTS_DIR = config.results_dir("suppl_PR_extrapolated")
STATS_FILE = DATA_DIR / "df_stats_area.pk"

SAVE_PLOTS = True
MIN_UNITS = 30
N_LOG = 400
N_MAX = 100_000
MOMENT_COLS = ["mii", "mij", "sii", "sij"]

GROUPED_AREAS = ["LGd", "VISpl", "VISm", "SCMRN"]
VIS_AREAS = ["VISp", "VISpm", "VISam", "VISrl", "VISl", "VISal"]

PALETTE_MAP = {
    "LGd": {True: "#80B918", False: "#BFD200"},
    "VISpl": {True: "#34A0A4", False: "#48CAE4"},
    "VISm": {True: "#34A0A4", False: "#48CAE4"},
    "VISp": {True: "#34A0A4", False: "#48CAE4"},
    "VISpm": {True: "#34A0A4", False: "#48CAE4"},
    "VISam": {True: "#34A0A4", False: "#48CAE4"},
    "VISrl": {True: "#34A0A4", False: "#48CAE4"},
    "VISl": {True: "#34A0A4", False: "#48CAE4"},
    "VISal": {True: "#34A0A4", False: "#48CAE4"},
    "SCMRN": {True: "#C36F09", False: "#EEBA0B"},
}
DEFAULT_COLORS = {True: "#34A0A4", False: "#48CAE4"}

# %%
# -------------
#  LOAD DATA
# -------------
if not STATS_FILE.exists():
    raise FileNotFoundError(
        f"Missing {STATS_FILE}. Run code_gendata/generate_covar_stats.py "
        "to write df_stats_area.pk."
    )

df_stats_all = pd.read_pickle(STATS_FILE)
data_filter = (
    (df_stats_all.session_number == 1)
    & (df_stats_all.experience_level == "Familiar")
    & (df_stats_all.image_name != "omitted")
    & (df_stats_all.is_image_novel == False)
    & (df_stats_all.is_change == False)
    & (df_stats_all.N >= MIN_UNITS)
    & (df_stats_all.area.isin(GROUPED_AREAS + VIS_AREAS))
)
df_stats = df_stats_all.loc[data_filter].copy()

missing = [c for c in MOMENT_COLS if c not in df_stats.columns]
if missing:
    raise KeyError(f"df_stats_area.pk is missing moment columns: {missing}")

# %%
# --------------------------------
#  EXTRAPOLATED PARTICIPATION RATIO
# --------------------------------


def statsvalues2PR_matrix(mii, mij, sii, sij, N_grid):
    """PR(N) from covariance moments; one row per (mii, mij, sii, sij)."""
    mii = np.asarray(mii, dtype=float).reshape(-1, 1)
    mij = np.asarray(mij, dtype=float).reshape(-1, 1)
    sii = np.asarray(sii, dtype=float).reshape(-1, 1)
    sij = np.asarray(sij, dtype=float).reshape(-1, 1)
    N = np.asarray(N_grid, dtype=float).reshape(1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = (sii / mii) ** 2
        B = (sij / mii) ** 2 + (mij / mii) ** 2
        PR = N / (1 + A + (N - 1) * B)
    bad = (mii == 0) | ~np.isfinite(PR)
    return np.where(bad, np.nan, PR)


# Average moments over time chunks; keep session × image as the SE unit.
df_mom = (
    df_stats.groupby(
        ["session_id", "image_name", "active", "area"], as_index=False
    )[MOMENT_COLS]
    .mean()
)

N_grid = np.unique(
    np.concatenate([[1], np.round(np.logspace(0, np.log10(N_MAX), N_LOG)).astype(int)])
)
N_grid = np.sort(N_grid[(N_grid >= 1) & (N_grid <= N_MAX)])

PR_mat = statsvalues2PR_matrix(
    df_mom["mii"].to_numpy(),
    df_mom["mij"].to_numpy(),
    df_mom["sii"].to_numpy(),
    df_mom["sij"].to_numpy(),
    N_grid,
)
n_rows = len(df_mom)
df_pr_extrap = pd.DataFrame(
    {
        "session_id": np.repeat(df_mom["session_id"].to_numpy(), len(N_grid)),
        "image_name": np.repeat(df_mom["image_name"].to_numpy(), len(N_grid)),
        "active": np.repeat(df_mom["active"].to_numpy(), len(N_grid)),
        "area": np.repeat(df_mom["area"].to_numpy(), len(N_grid)),
        "N_neurons": np.tile(N_grid, n_rows),
        "PR_extrap": PR_mat.ravel(),
    }
)

# %%
# -------------
#  PLOTS
# -------------


def plot_pr_extrap(df_long, area_order, filename, title_suffix):
    areas = [a for a in area_order if a in set(df_long["area"].dropna())]
    if not areas:
        print(f"No data for {title_suffix}; skipping.")
        return

    hue_order = [True, False]
    n_a = len(areas)
    fig, axes = plt.subplots(
        1, n_a, figsize=(4.2 * n_a, 4.5), sharex=True, sharey=True, squeeze=False
    )
    for idx, (ax, area) in enumerate(zip(axes.flat, areas)):
        d_area = df_long[df_long["area"] == area]
        area_pal = PALETTE_MAP.get(area, DEFAULT_COLORS)
        pal_list = [area_pal[h] for h in hue_order]
        sns.lineplot(
            data=d_area,
            x="N_neurons",
            y="PR_extrap",
            hue="active",
            hue_order=hue_order,
            palette=pal_list,
            errorbar="se",
            ax=ax,
            legend=(idx == 0),
        )
        ax.set_title(area, fontsize=12)
        ax.set_xscale("log")
        ax.set_xlabel("Number of neurons (extrapolated)")
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylabel("PR (extrapolated)" if idx == 0 else "")
        if idx == 0 and ax.get_legend() is not None:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title="active")

    fig.suptitle(
        "Extrapolated participation ratio "
        f"({title_suffix}; mean moments per session×image; SE across pairs)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()

    if SAVE_PLOTS:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(RESULTS_DIR / filename, format="pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


plot_pr_extrap(
    df_pr_extrap,
    GROUPED_AREAS,
    "suppl_stats_PR_extrapolated_vs_N_grouped.pdf",
    "LGd, VISpl, VISm, SCMRN",
)
plot_pr_extrap(
    df_pr_extrap,
    VIS_AREAS,
    "suppl_stats_PR_extrapolated_vs_N_visual_cortex.pdf",
    "single VIS areas",
)
# %%
