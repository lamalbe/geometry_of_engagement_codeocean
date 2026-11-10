# %%
"""
Time-resolved participation ratio within active and passive blocks.

Aim
---
The main figures compare PR in the full active (task) block vs the full
passive (replay) block. In Allen Visual Behavior Neuropixels those blocks
are sequential (stimulus_block 0 then 5), so a whole-block difference could
in principle reflect slow nonstationarity — adaptation, satiety, electrode
drift — rather than engagement. This script asks whether the
engagement-dependent PR compression is stable in time once each block has
started, i.e. whether dimensionality is robust to temporal nonstationarity.

Design
------
For each day-1 Familiar session (image sets G and H) and each grouped area
(LGd, VISpl, VISm, SCMRN):

1. Load viewing bins with the same cohort filters as the other analysis
   notebooks: time_from_stimulus >= 0.1 s, non-change, non-novel,
   non-omitted, N >= 30. Activity is read from session df_activity.pk /
   df_units.pk (not from df_stats_area.pk: those subsample_id chunks are
   200 bins per image, not clock-time epochs).

2. Treat active and passive separately. Wall-clock time is bin_start_times.
   The block origin t0 is the first kept bin in that condition.

3. Discard the first 5 minutes of *each* block (not of the whole session).
   Dropping only session start would trim active and leave passive intact.

4. Split the remainder into five non-overlapping 11-minute clock epochs
   (~1 h block after the 5 min discard). The last epoch absorbs leftover
   time (11 min ± a bit); it is kept. PR is computed per image inside an
   epoch — the unit is (session_id, image_name), as in vis_circuit_metrics.py,
   with epoch in place of subsample_id.

5. Empirical PR = trace(C)^2 / trace(C @ C), then PR / N so sessions with
   different population sizes are comparable. No N-extrapolation here.

What to look for
----------------
- PR/N vs minutes into the block: two flat, separated traces (active vs
  passive) argue for a condition offset, not a slow slope.
- Active/passive ratio of window-averaged PR/N, one point per
  (session_id, image_name), still away from 1 after discarding the first 5 min.

A jump at the block switch with flat traces on either side is evidence
against slow drift. A slope that continues through the switch would be
the worrying pattern.

Outputs
-------
Optional cache: data/suppl_stats_PR_timewindows.pk
Figures: results/suppl_PR_timewindows/
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

import config

ALLEN_DATA_DIR = config.ALLEN_PREPROCESSED_DIR
DATA_DIR = config.DATA_DIR
RESULTS_DIR = config.results_dir("suppl_PR_timewindows")
STATS_FILE = DATA_DIR / "suppl_stats_PR_timewindows.pk"

SAVE_PLOTS = True
SHOW_PLOTS = True
FORCE_REBUILD = False
MIN_UNITS = 30
MIN_BINS = 200
TIME_FROM_STIMULUS_MIN = 0.1
DISCARD_SEC = 5 * 60
WINDOW_SEC = 11 * 60
N_EPOCHS = 5  # (60 min - 5 min discard) / 11 min
N_WORKERS = 4

AREAS = [
    ("LGd", ["LGd"]),
    ("VISpl", ["VISp", "VISrl", "VISl", "VISal"]),
    ("VISm", ["VISpm", "VISam"]),
    ("SCMRN", ["MRN", "SCig", "SCiw"]),
]
AREA_ORDER = ["LGd", "VISpl", "VISm", "SCMRN"]

PALETTE_MAP = {
    "LGd": {True: "#80B918", False: "#BFD200"},
    "VISpl": {True: "#34A0A4", False: "#48CAE4"},
    "VISm": {True: "#34A0A4", False: "#48CAE4"},
    "SCMRN": {True: "#C36F09", False: "#EEBA0B"},
}
AREA_FILL = {
    "LGd": "#80B918",
    "VISpl": "#34A0A4",
    "VISm": "#2A9D8F",
    "SCMRN": "#C36F09",
}

# %%
# -------------
#  PR HELPERS
# -------------


def empirical_pr(spikecounts):
    """PR from bins × neurons spike counts. None if covariance is degenerate."""
    if spikecounts.size == 0 or spikecounts.shape[0] < 2 or spikecounts.shape[1] < 2:
        return None
    if np.sum(spikecounts) == 0:
        return None
    C = np.cov(spikecounts.T)
    trC2 = np.trace(C @ C)
    if np.sum(C) == 0 or trC2 == 0:
        return None
    return float((np.trace(C) ** 2) / trC2)


def assign_epochs(rel_times):
    """Non-overlapping 11 min epochs after discarding the first 5 min of the block.

    Overflow past N_EPOCHS is folded into the last window so a slightly long
    block still has five epochs; a short tail is kept as the last window.
    """
    kept = rel_times >= DISCARD_SEC
    epoch = np.full(rel_times.shape, -1, dtype=int)
    epoch[kept] = np.floor((rel_times[kept] - DISCARD_SEC) / WINDOW_SEC).astype(int)
    overflow = epoch >= N_EPOCHS
    epoch[overflow] = N_EPOCHS - 1
    return epoch


def epoch_center_minutes(epoch_id):
    """Nominal minutes from block start to the center of a kept epoch."""
    start = DISCARD_SEC + epoch_id * WINDOW_SEC
    return (start + WINDOW_SEC / 2.0) / 60.0


# %%
# -------------
#  SESSION
# -------------


def process_session(session_id, session_number, experience_level, image_set):
    session_str = str(session_id)
    base_dir = ALLEN_DATA_DIR / "day1"
    activity_file = base_dir / session_str / "df_activity.pk"
    units_file = base_dir / session_str / "df_units.pk"
    if not activity_file.exists() or not units_file.exists():
        print(f"Missing files for session {session_str}")
        return pd.DataFrame()

    df_activity = pd.read_pickle(activity_file)
    df_units = pd.read_pickle(units_file)
    print(f"Processing session {session_str}")

    df_activity = df_activity[df_activity["is_viewing"] == True].copy()
    df_activity = df_activity[df_activity["time_from_stimulus"] >= TIME_FROM_STIMULUS_MIN]
    df_activity = df_activity[df_activity["is_change"] == False]
    df_activity = df_activity[df_activity["image_name"] != "omitted"]
    df_activity = df_activity[df_activity["is_image_novel"] == False]
    if df_activity.empty:
        return pd.DataFrame()

    times_all = df_activity["bin_start_times"].to_numpy(dtype=float)
    active_all = df_activity["active"].to_numpy()
    images_all = df_activity["image_name"].astype(str).to_numpy()
    counts_all = np.stack(
        [np.asarray(v, dtype=float) for v in df_activity["spikecounts"].to_numpy()],
        axis=0,
    )

    rows = []
    for area, acronyms in AREAS:
        df_units_area = df_units[df_units["structure_acronym"].isin(acronyms)]
        n_units = int(df_units_area.shape[0])
        if n_units < MIN_UNITS:
            continue
        idx = df_units_area.index.values
        counts = counts_all[:, idx]

        for active_val in (True, False):
            mask = active_all == active_val
            if mask.sum() < MIN_BINS:
                continue
            t = times_all[mask]
            X = counts[mask]
            imgs = images_all[mask]
            t0 = float(np.min(t))
            rel = t - t0
            epoch = assign_epochs(rel)
            for epoch_id in np.unique(epoch[epoch >= 0]):
                in_ep = epoch == epoch_id
                for image_name in np.unique(imgs[in_ep]):
                    in_img = in_ep & (imgs == image_name)
                    Xe = X[in_img]
                    if Xe.shape[0] < MIN_BINS:
                        continue
                    pr = empirical_pr(Xe)
                    if pr is None:
                        continue
                    rows.append(
                        {
                            "session_id": session_str,
                            "session_number": session_number,
                            "experience_level": experience_level,
                            "image_set": image_set,
                            "image_name": str(image_name),
                            "area": area,
                            "active": active_val,
                            "N": n_units,
                            "epoch_id": int(epoch_id),
                            "t_center_min": epoch_center_minutes(int(epoch_id)),
                            "n_bins": int(Xe.shape[0]),
                            "PR_emp": pr,
                            "PR_norm": pr / n_units,
                        }
                    )
    return pd.DataFrame(rows)


# %%
# -------------
#  BOXPLOT STYLE (line graphics from suppl_analysis_behavior.py)
# -------------


def get_asterisk(p_value):
    if np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _y_offset_for_annotations(y_series, frac=0.03):
    y = y_series.dropna()
    if y.empty:
        return 0.1
    span = y.max() - y.min()
    return frac * span if span > 0 else 0.1


def _annotate_one_sample_vs_mu(ax, data, y_col, categories, mu=1.0, category_col="area"):
    y_offset = _y_offset_for_annotations(data[y_col])
    for i, category in enumerate(categories):
        vals = data.loc[data[category_col] == category, y_col].dropna()
        p = np.nan
        if len(vals) > 1:
            _, p = stats.ttest_1samp(vals, mu)
        ast = get_asterisk(p)
        if not ast:
            continue
        box_y = vals.quantile(0.75) + y_offset
        ax.text(
            i, box_y, ast,
            ha="center", va="bottom", color="black",
            fontsize=10, fontweight="bold", clip_on=True,
        )


def _pad_ylim_for_asterisks(ax):
    ymin, ymax = ax.get_ylim()
    pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.1
    ax.set_ylim(ymin, ymax + pad)


def _style_filled_boxes(bp, box_colors):
    for i, (patch, color) in enumerate(zip(bp["boxes"], box_colors)):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)
        patch.set_alpha(0.7)
    for i in range(len(box_colors)):
        bp["whiskers"][i * 2].set_color("black")
        bp["whiskers"][i * 2 + 1].set_color("black")
        bp["whiskers"][i * 2].set_linewidth(1.5)
        bp["whiskers"][i * 2 + 1].set_linewidth(1.5)
        bp["caps"][i * 2].set_color("black")
        bp["caps"][i * 2 + 1].set_color("black")
        bp["caps"][i * 2].set_linewidth(1.5)
        bp["caps"][i * 2 + 1].set_linewidth(1.5)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    for flier in bp.get("fliers", []):
        flier.set_markerfacecolor("white")
        flier.set_markeredgecolor("black")
        flier.set_markersize(5)
        flier.set_alpha(0.8)
        flier.set_marker("o")


def _print_ratio_ttests(df_metric, tick_labels, label):
    print(f"\nOne-sample t-test vs 1 ({label}):")
    for area in tick_labels:
        vals = df_metric.loc[df_metric["area"] == area, "ratio"].dropna()
        if len(vals) < 2:
            print(f"  {area}: n={len(vals)} (skip)")
            continue
        t_stat, p = stats.ttest_1samp(vals, 1.0)
        print(
            f"  {area}: n={len(vals)} mean={vals.mean():.3f} "
            f"t={t_stat:.2f} p={p:.3g}"
        )


def _save_fig(fig, filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PLOTS:
        fig.savefig(
            RESULTS_DIR / filename, format="pdf", bbox_inches="tight", dpi=300,
        )
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()


# %%
# -------------
#  BUILD / LOAD TABLE
# -------------


def build_timewindow_stats():
    df_sessions = pd.read_pickle(ALLEN_DATA_DIR / "df_sessions.pk")
    df_sessions = df_sessions.query(
        'session_number == 1 and experience_level == "Familiar"'
    ).copy()
    session_meta = df_sessions[
        ["ecephys_session_id", "session_number", "experience_level", "image_set"]
    ].values.tolist()

    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        parts = pool.starmap(process_session, session_meta)
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _pickle_compat(df):
    """Cast pandas StringDtype (pandas 3 default) to object so older pandas can load."""
    df = df.copy()
    for col in df.columns:
        dtype_name = str(df[col].dtype)
        if dtype_name in ("str", "string") or dtype_name.startswith("string"):
            df[col] = df[col].astype(object)
    df.columns = pd.Index([str(c) for c in df.columns], dtype=object)
    return df


def load_or_build_stats():
    if STATS_FILE.exists() and not FORCE_REBUILD:
        try:
            df = pd.read_pickle(STATS_FILE)
            if "image_name" in df.columns:
                print(f"Loaded {len(df)} rows from {STATS_FILE}")
                return df
            print(f"{STATS_FILE} lacks image_name; rebuilding.")
        except (NotImplementedError, ValueError, ModuleNotFoundError) as exc:
            print(f"Could not load {STATS_FILE} ({exc}); rebuilding.")
    df = _pickle_compat(build_timewindow_stats())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(STATS_FILE, protocol=4)
    print(f"Saved {len(df)} rows to {STATS_FILE}")
    return df


def plot_pr_vs_time(df_win):
    hue_order = [True, False]
    x_order = sorted(df_win["t_center_min"].dropna().unique().tolist())
    n_a = len(AREA_ORDER)
    fig, axes = plt.subplots(
        1, n_a, figsize=(4.0 * n_a, 4.2), sharex=True, sharey=True, squeeze=False
    )
    for idx, (ax, area) in enumerate(zip(axes.flat, AREA_ORDER)):
        d = df_win[df_win["area"] == area]
        pal = PALETTE_MAP[area]
        sns.pointplot(
            data=d,
            x="t_center_min",
            y="PR_norm",
            hue="active",
            hue_order=hue_order,
            order=x_order,
            palette=[pal[h] for h in hue_order],
            errorbar="se",
            dodge=0.25,
            markersize=4.5,
            capsize=0.12,
            err_kws={"linewidth": 1.8},
            ax=ax,
            legend=(idx == 0),
        )
        ax.set_title(area, fontsize=12)
        ax.set_xlabel("Time into block (min)")
        ax.set_ylabel("PR / N" if idx == 0 else "")
        ax.set_xticks(range(len(x_order)))
        ax.set_xticklabels([f"{t:g}" for t in x_order])
        if idx == 0 and ax.get_legend() is not None:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title="active")
    fig.suptitle(
        "PR/N in 11 min epochs after discarding the first 5 min of each block "
        "(mean ± SE across session × image)",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout()
    _save_fig(fig, "suppl_PR_timewindows_vs_time.pdf")


def plot_ratio_box(df_win):
    df_pair_mean = (
        df_win.groupby(
            ["session_id", "image_name", "area", "active"], as_index=False
        )["PR_norm"].mean()
    )
    wide = df_pair_mean.pivot_table(
        index=["session_id", "image_name", "area"],
        columns="active",
        values="PR_norm",
        aggfunc="mean",
    ).reset_index()
    wide = wide.dropna(subset=[True, False])
    wide["ratio"] = wide[True] / wide[False]
    _plot_filled_ratio_box(
        wide,
        ylabel="Active / passive PR/N",
        title="Window-averaged PR/N ratio (first 5 min of each block dropped)",
        filename="suppl_PR_timewindows_ratio_box.pdf",
        ttest_label="window-averaged PR/N, active/passive (session × image)",
    )


def _plot_filled_ratio_box(wide, ylabel, title, filename, ttest_label):
    box_data, box_colors, tick_labels = [], [], []
    for area in AREA_ORDER:
        vals = wide.loc[wide["area"] == area, "ratio"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        box_data.append(vals)
        box_colors.append(AREA_FILL[area])
        tick_labels.append(area)
    if not box_data:
        return

    y_min, y_max = wide["ratio"].min(), wide["ratio"].max()
    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bp = ax.boxplot(
        box_data,
        positions=range(len(box_data)),
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        zorder=2,
    )
    _style_filled_boxes(bp, box_colors)
    ax.set_xticks(range(len(box_data)))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Area")
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlim(-0.5, len(box_data) - 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8, alpha=0.5, zorder=0)
    _annotate_one_sample_vs_mu(
        ax, wide, "ratio", tick_labels, mu=1.0, category_col="area",
    )
    _pad_ylim_for_asterisks(ax)
    ax.set_title(title)
    _print_ratio_ttests(wide, tick_labels, ttest_label)
    fig.tight_layout()
    _save_fig(fig, filename)


if __name__ == "__main__":
    df_win = load_or_build_stats()
    if df_win.empty:
        raise SystemExit("No time-window PR rows; nothing to plot.")
    plot_pr_vs_time(df_win)
    plot_ratio_box(df_win)
# %%
