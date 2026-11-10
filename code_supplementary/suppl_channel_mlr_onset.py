# %%
"""
Channel MLR on the stimulus-onset window that the main analysis drops.

The main pipeline (`generate_covar_stats.py` → `multiple_linear_regression.py`)
keeps viewing bins with time_from_stimulus >= 0.1 s. This notebook repeats that
channel MLR (speed, pupil, source rbar, source PR → target rbar / PR; active
and passive fit separately) on the complementary onset window
0 <= time_from_stimulus < 0.1 s (0–100 ms).

Same cohort, areas, 200-bin subsample_id chunks, session-level z-scoring, VIF
flag, coefficient boxplots, R^2 boxes, and source→target heatmaps.

Cache: data/suppl_stats_onset_window.pk
Figures / CSVs: results/suppl_channel_mlr_onset/suppl_onset_channel_mlr_*
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from sklearn.linear_model import LinearRegression

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

import config

ALLEN_DATA_DIR = config.ALLEN_PREPROCESSED_DIR
DATA_DIR = config.DATA_DIR
OUTPUT_DIR = config.results_dir("suppl_channel_mlr_onset")
STATS_FILE = DATA_DIR / "suppl_stats_onset_window.pk"

SAVE_PLOTS = True
SHOW_PLOTS = True
SHOW_STRIP = False
FORCE_REBUILD = False
N_WORKERS = 4

TIME_FROM_STIMULUS_MIN = 0.0
TIME_FROM_STIMULUS_MAX = 0.1
GROUP_SIZE = 200
FILE_PREFIX = "suppl_onset_channel_mlr_active_passive"

if SAVE_PLOTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only the grouped areas used by the channel MLR.
AREAS_OF_INTEREST = [
    ("VISpl", ["VISp", "VISrl", "VISl", "VISal"]),
    ("VISm", ["VISpm", "VISam"]),
    ("LGd", ["LGd"]),
    ("SCMRN", ["MRN", "SCig", "SCiw"]),
]

area_order = ["LGd", "VISpl", "VISm", "SCMRN"]
active_order = [True, False]
ACTIVE_PALETTE = {True: "#2A7FFF", False: "#AA0000"}
channel_order = ["LGd_to_VISpl", "VISpl_to_VISm", "VISm_to_SCMRN"]
channel_display_labels = {
    "LGd_to_VISpl": "LGd -> VISpl",
    "VISpl_to_VISm": "VISpl -> VISm",
    "VISm_to_SCMRN": "VISm -> SCMRN",
}

CHANNEL_MIN_ROWS_PER_SESSION = 20
CHANNEL_VIF_FLAG_THRESHOLD = 5.0

MODEL_TERMS = ["speed_z", "pupil_z", "input_rbar_z", "input_PR_z"]
PLOT_TERM_ORDER = ["input_rbar_z", "input_PR_z", "speed_z", "pupil_z"]
TERM_LABELS = {
    "input_rbar_z": "source activity",
    "input_PR_z": "source dimensionality",
    "speed_z": "speed",
    "pupil_z": "pupil",
}
channel_outcomes = {
    "rbar_target": "target_rbar",
    "PR_target": "target_PR",
}

CROSS_PRED_ORDER = ["input_rbar_z", "input_PR_z"]
CROSS_OUTCOME_ORDER = ["rbar_target", "PR_target"]
CROSS_PRED_LABELS = {
    "input_rbar_z": "source activity",
    "input_PR_z": "source PR",
}
CROSS_OUTCOME_LABELS = {
    "rbar_target": "target activity",
    "PR_target": "target PR",
}
CMAP_ACTIVE = LinearSegmentedColormap.from_list(
    "active_blue",
    ["#9FC4FF", "#FFFFFF", "#2A7FFF"],
)
CMAP_PASSIVE = LinearSegmentedColormap.from_list(
    "passive_red",
    ["#E08A8A", "#FFFFFF", "#AA0000"],
)
CROSS_CMAPS = {True: CMAP_ACTIVE, False: CMAP_PASSIVE}
CROSS_MIN_SESSIONS = 10


# %%
# --------------------------------------------------------------------
# COVARIANCE STATS FOR THE ONSET WINDOW (same recipe as generate_covar_stats)
# --------------------------------------------------------------------

def statsvalues2PR(mii, mij, sii, sij, N=10**4):
    return 1 / (
        1 / N
        + 1 / N * (sii / mii) ** 2
        + (N - 1) / N * (sij / mii) ** 2
        + (N - 1) / N * (mij / mii) ** 2
    )


def cov_stats(spkcounts):
    C = np.cov(spkcounts)
    if np.sum(C) == 0 or np.sum(spkcounts) == 0:
        return {}
    ofdiag = np.triu(np.ones_like(C, dtype=bool), k=1)
    ondiag = np.eye(C.shape[0], dtype=bool)

    N = C.shape[0]
    mii = C[ondiag].mean()
    mij = C[ofdiag].mean()
    sii = np.sqrt(1 / (N - 1) * np.sum((C[ondiag] - mii) ** 2))
    sij = np.sqrt(1 / (N * (N - 1) - 2) * np.sum((C[ofdiag] - mij) ** 2))

    TV_emp = np.trace(C)
    PR_emp = (np.trace(C)) ** 2 / (np.trace(C @ C))
    diag_elements = C[ondiag]
    PR_fit = statsvalues2PR(mii, mij, sii, sij)

    return {
        "mii": mii,
        "mij": mij,
        "sii": sii,
        "sij": sij,
        "TV_emp": TV_emp,
        "PR_emp": PR_emp,
        "PR_fit": PR_fit,
        "variances": [diag_elements],
    }


def compute_subsampled_stats(
    spikecounts_all,
    group_metadata,
    cov_stats_func,
    area,
    group_size=GROUP_SIZE,
    speed_all=None,
    pupil_all=None,
):
    N_rows = spikecounts_all.shape[0]
    N_neurons = spikecounts_all.shape[1]

    is_change_flag = bool(group_metadata.get("is_change", False))
    if is_change_flag:
        if N_rows == 0:
            return []
        chunks = [np.arange(N_rows)]
    else:
        n_groups = N_rows // group_size
        if n_groups == 0:
            return []
        chunks = [
            np.arange(i * group_size, (i + 1) * group_size) for i in range(n_groups)
        ]

    stats_list = []
    for chunk_idx, idx_chunk in enumerate(chunks):
        submatrix = spikecounts_all[idx_chunk, :]
        avgspikecounts = np.mean(submatrix, axis=0)
        dict_stats = cov_stats_func(submatrix.T)
        speed_chunk = speed_all[idx_chunk] if speed_all is not None else None
        pupil_chunk = pupil_all[idx_chunk] if pupil_all is not None else None
        dict_stats.update(group_metadata)
        dict_stats.update(
            {
                "area": area,
                "N": N_neurons,
                "avgspikecounts": [avgspikecounts],
                "subsample_id": chunk_idx,
            }
        )
        if speed_chunk is not None:
            dict_stats["speed_values"] = [speed_chunk.astype(float)]
        if pupil_chunk is not None:
            dict_stats["pupil_values"] = [pupil_chunk.astype(float)]
        stats_list.append(dict_stats)
    return stats_list


def process_session_onset(session_id, session_number, experience_level, image_set):
    session_str = str(session_id)
    activity_file = ALLEN_DATA_DIR / "day1" / session_str / "df_activity.pk"
    units_file = ALLEN_DATA_DIR / "day1" / session_str / "df_units.pk"
    if not activity_file.exists() or not units_file.exists():
        print(f"Skipping session {session_str}: missing activity or units pickle")
        return pd.DataFrame()

    df_activity_ses = pd.read_pickle(activity_file)
    df_units_ses = pd.read_pickle(units_file)
    df_activity_ses["session_id"] = session_str
    df_units_ses["session_id"] = session_str
    print(f"Processing session: {int(session_str)}")

    df_activity_ses = df_activity_ses[df_activity_ses.is_viewing == True]
    df_activity_ses = df_activity_ses[
        (df_activity_ses.time_from_stimulus >= TIME_FROM_STIMULUS_MIN)
        & (df_activity_ses.time_from_stimulus < TIME_FROM_STIMULUS_MAX)
    ]
    df_activity_ses = df_activity_ses[df_activity_ses.is_change == False]
    df_activity_ses["image_name"] = pd.Categorical(
        df_activity_ses["image_name"], ordered=True
    )

    df_activity_ses["mean_speed"] = df_activity_ses["speed"].apply(
        lambda d: np.mean(list(d.values())) if isinstance(d, dict) and len(d) > 0 else np.nan
    )
    df_activity_ses["mean_pupil_area"] = df_activity_ses["pupil_area"].apply(
        lambda d: np.mean(list(d.values())) if isinstance(d, dict) and len(d) > 0 else np.nan
    )

    stats_list = []
    for area, structures in AREAS_OF_INTEREST:
        df_units_area_ses = df_units_ses[df_units_ses.structure_acronym.isin(structures)]
        if df_units_area_ses.shape[0] < 10:
            continue

        idx_units_area = df_units_area_ses.index.values
        df_activity_local = df_activity_ses.copy()
        df_activity_local["spikecounts_area"] = df_activity_local["spikecounts"].apply(
            lambda x: x[idx_units_area]
        )

        for group_name, df_activity_group in df_activity_local.groupby(
            ["session_id", "active", "image_name", "is_image_novel"],
            observed=True,
        ):
            if df_activity_group.empty:
                continue

            session_id_g, active, image_name, is_image_novel = group_name
            group_keys = {
                "session_id": session_id_g,
                "active": active,
                "image_name": image_name,
                "is_change": False,
                "is_image_novel": is_image_novel,
                "session_number": session_number,
                "experience_level": experience_level,
                "image_set": image_set,
            }
            spikecounts_all = np.stack(df_activity_group.spikecounts_area.values, axis=0)
            speed_all = df_activity_group["mean_speed"].to_numpy(dtype=float)
            pupil_all = df_activity_group["mean_pupil_area"].to_numpy(dtype=float)
            stats_list.extend(
                compute_subsampled_stats(
                    spikecounts_all,
                    group_keys,
                    cov_stats_func=cov_stats,
                    area=area,
                    speed_all=speed_all,
                    pupil_all=pupil_all,
                )
            )

    return pd.DataFrame(stats_list)


def build_or_load_onset_stats():
    if STATS_FILE.exists() and not FORCE_REBUILD:
        print(f"Loading cached onset stats: {STATS_FILE}")
        return pd.read_pickle(STATS_FILE)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_sessions = pd.read_pickle(ALLEN_DATA_DIR / "df_sessions.pk")
    df_sessions = df_sessions.query(
        'session_number==1 and experience_level=="Familiar"'
    )
    session_meta = df_sessions[
        ["ecephys_session_id", "session_number", "experience_level", "image_set"]
    ].values.tolist()
    print(
        f"Building onset-window cov stats "
        f"({TIME_FROM_STIMULUS_MIN:g} <= t < {TIME_FROM_STIMULUS_MAX:g} s) "
        f"for {len(session_meta)} sessions."
    )
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        results = pool.starmap(process_session_onset, session_meta)
    frames = [df for df in results if df is not None and not df.empty]
    if not frames:
        raise RuntimeError("No onset-window covariance rows were produced.")
    df_stats = pd.concat(frames, ignore_index=True)
    df_stats.to_pickle(STATS_FILE)
    print(f"Saved {STATS_FILE} ({len(df_stats)} rows)")
    return df_stats


# %%
# --------------------------------------------------------------------
# MLR HELPERS (same as multiple_linear_regression.py)
# --------------------------------------------------------------------

def unpack_list_of_one(values):
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], (list, np.ndarray)):
        arr = np.asarray(values[0], dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
    return arr.ravel()


def zscore(series):
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def compute_rbar(avgspikecounts, bin_width=0.05):
    counts = np.asarray(avgspikecounts, dtype=float).ravel()
    if counts.size == 0:
        return np.nan
    if isinstance(avgspikecounts, (list, tuple)) and len(avgspikecounts) == 1:
        counts = np.asarray(avgspikecounts[0], dtype=float).ravel()
    return float(counts.mean() / bin_width)


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


def _style_seaborn_box_artists(ax):
    for patch in ax.patches:
        if not hasattr(patch, "set_edgecolor"):
            continue
        facecolor = patch.get_facecolor()
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)
        patch.set_alpha(0.7)
        if len(facecolor) >= 3:
            patch.set_facecolor((*facecolor[:3], 0.7))
    for line in ax.lines:
        line.set_color("black")
        line.set_linewidth(1.5)
        marker = line.get_marker()
        if marker not in (None, "None", ""):
            line.set_marker("o")
            line.set_markerfacecolor("white")
            line.set_markeredgecolor("black")
            line.set_markersize(5)
            line.set_alpha(0.8)
    for collection in ax.collections:
        collection.set_facecolor("white")
        collection.set_edgecolor("black")
        collection.set_linewidth(0.8)
        collection.set_alpha(0.8)
        collection.set_sizes([25])


def compute_significance_stats(df_coef_sessions, value_col="coef"):
    results = []
    group_cols = ["channel", "outcome", "term"]
    for group_keys, group_data in df_coef_sessions.groupby(group_cols, dropna=False):
        channel, outcome, term = group_keys
        active_data = group_data[group_data["active"] == True][value_col].dropna()
        passive_data = group_data[group_data["active"] == False][value_col].dropna()

        p_active_vs_zero = np.nan
        p_passive_vs_zero = np.nan
        if len(active_data) > 1:
            _, p_active_vs_zero = stats.ttest_1samp(active_data, 0)
        if len(passive_data) > 1:
            _, p_passive_vs_zero = stats.ttest_1samp(passive_data, 0)

        p_active_vs_passive = np.nan
        if len(active_data) > 1 and len(passive_data) > 1:
            active_df = group_data[group_data["active"] == True][["session_id", value_col]].dropna()
            passive_df = group_data[group_data["active"] == False][["session_id", value_col]].dropna()
            paired = active_df.merge(passive_df, on="session_id", suffixes=("_active", "_passive"))
            if len(paired) > 1:
                _, p_active_vs_passive = stats.ttest_rel(
                    paired[f"{value_col}_active"].values,
                    paired[f"{value_col}_passive"].values,
                )

        results.append(
            {
                "channel": channel,
                "outcome": outcome,
                "term": term,
                "p_active_vs_zero": p_active_vs_zero,
                "p_passive_vs_zero": p_passive_vs_zero,
                "p_active_vs_passive": p_active_vs_passive,
            }
        )
    return pd.DataFrame(results)


def compute_sem(x):
    x_clean = pd.Series(x).dropna()
    if len(x_clean) > 1:
        return float(x_clean.sem())
    return np.nan


def plot_coefficients_by_active(
    df_coef_sessions,
    df_sig,
    channel_name,
    outcome_name,
    filename,
    strip_flag=True,
    y_lim=None,
):
    df_plot = df_coef_sessions[
        (df_coef_sessions["channel"] == channel_name)
        & (df_coef_sessions["outcome"] == outcome_name)
    ].copy()
    if df_plot.empty:
        return

    terms = [t for t in PLOT_TERM_ORDER if t in set(df_plot["term"])]
    if not terms:
        return

    df_plot["term"] = pd.Categorical(df_plot["term"], categories=terms, ordered=True)
    df_plot["term_label"] = df_plot["term"].map(TERM_LABELS)
    term_labels = [TERM_LABELS[t] for t in terms]
    df_plot["term_label"] = pd.Categorical(
        df_plot["term_label"], categories=term_labels, ordered=True
    )
    df_plot["active"] = pd.Categorical(df_plot["active"], categories=active_order, ordered=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.boxplot(
        data=df_plot,
        x="term_label",
        y="coef",
        hue="active",
        hue_order=active_order,
        order=term_labels,
        dodge=0.45,
        fill=True,
        showfliers=True,
        palette=ACTIVE_PALETTE,
        ax=ax,
    )
    _style_seaborn_box_artists(ax)

    if strip_flag:
        sns.stripplot(
            data=df_plot,
            x="term_label",
            y="coef",
            hue="active",
            hue_order=active_order,
            order=term_labels,
            dodge=True,
            jitter=0.25,
            alpha=0.5,
            size=4,
            linewidth=0,
            edgecolor="none",
            palette=ACTIVE_PALETTE,
            ax=ax,
            zorder=0,
            legend=False,
        )
        for collection in ax.collections:
            if hasattr(collection, "set_rasterized"):
                collection.set_rasterized(True)

    handles, labels = ax.get_legend_handles_labels()
    label_map = {"True": "active", "False": "passive", True: "active", False: "passive"}
    seen = set()
    clean_handles, clean_labels = [], []
    for handle, label in zip(handles, labels):
        mapped = label_map.get(label, label)
        if mapped in seen:
            continue
        seen.add(mapped)
        clean_handles.append(handle)
        clean_labels.append(mapped)
    ax.legend(clean_handles, clean_labels, title=None, frameon=False, loc="best")

    if y_lim is None:
        coef_min = float(np.nanmin(df_plot["coef"].to_numpy()))
        coef_max = float(np.nanmax(df_plot["coef"].to_numpy()))
        pad = 0.08 * (coef_max - coef_min if coef_max > coef_min else 1.0)
        y_lim = (coef_min - pad, coef_max + pad)

    y_offset = (y_lim[1] - y_lim[0]) * 0.03
    x_base = np.arange(len(terms))
    offsets = {True: -0.2, False: 0.2}

    df_sig_plot = df_sig[
        (df_sig["channel"] == channel_name) & (df_sig["outcome"] == outcome_name)
    ].copy()
    for _, sig_row in df_sig_plot.iterrows():
        term = sig_row["term"]
        if term not in terms:
            continue
        x_pos = x_base[terms.index(term)]
        pred_data = df_plot[df_plot["term"] == term]
        active_data = pred_data[pred_data["active"] == True]["coef"]
        passive_data = pred_data[pred_data["active"] == False]["coef"]
        active_y_base = active_data.max() if len(active_data) > 0 else y_lim[1] * 0.85
        passive_y_base = passive_data.max() if len(passive_data) > 0 else y_lim[1] * 0.85
        active_y = min(active_y_base + y_offset, y_lim[1] - y_offset)
        passive_y = min(passive_y_base + y_offset, y_lim[1] - y_offset)

        ast_active = get_asterisk(sig_row["p_active_vs_zero"])
        ast_passive = get_asterisk(sig_row["p_passive_vs_zero"])
        if ast_active:
            ax.text(
                x_pos + offsets[True], active_y, ast_active,
                ha="center", va="bottom", color=ACTIVE_PALETTE[True],
                fontsize=10, fontweight="bold", clip_on=True,
            )
        if ast_passive:
            ax.text(
                x_pos + offsets[False], passive_y, ast_passive,
                ha="center", va="bottom", color=ACTIVE_PALETTE[False],
                fontsize=10, fontweight="bold", clip_on=True,
            )
        ast_diff = get_asterisk(sig_row["p_active_vs_passive"])
        if ast_diff:
            mid_y = min(max(active_y, passive_y) + y_offset, y_lim[1] - y_offset)
            ax.text(
                x_pos, mid_y, ast_diff,
                ha="center", va="bottom", color="black",
                fontsize=10, fontweight="bold", clip_on=True,
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predictor")
    ax.set_ylabel(f"Coefficient (z-scored {outcome_name})")
    ax.set_title(
        f"{channel_display_labels[channel_name]} - {outcome_name} (onset 0–100 ms)"
    )
    ax.set_ylim(*y_lim)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(OUTPUT_DIR / filename, format="pdf", bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_cross_term_heatmap(df_coef_sessions, channel_name, min_sessions=10):
    eff = df_coef_sessions[
        (df_coef_sessions["channel"] == channel_name)
        & (df_coef_sessions["term"].isin(CROSS_PRED_ORDER))
        & (df_coef_sessions["outcome"].isin(CROSS_OUTCOME_ORDER))
    ].copy()
    if eff.empty:
        print(f"No cross-term data for {channel_name}.")
        return

    counts = (
        eff.groupby(["term", "outcome", "active"], as_index=False)["session_id"]
        .nunique()
        .rename(columns={"session_id": "n_sessions"})
    )
    eligible = counts[counts["n_sessions"] >= min_sessions][["term", "outcome", "active"]]
    eff = eff.merge(eligible, on=["term", "outcome", "active"], how="inner")
    if eff.empty:
        print(f"No cross-term groups with ≥{min_sessions} sessions for {channel_name}.")
        return

    df_avg = (
        eff.groupby(["term", "outcome", "active"], as_index=False)
        .agg(
            coef_mean=("coef", "mean"),
            coef_sem=("coef", compute_sem),
            n_sessions=("coef", "count"),
        )
    )

    predictors = CROSS_PRED_ORDER
    outcomes = CROSS_OUTCOME_ORDER
    pred_labels = [CROSS_PRED_LABELS[p] for p in predictors]
    outcome_labels = [CROSS_OUTCOME_LABELS[o] for o in outcomes]

    matrices, annot_matrices = {}, {}
    for active_val in active_order:
        df_subset = df_avg[df_avg["active"] == active_val]
        if df_subset.empty:
            continue
        matrix = np.full((len(predictors), len(outcomes)), np.nan)
        annot_matrix = []
        for i, pred in enumerate(predictors):
            annot_row = []
            for j, out in enumerate(outcomes):
                row = df_subset[(df_subset["term"] == pred) & (df_subset["outcome"] == out)]
                if not row.empty:
                    mean_val = float(row["coef_mean"].iloc[0])
                    sem_val = row["coef_sem"].iloc[0]
                    matrix[i, j] = mean_val
                    if pd.isna(sem_val):
                        annot_row.append(f"{mean_val:.3f}")
                    else:
                        annot_row.append(f"{mean_val:.3f}\n±{sem_val:.3f}")
                else:
                    annot_row.append("N/A")
            annot_matrix.append(annot_row)
        matrices[active_val] = matrix
        annot_matrices[active_val] = annot_matrix

    if not matrices:
        return

    all_values = []
    for matrix in matrices.values():
        all_values.extend(matrix[~np.isnan(matrix)].ravel().tolist())
    if not all_values:
        return
    vmax = float(np.nanmax(np.abs(all_values)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    for idx, active_val in enumerate([True, False]):
        ax = axes[idx]
        if active_val not in matrices:
            ax.axis("off")
            continue
        sns.heatmap(
            data=matrices[active_val],
            annot=np.array(annot_matrices[active_val]),
            fmt="",
            cmap=CROSS_CMAPS[active_val],
            vmin=vmin,
            vmax=vmax,
            center=0,
            xticklabels=outcome_labels,
            yticklabels=pred_labels,
            cbar=False,
            ax=ax,
            linewidths=1.5,
            linecolor="gray",
            square=False,
        )
        ax.set_title("Active" if active_val else "Passive", fontsize=12, fontweight="bold")
        ax.set_xlabel("Target metric", fontsize=11)
        ax.set_ylabel("Source predictor", fontsize=11)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="6%", pad=0.65)
        cbar = fig.colorbar(ax.collections[0], cax=cax, orientation="horizontal")
        cbar.set_label("Coefficient (mean ± SEM)", fontsize=10)

    fig.suptitle(
        f"{channel_display_labels[channel_name]}: source → target (onset 0–100 ms)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    filename = f"{FILE_PREFIX}_cross_term_{channel_name}.pdf"
    if SAVE_PLOTS:
        fig.savefig(OUTPUT_DIR / filename, format="pdf", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# %%
# --------------------------------------------------------------------
# BUILD / LOAD ONSET STATS AND CHANNEL TABLES
# --------------------------------------------------------------------

df_stats_all = build_or_load_onset_stats()

data_filter = (
    (df_stats_all.session_number == 1)
    & (df_stats_all.experience_level == "Familiar")
    & (df_stats_all.image_name != "omitted")
    & (df_stats_all.is_image_novel == False)
    & (df_stats_all.is_change == False)
    & (df_stats_all.area.isin(area_order))
    & (df_stats_all.N >= 30)
)
df_stats = df_stats_all[data_filter].drop(
    ["session_number", "experience_level", "is_image_novel", "is_change", "image_set"],
    axis=1,
).copy()
print(f"Onset-window rows after cohort filter: {len(df_stats)}")

if "rbar" in df_stats.columns:
    df_stats["rbar"] = pd.to_numeric(df_stats["rbar"], errors="coerce")
else:
    df_stats["rbar"] = df_stats["avgspikecounts"].apply(compute_rbar)

df_stats["speed_arr"] = df_stats["speed_values"].apply(unpack_list_of_one)
df_stats["pupil_arr"] = df_stats["pupil_values"].apply(unpack_list_of_one)
df_stats["speed"] = df_stats["speed_arr"].apply(np.nanmean)
df_stats["pupil"] = df_stats["pupil_arr"].apply(np.nanmean)
df_stats = df_stats.drop(
    [
        "speed_values",
        "pupil_values",
        "avgspikecounts",
        "speed_arr",
        "pupil_arr",
        "mii",
        "mij",
        "sii",
        "sij",
        "PR_fit",
        "variances",
    ],
    axis=1,
    errors="ignore",
)
if "subsample_id" not in df_stats.columns:
    df_stats["subsample_id"] = (
        df_stats.groupby(["session_id", "active", "image_name", "area"], dropna=False).cumcount()
    )

channels = [
    ("LGd", "VISpl"),
    ("VISpl", "VISm"),
    ("VISm", "SCMRN"),
]
channel_dfs = {}

for input_area, target_area in channels:
    df_channel = df_stats[df_stats["area"].isin([input_area, target_area])].copy()
    df_channel["area_role"] = np.where(df_channel["area"] == input_area, "input", "target")
    group_cols = ["session_id", "active", "image_name", "subsample_id"]
    role_size = df_channel.groupby(group_cols)["area_role"].transform("size")
    role_unique = df_channel.groupby(group_cols)["area_role"].transform("nunique")
    df_channel = df_channel[(role_size == 2) & (role_unique == 2)].copy()
    df_channel = (
        df_channel.pivot(
            index=group_cols,
            columns="area_role",
            values=["area", "rbar", "PR_emp", "speed", "pupil", "N"],
        )
        .reset_index()
    )
    df_channel.columns = [
        c if isinstance(c, str) else "_".join([str(x) for x in c if x != ""])
        for c in df_channel.columns
    ]
    df_channel = df_channel.rename(
        columns={
            "area_input": "input_area",
            "area_target": "target_area",
            "rbar_input": "input_rbar",
            "rbar_target": "target_rbar",
            "PR_emp_input": "input_PR",
            "PR_emp_target": "target_PR",
            "speed_input": "speed",
            "pupil_input": "pupil",
            "N_input": "N_input",
            "N_target": "N_target",
        }
    )
    df_channel = df_channel[
        [
            "session_id",
            "active",
            "image_name",
            "subsample_id",
            "input_area",
            "target_area",
            "input_rbar",
            "target_rbar",
            "input_PR",
            "target_PR",
            "speed",
            "pupil",
            "N_input",
            "N_target",
        ]
    ].copy()
    channel_dfs[f"{input_area}_to_{target_area}"] = df_channel
    print(f"{input_area}_to_{target_area}: {len(df_channel)} paired rows")


# %%
# --------------------------------------------------------------------
# CHANNEL MLR: ACTIVE VS PASSIVE
# --------------------------------------------------------------------

channel_coef_records = []
channel_r2_records = []
channel_vif_records = []
channel_session_counts = []

print("Running onset-window channel MLR (active and passive separately).")

for channel_name, df_channel in channel_dfs.items():
    for active_state in active_order:
        df_channel_active = df_channel[df_channel["active"] == active_state].copy()
        for outcome_name, outcome_col in channel_outcomes.items():
            skipped_sessions = 0
            fitted_sessions = []
            for session_id, df_sess in df_channel_active.groupby("session_id"):
                if len(df_sess) < CHANNEL_MIN_ROWS_PER_SESSION:
                    skipped_sessions += 1
                    continue

                df_sess = df_sess.copy()
                df_sess["speed_z"] = zscore(df_sess["speed"])
                df_sess["pupil_z"] = zscore(df_sess["pupil"])
                df_sess["input_rbar_z"] = zscore(df_sess["input_rbar"])
                df_sess["input_PR_z"] = zscore(df_sess["input_PR"])
                df_sess["target_z"] = zscore(df_sess[outcome_col])

                X = df_sess[MODEL_TERMS].astype(float)
                y = df_sess["target_z"].astype(float)

                vif_values = {}
                for term in X.columns:
                    y_vif = X[term]
                    X_other = X.drop(columns=[term])
                    if y_vif.std(ddof=0) == 0:
                        vif_values[term] = np.inf
                    else:
                        r2_vif = LinearRegression().fit(X_other, y_vif).score(X_other, y_vif)
                        vif_values[term] = np.inf if np.isclose(1.0 - r2_vif, 0.0) else 1.0 / (1.0 - r2_vif)

                max_vif = np.nanmax(list(vif_values.values())) if len(vif_values) > 0 else np.nan
                is_problematic = (np.isfinite(max_vif) and max_vif >= CHANNEL_VIF_FLAG_THRESHOLD) or (
                    not np.isfinite(max_vif)
                )
                for term, vif in vif_values.items():
                    channel_vif_records.append(
                        {
                            "channel": channel_name,
                            "outcome": outcome_name,
                            "active": active_state,
                            "session_id": session_id,
                            "n_rows": len(df_sess),
                            "term": term,
                            "vif": float(vif),
                            "max_vif": float(max_vif),
                            "is_problematic": bool(is_problematic),
                        }
                    )

                reg = LinearRegression().fit(X, y)
                coef_map = dict(zip(X.columns, reg.coef_))
                channel_r2_records.append(
                    {
                        "channel": channel_name,
                        "outcome": outcome_name,
                        "active": active_state,
                        "session_id": session_id,
                        "r2": reg.score(X, y),
                        "n_rows": len(df_sess),
                    }
                )
                for term, coef in coef_map.items():
                    channel_coef_records.append(
                        {
                            "channel": channel_name,
                            "outcome": outcome_name,
                            "active": active_state,
                            "session_id": session_id,
                            "term": term,
                            "coef": coef,
                        }
                    )
                fitted_sessions.append(session_id)

            channel_session_counts.append(
                {
                    "channel": channel_name,
                    "outcome": outcome_name,
                    "active": active_state,
                    "fitted_area_sessions": len(fitted_sessions),
                    "fitted_unique_sessions": len(set(fitted_sessions)),
                    "skipped_sessions": skipped_sessions,
                }
            )

df_channel_coef_sessions = pd.DataFrame(channel_coef_records)
df_channel_r2_sessions = pd.DataFrame(channel_r2_records)
df_channel_vif_sessions = pd.DataFrame(channel_vif_records)
df_channel_session_counts = pd.DataFrame(channel_session_counts)

if not df_channel_coef_sessions.empty:
    df_channel_coef = (
        df_channel_coef_sessions.groupby(["channel", "outcome", "active", "term"], as_index=False)
        .agg(
            coef_mean=("coef", "mean"),
            coef_std=("coef", "std"),
            n_sessions=("session_id", "nunique"),
        )
    )
    df_channel_coef["coef_sem"] = df_channel_coef["coef_std"] / np.sqrt(
        df_channel_coef["n_sessions"].clip(lower=1)
    )
else:
    df_channel_coef = pd.DataFrame(
        columns=["channel", "outcome", "active", "term", "coef_mean", "coef_std", "coef_sem", "n_sessions"]
    )

if not df_channel_r2_sessions.empty:
    df_channel_r2 = (
        df_channel_r2_sessions.groupby(["channel", "outcome", "active"], as_index=False)
        .agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"), n_sessions=("session_id", "nunique"))
    )
    df_channel_r2["r2_sem"] = df_channel_r2["r2_std"] / np.sqrt(df_channel_r2["n_sessions"].clip(lower=1))
else:
    df_channel_r2 = pd.DataFrame(
        columns=["channel", "outcome", "active", "r2_mean", "r2_std", "r2_sem", "n_sessions"]
    )

if not df_channel_vif_sessions.empty:
    df_channel_problematic_sessions = (
        df_channel_vif_sessions[df_channel_vif_sessions["is_problematic"]]
        .groupby(["channel", "outcome", "active", "session_id"], as_index=False)
        .agg(n_rows=("n_rows", "first"), max_vif=("max_vif", "first"))
    )
else:
    df_channel_problematic_sessions = pd.DataFrame(
        columns=["channel", "outcome", "active", "session_id", "n_rows", "max_vif"]
    )

for channel_name in channel_dfs.keys():
    for outcome_name in channel_outcomes.keys():
        for active_state in active_order:
            df_fit = df_channel_r2_sessions[
                (df_channel_r2_sessions["channel"] == channel_name)
                & (df_channel_r2_sessions["outcome"] == outcome_name)
                & (df_channel_r2_sessions["active"] == active_state)
            ].copy()
            df_flag = df_channel_problematic_sessions[
                (df_channel_problematic_sessions["channel"] == channel_name)
                & (df_channel_problematic_sessions["outcome"] == outcome_name)
                & (df_channel_problematic_sessions["active"] == active_state)
            ].copy()
            initial_unique_session_ids = sorted(df_fit["session_id"].unique().tolist())
            flagged_unique_session_ids = sorted(df_flag["session_id"].unique().tolist())
            remaining_unique_session_ids = sorted(
                set(initial_unique_session_ids) - set(flagged_unique_session_ids)
            )
            active_label = "active" if active_state else "passive"
            print(
                f"[{channel_name} | {outcome_name} | {active_label}] Initial fitted sessions: "
                f"{len(df_fit)} ({len(initial_unique_session_ids)} unique)."
            )
            if not df_flag.empty:
                print(
                    f"[{channel_name} | {outcome_name} | {active_label}] Flagged "
                    f"{len(df_flag)} fits with max VIF >= {CHANNEL_VIF_FLAG_THRESHOLD}; "
                    f"remaining {len(remaining_unique_session_ids)} unique sessions."
                )
            else:
                print(f"[{channel_name} | {outcome_name} | {active_label}] Flagged sessions: none.")


# %%
# --------------------------------------------------------------------
# SIGNIFICANCE, BOXPLOTS, R^2, HEATMAPS
# --------------------------------------------------------------------

if not df_channel_coef_sessions.empty:
    df_channel_sig = compute_significance_stats(df_channel_coef_sessions, value_col="coef")
else:
    df_channel_sig = pd.DataFrame(
        columns=["channel", "outcome", "term", "p_active_vs_zero", "p_passive_vs_zero", "p_active_vs_passive"]
    )

if not df_channel_coef_sessions.empty:
    coef_vals = df_channel_coef_sessions["coef"].to_numpy(dtype=float)
    coef_min = float(np.nanmin(coef_vals))
    coef_max = float(np.nanmax(coef_vals))
    coef_pad = 0.08 * (coef_max - coef_min if coef_max > coef_min else 1.0)
    shared_coef_ylim = (coef_min - coef_pad, coef_max + coef_pad)
else:
    shared_coef_ylim = (-1.0, 1.0)

for channel_name in channel_order:
    for outcome_name in channel_outcomes.keys():
        plot_coefficients_by_active(
            df_channel_coef_sessions,
            df_channel_sig,
            channel_name,
            outcome_name,
            filename=f"{FILE_PREFIX}_coefficients_{channel_name}_{outcome_name}.pdf",
            strip_flag=SHOW_STRIP,
            y_lim=shared_coef_ylim,
        )

if not df_channel_r2_sessions.empty:
    df_channel_r2_plot = df_channel_r2_sessions.copy()
    df_channel_r2_plot["channel"] = pd.Categorical(
        df_channel_r2_plot["channel"], categories=channel_order, ordered=True
    )
    df_channel_r2_plot["channel_label"] = df_channel_r2_plot["channel"].map(channel_display_labels)
    g_channel_r2 = sns.catplot(
        data=df_channel_r2_plot,
        x="channel_label",
        y="r2",
        col="outcome",
        hue="active",
        kind="box",
        order=[channel_display_labels[ch] for ch in channel_order],
        hue_order=active_order,
        palette=ACTIVE_PALETTE,
        sharey=True,
        height=4.0,
        aspect=1.2,
    )
    g_channel_r2.set_axis_labels("Channel", "R^2")
    g_channel_r2.fig.suptitle(
        "Channel model R^2 by outcome and block (onset 0–100 ms)", y=1.03
    )
    for ax in g_channel_r2.axes.flat:
        _style_seaborn_box_artists(ax)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=20)
    if g_channel_r2._legend is not None:
        for text in g_channel_r2._legend.texts:
            if text.get_text() == "True":
                text.set_text("active")
            elif text.get_text() == "False":
                text.set_text("passive")
    if SAVE_PLOTS:
        g_channel_r2.figure.savefig(
            OUTPUT_DIR / f"{FILE_PREFIX}_r2_by_channel.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(g_channel_r2.figure)

if SAVE_PLOTS and not df_channel_coef.empty:
    df_channel_coef.to_csv(OUTPUT_DIR / f"{FILE_PREFIX}_coefficients.csv", index=False)
if SAVE_PLOTS and not df_channel_sig.empty:
    df_channel_sig.to_csv(OUTPUT_DIR / f"{FILE_PREFIX}_significance.csv", index=False)
if SAVE_PLOTS and not df_channel_r2.empty:
    df_channel_r2.to_csv(OUTPUT_DIR / f"{FILE_PREFIX}_r2.csv", index=False)
if SAVE_PLOTS and not df_channel_vif_sessions.empty:
    df_channel_vif_sessions.to_csv(OUTPUT_DIR / f"{FILE_PREFIX}_vif_by_session.csv", index=False)
if SAVE_PLOTS and not df_channel_problematic_sessions.empty:
    df_channel_problematic_sessions.to_csv(
        OUTPUT_DIR / f"{FILE_PREFIX}_problematic_sessions.csv", index=False
    )

print("Creating onset-window cross-term heatmaps.")
for channel_name in channel_order:
    print(f"=== {channel_display_labels[channel_name]} ===")
    plot_cross_term_heatmap(
        df_channel_coef_sessions,
        channel_name,
        min_sessions=CROSS_MIN_SESSIONS,
    )
# %%
