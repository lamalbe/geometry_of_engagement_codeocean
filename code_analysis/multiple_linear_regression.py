# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from sklearn.linear_model import LinearRegression

"""
Channel MLR with behavior and source-area predictors, fit separately for active
and passive blocks.

Predictors: speed, pupil, source mean activity (rbar), source dimensionality (PR_emp)
Targets:    target mean activity (rbar), target dimensionality (PR_emp)
"""

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

import config

DATA_DIR = config.DATA_DIR
OUTPUT_DIR = config.results_dir("multiple_linear_regression")
SAVE_PLOTS = True
SHOW_PLOTS = True
SHOW_STRIP = False
if SAVE_PLOTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def unpack_list_of_one(values):
    """speed_values and pupil_values are stored as list([list_of_200])."""
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
    return float(counts.mean() / bin_width)


def get_asterisk(p_value):
    """Convert p-value to asterisk notation."""
    if np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def _style_seaborn_box_artists(ax):
    """Black 1.5 pt box/whisker/median lines; white fliers (visual-cortex by-image style)."""
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
    """
    Compute p-values for:
    1. One-sample t-test: each group (active/passive) vs 0
    2. Paired t-test: active vs passive (matched by session_id)
    """
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
    df_plot["term_label"] = pd.Categorical(df_plot["term_label"], categories=term_labels, ordered=True)
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

    # Keep a single legend with readable labels
    handles, labels = ax.get_legend_handles_labels()
    label_map = {"True": "active", "False": "passive", True: "active", False: "passive"}
    seen = set()
    clean_handles = []
    clean_labels = []
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

        active_y_base = (
            active_data.max() if len(active_data) > 0 else y_lim[1] * 0.85
        )
        passive_y_base = (
            passive_data.max() if len(passive_data) > 0 else y_lim[1] * 0.85
        )
        active_y = min(active_y_base + y_offset, y_lim[1] - y_offset)
        passive_y = min(passive_y_base + y_offset, y_lim[1] - y_offset)

        ast_active = get_asterisk(sig_row["p_active_vs_zero"])
        ast_passive = get_asterisk(sig_row["p_passive_vs_zero"])
        if ast_active:
            ax.text(
                x_pos + offsets[True],
                active_y,
                ast_active,
                ha="center",
                va="bottom",
                color=ACTIVE_PALETTE[True],
                fontsize=10,
                fontweight="bold",
                clip_on=True,
            )
        if ast_passive:
            ax.text(
                x_pos + offsets[False],
                passive_y,
                ast_passive,
                ha="center",
                va="bottom",
                color=ACTIVE_PALETTE[False],
                fontsize=10,
                fontweight="bold",
                clip_on=True,
            )

        ast_diff = get_asterisk(sig_row["p_active_vs_passive"])
        if ast_diff:
            mid_y = min(max(active_y, passive_y) + y_offset, y_lim[1] - y_offset)
            ax.text(
                x_pos,
                mid_y,
                ast_diff,
                ha="center",
                va="bottom",
                color="black",
                fontsize=10,
                fontweight="bold",
                clip_on=True,
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predictor")
    ax.set_ylabel(f"Coefficient (z-scored {outcome_name})")
    ax.set_title(f"{channel_display_labels[channel_name]} - {outcome_name}")
    ax.set_ylim(*y_lim)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    if SAVE_PLOTS:
        fig.savefig(OUTPUT_DIR / filename, format="pdf", bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# %%
# ---------------------------------------------------------
#  LOAD DATA AND BUILD CHANNEL TABLES
# ---------------------------------------------------------

stats_file = DATA_DIR / "df_stats_area.pk"
df_stats_all = pd.read_pickle(stats_file)

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

# %%
# ---------------------------------------------------------
#  CHANNEL MLR: ACTIVE VS PASSIVE
# ---------------------------------------------------------

channel_coef_records = []
channel_r2_records = []
channel_vif_records = []
channel_session_counts = []

print("Running channel MLR (active and passive separately).")

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
                r2_value = reg.score(X, y)

                channel_r2_records.append(
                    {
                        "channel": channel_name,
                        "outcome": outcome_name,
                        "active": active_state,
                        "session_id": session_id,
                        "r2": r2_value,
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
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            n_sessions=("session_id", "nunique"),
        )
    )
    df_channel_r2["r2_sem"] = df_channel_r2["r2_std"] / np.sqrt(df_channel_r2["n_sessions"].clip(lower=1))
else:
    df_channel_r2 = pd.DataFrame(columns=["channel", "outcome", "active", "r2_mean", "r2_std", "r2_sem", "n_sessions"])

if not df_channel_vif_sessions.empty:
    df_channel_problematic_sessions = (
        df_channel_vif_sessions[df_channel_vif_sessions["is_problematic"]]
        .groupby(["channel", "outcome", "active", "session_id"], as_index=False)
        .agg(
            n_rows=("n_rows", "first"),
            max_vif=("max_vif", "first"),
        )
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
            remaining_unique_session_ids = sorted(set(initial_unique_session_ids) - set(flagged_unique_session_ids))
            active_label = "active" if active_state else "passive"

            print(
                f"[{channel_name} | {outcome_name} | {active_label}] Initial fitted sessions available: "
                f"{len(df_fit)} area-session fits ({len(initial_unique_session_ids)} unique session IDs)."
            )
            if not df_flag.empty:
                print(
                    f"[{channel_name} | {outcome_name} | {active_label}] Flagged "
                    f"{len(df_flag)} area-session fits with max VIF >= {CHANNEL_VIF_FLAG_THRESHOLD}."
                )
                print(
                    f"[{channel_name} | {outcome_name} | {active_label}] Remaining sessions after flagging: "
                    f"{len(df_fit) - len(df_flag)} area-session fits "
                    f"({len(remaining_unique_session_ids)} unique session IDs)."
                )
            else:
                print(
                    f"[{channel_name} | {outcome_name} | {active_label}] Flagged sessions: none "
                    f"({len(initial_unique_session_ids)} unique session IDs)."
                )

# %%
# ---------------------------------------------------------
#  SIGNIFICANCE TESTS AND PLOTS
# ---------------------------------------------------------

if not df_channel_coef_sessions.empty:
    df_channel_sig = compute_significance_stats(df_channel_coef_sessions, value_col="coef")
else:
    df_channel_sig = pd.DataFrame(
        columns=[
            "channel",
            "outcome",
            "term",
            "p_active_vs_zero",
            "p_passive_vs_zero",
            "p_active_vs_passive",
        ]
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
            filename=f"channel_mlr_active_passive_coefficients_{channel_name}_{outcome_name}.pdf",
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
    g_channel_r2.fig.suptitle("Channel model R^2 by outcome and block", y=1.03)
    for ax in g_channel_r2.axes.flat:
        _style_seaborn_box_artists(ax)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=20)
    # Relabel legend entries True/False -> active/passive
    if g_channel_r2._legend is not None:
        for text in g_channel_r2._legend.texts:
            if text.get_text() == "True":
                text.set_text("active")
            elif text.get_text() == "False":
                text.set_text("passive")

    if SAVE_PLOTS:
        g_channel_r2.figure.savefig(
            OUTPUT_DIR / "channel_mlr_active_passive_r2_by_channel.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(g_channel_r2.figure)

if SAVE_PLOTS and not df_channel_coef.empty:
    df_channel_coef.to_csv(OUTPUT_DIR / "channel_mlr_active_passive_coefficients.csv", index=False)
if SAVE_PLOTS and not df_channel_sig.empty:
    df_channel_sig.to_csv(OUTPUT_DIR / "channel_mlr_active_passive_significance.csv", index=False)
if SAVE_PLOTS and not df_channel_r2.empty:
    df_channel_r2.to_csv(OUTPUT_DIR / "channel_mlr_active_passive_r2.csv", index=False)
if SAVE_PLOTS and not df_channel_vif_sessions.empty:
    df_channel_vif_sessions.to_csv(OUTPUT_DIR / "channel_mlr_active_passive_vif_by_session.csv", index=False)
if SAVE_PLOTS and not df_channel_problematic_sessions.empty:
    df_channel_problematic_sessions.to_csv(
        OUTPUT_DIR / "channel_mlr_active_passive_problematic_sessions.csv", index=False
    )


# %%
# -------------------------------------------------------------------------------
# CROSS-TERM VISUALIZATION: 2x2 Grid of Source Predictors → Target Metrics
# -------------------------------------------------------------------------------
# Rows: source activity (rbar), source dimensionality (PR)
# Cols: target activity (rbar), target dimensionality (PR)
# Separate active / passive panels, colored in shades of the box palette.

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

# Diverging single-hue maps around white (0), anchored to the boxplot colors.
CMAP_ACTIVE = LinearSegmentedColormap.from_list(
    "active_blue",
    ["#9FC4FF", "#FFFFFF", "#2A7FFF"],
)
CMAP_PASSIVE = LinearSegmentedColormap.from_list(
    "passive_red",
    ["#E08A8A", "#FFFFFF", "#AA0000"],
)
CROSS_CMAPS = {True: CMAP_ACTIVE, False: CMAP_PASSIVE}


def compute_sem(x):
    x_clean = pd.Series(x).dropna()
    if len(x_clean) > 1:
        return float(x_clean.sem())
    return np.nan


def plot_cross_term_heatmap(df_coef_sessions, channel_name, min_sessions=10):
    """
    2x2 heatmap of mean session coefficients (± SEM):
        source activity / source PR  →  target activity / target PR
    One panel each for passive and active, with shared color limits.
    """
    eff = df_coef_sessions[
        (df_coef_sessions["channel"] == channel_name)
        & (df_coef_sessions["term"].isin(CROSS_PRED_ORDER))
        & (df_coef_sessions["outcome"].isin(CROSS_OUTCOME_ORDER))
    ].copy()
    if eff.empty:
        print(f"No cross-term data for {channel_name}.")
        return

    # Require enough sessions per (term, outcome, active)
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

    matrices = {}
    annot_matrices = {}
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
    # Active left, passive right
    panel_order = [True, False]

    for idx, active_val in enumerate(panel_order):
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
        cbar = fig.colorbar(
            ax.collections[0],
            cax=cax,
            orientation="horizontal",
        )
        cbar.set_label("Coefficient (mean ± SEM)", fontsize=10)

    fig.suptitle(
        f"{channel_display_labels[channel_name]}: source → target",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    filename = f"channel_mlr_active_passive_cross_term_{channel_name}.pdf"
    if SAVE_PLOTS:
        fig.savefig(OUTPUT_DIR / filename, format="pdf", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


print("Creating cross-term heatmaps (source activity/PR → target activity/PR).")
CROSS_MIN_SESSIONS = 10
for channel_name in channel_order:
    print(f"=== {channel_display_labels[channel_name]} ===")
    plot_cross_term_heatmap(
        df_channel_coef_sessions,
        channel_name,
        min_sessions=CROSS_MIN_SESSIONS,
    )

# %%
