# %%
"""Plots for covariance stats split by unengagement and speed/pupil terciles,
plus session-level MLR of PR on behavior and channel MLR.

Reads clean_scripts/data/suppl_stats_*.pk and df_stats_area.pk.
Writes figures to clean_scripts/results/.
"""
from pathlib import Path
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

DATA_DIR = _SCRIPT_DIR.parent / "data"
RESULTS_DIR = _SCRIPT_DIR.parent / "results"
SAVE_PLOTS = True
SHOW_PLOTS = True
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

area_order = ['LGd', 'VISpl', 'VISm', 'SCMRN']
# area_order = ['VISp', 'VISrl', 'VISl', 'VISal', 'VISpm', 'VISam']
tercile_order = ['low', 'mid', 'high']
active_order = [True, False]
tercile_pair_keys = ["session_id", "image_name", "active", "area", "tercile"]
SPEED_BASE_COLOR = "#4D4D4D"
PUPIL_BASE_COLOR = "#8E6BBE"


def finalize_catplot(grid, filename):
    if SAVE_PLOTS:
        grid.figure.savefig(RESULTS_DIR / filename, format="pdf", bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(grid.figure)


def compute_treves_rolls(avgspikecounts, bin_width=0.05):
    rates = np.asarray(avgspikecounts, dtype=float).ravel() / bin_width
    if rates.size == 0:
        return np.nan
    mean_rate = np.nanmean(rates)
    mean_rate_sq = np.nanmean(rates ** 2)
    if np.isnan(mean_rate) or np.isnan(mean_rate_sq) or mean_rate_sq == 0:
        return np.nan
    return (mean_rate ** 2) / mean_rate_sq


def normalize_pr_by_n(row):
    counts = np.asarray(row["avgspikecounts"], dtype=float).ravel()
    n = counts.size
    if n == 0 or pd.isna(row["PR_emp"]):
        return np.nan
    return row["PR_emp"] / n


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


def _ttest_vs_zero_asterisk(vals):
    vals = pd.Series(vals).dropna()
    if len(vals) < 2:
        return ""
    _, p = stats.ttest_1samp(vals.to_numpy(dtype=float), 0.0)
    return get_asterisk(p)


def _annotate_forest_vs_zero(ax, df_sessions, terms, value_col="coef"):
    """Asterisks for one-sample t-test of session coefficients vs 0."""
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin if xmax > xmin else 1.0
    x_pad = 0.03 * span
    for y, term in enumerate(terms):
        vals = df_sessions.loc[df_sessions["term"] == term, value_col].dropna()
        ast = _ttest_vs_zero_asterisk(vals)
        if not ast:
            continue
        mean = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.text(
            mean + sem + x_pad,
            y,
            ast,
            ha="left",
            va="center",
            color="black",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )
    ax.set_xlim(xmin, xmax + 0.12 * span)


def _print_coef_ttests(df_sessions, label, terms, value_col="coef"):
    print(f"\nOne-sample t-test vs 0 ({label}):")
    for term in terms:
        vals = df_sessions.loc[df_sessions["term"] == term, value_col].dropna()
        if len(vals) < 2:
            print(f"  {term}: n={len(vals)} (skip)")
            continue
        t_stat, p = stats.ttest_1samp(vals.to_numpy(dtype=float), 0.0)
        print(
            f"  {term}: n={len(vals)} mean={vals.mean():.3f} "
            f"t={t_stat:.2f} p={p:.3g} {get_asterisk(p)}"
        )


def _draw_xaxis_cut_marks(ax_left, ax_right, size=0.03):
    """Diagonal marks at a broken x-axis join."""
    kwargs = dict(color="k", clip_on=False, linewidth=1.0)
    ax_left.plot((1 - size, 1 + size), (-size, size), transform=ax_left.transAxes, **kwargs)
    ax_left.plot((1 - size, 1 + size), (1 - size, 1 + size), transform=ax_left.transAxes, **kwargs)
    ax_right.plot((-size, size), (-size, size), transform=ax_right.transAxes, **kwargs)
    ax_right.plot((-size, size), (1 - size, 1 + size), transform=ax_right.transAxes, **kwargs)


def plot_forest_negative_xaxis_cut(
    df_agg,
    df_sessions,
    term_labels,
    filename,
    title,
    color="C1",
    cut_term="active",
    left_xlim=(-1.60, -0.98),
    right_xlim=(-0.85, 1.10),
    figsize=(5.2, 4.0),
):
    """Forest plot with a cut on the negative x-axis (same canvas as other forests)."""
    df_plot = df_agg.sort_values("term")
    terms = df_plot["term"].tolist()
    y = np.arange(len(df_plot))
    means = df_plot["coef_mean"].to_numpy(dtype=float)
    sems = df_plot["coef_sem"].to_numpy(dtype=float)
    on_left = np.array([t == cut_term for t in terms])

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.15, 5.05], "wspace": 0.08},
    )
    for ax, mask in ((ax_l, on_left), (ax_r, ~on_left)):
        if not np.any(mask):
            continue
        ax.errorbar(
            means[mask],
            y[mask],
            xerr=sems[mask],
            fmt="o",
            color=color,
            capsize=3,
        )
    ax_r.axvline(0, color="black", linestyle="--", linewidth=1)

    ax_l.set_xlim(*left_xlim)
    ax_r.set_xlim(*right_xlim)
    ax_l.set_xticks([-1.4, -1.2])
    ax_r.set_xticks([-0.5, 0.0, 0.5, 1.0])
    ax_l.set_yticks(y)
    ax_l.set_yticklabels([term_labels.get(t, t) for t in terms])
    ax_r.tick_params(axis="y", left=False, labelleft=False)

    ax_l.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    _draw_xaxis_cut_marks(ax_l, ax_r)

    for ax, mask, xlim in ((ax_l, on_left, left_xlim), (ax_r, ~on_left, right_xlim)):
        span = xlim[1] - xlim[0]
        x_pad = 0.04 * span
        for y_i, term, mean, sem in zip(y[mask], np.array(terms)[mask], means[mask], sems[mask]):
            ast = _ttest_vs_zero_asterisk(df_sessions.loc[df_sessions["term"] == term, "coef"])
            if not ast:
                continue
            x_text = mean + sem + x_pad
            if x_text >= xlim[1]:
                x_text = xlim[1] - 0.02 * span
            ax.text(
                x_text,
                y_i,
                ast,
                ha="left",
                va="center",
                color="black",
                fontsize=10,
                fontweight="bold",
                clip_on=False,
            )

    ax_l.set_ylabel("Predictor")
    fig.supxlabel("Coefficient (z-scored PR_emp)", y=0.04, fontsize=10)
    fig.suptitle(title, y=0.98, fontsize=11)
    fig.subplots_adjust(left=0.28, right=0.97, top=0.88, bottom=0.16)

    if SAVE_PLOTS:
        fig.savefig(RESULTS_DIR / filename, format="pdf", bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _require_image_name(df, pickle_name):
    if "image_name" not in df.columns:
        raise KeyError(
            f"{pickle_name} has no image_name. "
            "Re-run generate_covstats_behavior.py so stats are "
            "grouped by (session_id, image_name)."
        )


def _mix_hex_with_white(hex_color, white_frac):
    """Tint a hex color toward white. white_frac=0 keeps the color; 1 is white."""
    hex_color = hex_color.lstrip("#")
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)
    mixed = rgb + (255.0 - rgb) * white_frac
    return "#{:02X}{:02X}{:02X}".format(*np.clip(mixed, 0, 255).astype(int))


def tercile_palette(base_hex):
    """high = base, mid/low = lighter tints. Opaque (no alpha)."""
    return {
        "high": _mix_hex_with_white(base_hex, 0.0),
        "mid": _mix_hex_with_white(base_hex, 0.32),
        "low": _mix_hex_with_white(base_hex, 0.58),
    }


SPEED_TERCILE_PALETTE = tercile_palette(SPEED_BASE_COLOR)
PUPIL_TERCILE_PALETTE = tercile_palette(PUPIL_BASE_COLOR)
TERCILE_METRIC_SPECS = [
    ("mean_firing_rate", "mean firing rate", "mean_firing_rate"),
    ("PR_emp", "PR_emp", "pr_emp"),
    ("TR_emp", "TR_emp × N", "tr_emp"),
]
TERCILE_XLIM = (-0.5, len(area_order) - 0.5)


def _pointplot_y_range(df, y_col):
    """95% t-interval around cell means, matching seaborn pointplot error bars."""
    ymin, ymax = np.inf, -np.inf
    grouped = df.groupby(["area", "tercile", "active"], observed=True, dropna=False)[y_col]
    for _, vals in grouped:
        vals = vals.dropna()
        if vals.empty:
            continue
        mean = float(vals.mean())
        n = len(vals)
        half = 0.0
        if n > 1:
            sem = float(vals.std(ddof=1) / np.sqrt(n))
            half = float(stats.t.ppf(0.975, n - 1) * sem)
        ymin = min(ymin, mean - half)
        ymax = max(ymax, mean + half)
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return None
    return ymin, ymax


def _shared_tercile_ylim(dfs, y_col, pad_frac=0.05):
    ranges = [_pointplot_y_range(df, y_col) for df in dfs]
    ranges = [r for r in ranges if r is not None]
    if not ranges:
        return None
    ymin = min(r[0] for r in ranges)
    ymax = max(r[1] for r in ranges)
    if ymax <= ymin:
        pad = 0.05 if ymax == 0 else 0.05 * abs(ymax)
        return ymin - pad, ymax + pad
    pad = pad_frac * (ymax - ymin)
    return ymin - pad, ymax + pad


def plot_tercile_pointplots(df_avg, title_prefix, file_prefix, metric_specs, palette, ylims=None):
    """Seaborn point plots; each row is a (session_id, image_name) pair."""
    n_pairs = df_avg.drop_duplicates(subset=["session_id", "image_name"]).shape[0]
    print(
        f"{title_prefix}: {len(df_avg)} points "
        f"({n_pairs} unique session_id × image_name pairs)"
    )
    ylims = ylims or {}
    for y_col, ylabel, stem in metric_specs:
        g = sns.catplot(
            data=df_avg,
            kind="point",
            x="area",
            y=y_col,
            hue="tercile",
            col="active",
            sharex=True,
            sharey=True,
            order=area_order,
            hue_order=tercile_order,
            col_order=active_order,
            palette=palette,
        )
        for ax in g.axes.flat:
            for artist in list(ax.lines) + list(ax.collections):
                artist.set_alpha(1.0)
            ax.set_xlim(*TERCILE_XLIM)
            if y_col in ylims and ylims[y_col] is not None:
                ax.set_ylim(*ylims[y_col])
        legend = g._legend
        if legend is not None:
            for handle in legend.legend_handles:
                handle.set_alpha(1.0)
        g.fig.suptitle(f"{title_prefix}: {ylabel}", y=1.03)
        finalize_catplot(g, f"{file_prefix}_{stem}_point.pdf")


AREA_PALETTE = ["#03045E", "#023E8A", "#0077B6", "#0096C7", "#00B4D8", "#48CAE4"]
AREA_COLORS = {area: AREA_PALETTE[i % len(AREA_PALETTE)] for i, area in enumerate(area_order)}


def _ratio_box_data(df_metric):
    box_data = []
    box_colors = []
    tick_labels = []
    for area in area_order:
        vals = df_metric.loc[df_metric["area"] == area, "ratio"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        box_data.append(vals)
        box_colors.append(AREA_COLORS[area])
        tick_labels.append(area)
    return box_data, box_colors, tick_labels


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


def _style_empty_boxes(bp, box_colors):
    for i, (patch, color) in enumerate(zip(bp["boxes"], box_colors)):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        bp["whiskers"][i * 2].set_color(color)
        bp["whiskers"][i * 2 + 1].set_color(color)
        bp["whiskers"][i * 2].set_linewidth(1.5)
        bp["whiskers"][i * 2 + 1].set_linewidth(1.5)
        bp["caps"][i * 2].set_color(color)
        bp["caps"][i * 2 + 1].set_color(color)
        bp["caps"][i * 2].set_linewidth(1.5)
        bp["caps"][i * 2 + 1].set_linewidth(1.5)
        bp["medians"][i].set_color(color)
        bp["medians"][i].set_linewidth(1.5)


def _print_ratio_ttests(df_metric, metric, tick_labels):
    print(f"\nOne-sample t-test vs 1 ({metric}, engaged/unengaged):")
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


def _finalize_ratio_fig(fig, filename):
    fig.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(RESULTS_DIR / filename, format="pdf", bbox_inches="tight", dpi=300)
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()


def plot_filled_ratio_boxplots(df_ratios, metric, ylabel, filename):
    """Filled boxplots (vis_circuit_metrics supplementary style) of a ratio vs area."""
    df_metric = df_ratios[df_ratios["metric"] == metric].copy()
    if df_metric.empty:
        return
    box_data, box_colors, tick_labels = _ratio_box_data(df_metric)
    if not box_data:
        return

    y_min = df_metric["ratio"].min()
    y_max = df_metric["ratio"].max()
    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
    y_lim = (y_min - y_margin, y_max + y_margin)

    fig, ax = plt.subplots(figsize=(8, 5))
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
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_xlabel("Area")
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_lim)
    ax.set_xlim(-0.5, len(box_data) - 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    _annotate_one_sample_vs_mu(
        ax, df_metric, "ratio", tick_labels, mu=1.0, category_col="area",
    )
    _pad_ylim_for_asterisks(ax)
    _print_ratio_ttests(df_metric, metric, tick_labels)
    _finalize_ratio_fig(fig, filename)


def plot_empty_ratio_box_strip(df_ratios, metric, ylabel, filename):
    """Empty colored boxplots plus rasterized strip points (one per session_id, image_name)."""
    df_metric = df_ratios[df_ratios["metric"] == metric].copy()
    if df_metric.empty:
        return
    box_data, box_colors, tick_labels = _ratio_box_data(df_metric)
    if not box_data:
        return

    y_min = df_metric["ratio"].min()
    y_max = df_metric["ratio"].max()
    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
    y_lim = (y_min - y_margin, y_max + y_margin)

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        box_data,
        positions=range(len(box_data)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        zorder=2,
    )
    _style_empty_boxes(bp, box_colors)

    rng = np.random.default_rng(0)
    for i, area in enumerate(tick_labels):
        vals = df_metric.loc[df_metric["area"] == area, "ratio"].dropna().to_numpy()
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full(len(vals), i, dtype=float) + jitter,
            vals,
            s=14,
            c=box_colors[i],
            alpha=0.45,
            edgecolors="none",
            zorder=3,
            rasterized=True,
        )

    ax.set_xticks(range(len(box_data)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_xlabel("Area")
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_lim)
    ax.set_xlim(-0.5, len(box_data) - 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    _annotate_one_sample_vs_mu(
        ax, df_metric, "ratio", tick_labels, mu=1.0, category_col="area",
    )
    _pad_ylim_for_asterisks(ax)
    _finalize_ratio_fig(fig, filename)
# %%
# ---------------------------------------
#  UNENGAGEMENT RAISES PR IN ACTIVE BLOCK
# ---------------------------------------
unengagement_file = DATA_DIR / "suppl_stats_unengagement.pk"
df_bhvr_engaged = pd.read_pickle(unengagement_file)

df_bhvr_engaged = df_bhvr_engaged[df_bhvr_engaged.session_has_unengaged_state==True]
df_bhvr_engaged["N"] = df_bhvr_engaged["avgspikecounts"].apply(lambda x: np.asarray(x, dtype=float).ravel().size)
df_bhvr_engaged = df_bhvr_engaged[df_bhvr_engaged["N"] >= 30].copy()
df_bhvr_engaged = df_bhvr_engaged[df_bhvr_engaged["area"].isin(area_order)].copy()

df_bhvr_engaged['mean_firing_rate'] = df_bhvr_engaged['avgspikecounts'].apply(
    lambda x: np.asarray(x, dtype=float).ravel().mean() / 0.05)
df_bhvr_engaged["PR_emp"] = df_bhvr_engaged.apply(normalize_pr_by_n, axis=1)
df_bhvr_engaged["TR_emp"] = df_bhvr_engaged["avgspikecounts"].apply(compute_treves_rolls)
df_bhvr_engaged = df_bhvr_engaged.drop(
    columns=[
        "overall_unengagement_time",
        "view_flag",
        "session_has_unengaged_state",
        "active",
        "session_number",
        "experience_level",
        "n_bins_used",
        "behavior_metric",
        "avgspikecounts",
    ],
    errors="ignore",
)

stat_cols = ["mii", "mij", "sii", "sij", "TV_emp", "PR_emp", "PR_fit", "TR_emp", "mean_firing_rate"]
if "image_name" not in df_bhvr_engaged.columns:
    raise KeyError(
        "suppl_stats_unengagement.pk has no image_name. "
        "Re-run generate_covstats_behavior.py so unengagement stats are "
        "grouped by (session_id, image_name)."
    )

ratio_keys = ["session_id", "image_name", "area"]
df_bhvr_avg = (
    df_bhvr_engaged.groupby(ratio_keys + ["unengaged"], as_index=False)[stat_cols].mean()
)

ratio_metrics = ["mean_firing_rate", "PR_emp", "TR_emp"]
ratio_rows = []
for metric in ratio_metrics:
    df_pivot = df_bhvr_avg.pivot_table(
        index=ratio_keys,
        columns="unengaged",
        values=metric,
        aggfunc="mean",
    ).reset_index()
    if True not in df_pivot.columns or False not in df_pivot.columns:
        continue
    # engaged = unengaged False; unengaged = True
    df_pivot["ratio"] = df_pivot[False] / df_pivot[True]
    df_pivot = df_pivot[df_pivot["ratio"].notna() & np.isfinite(df_pivot["ratio"])]
    df_pivot["metric"] = metric
    ratio_rows.append(df_pivot[ratio_keys + ["ratio", "metric"]])

df_uneng_ratios = pd.concat(ratio_rows, ignore_index=True) if ratio_rows else pd.DataFrame()
if not df_uneng_ratios.empty:
    df_uneng_ratios["area"] = pd.Categorical(
        df_uneng_ratios["area"], categories=area_order, ordered=True
    )
    n_dup = int(
        df_uneng_ratios.duplicated(subset=ratio_keys + ["metric"]).sum()
    )
    n_pairs = df_uneng_ratios.drop_duplicates(subset=["session_id", "image_name"]).shape[0]
    print(
        f"Unengagement ratios: {len(df_uneng_ratios)} points "
        f"({n_pairs} unique session_id × image_name pairs; "
        f"{n_dup} duplicate (session_id, image_name, area, metric) rows)"
    )
    if n_dup != 0:
        raise ValueError("Expected one point per (session_id, image_name, area, metric)")

    metric_ylabels = {
        "mean_firing_rate": "Engaged / Unengaged Ratio (mean firing rate)",
        "PR_emp": "Engaged / Unengaged Ratio (PR_emp)",
        "TR_emp": "Engaged / Unengaged Ratio (TR_emp)",
    }
    for metric in ratio_metrics:
        plot_filled_ratio_boxplots(
            df_uneng_ratios,
            metric,
            metric_ylabels[metric],
            f"suppl_covstats_with_bhvr_unengagement_{metric}_ratio_box.pdf",
        )
        plot_empty_ratio_box_strip(
            df_uneng_ratios,
            metric,
            metric_ylabels[metric],
            f"suppl_covstats_with_bhvr_unengagement_{metric}_ratio_box_strip.pdf",
        )
# %%
# --------------------------------------------------------------
#  PUPIL SIZE DOES NOT AFFECT METRIC TRENDS ACROSS ACTIVE BLOCKS
# --------------------------------------------------------------
pupil_file = DATA_DIR / "suppl_stats_pupil_terciles.pk"
df_bhvr_pupil = pd.read_pickle(pupil_file)
_require_image_name(df_bhvr_pupil, "suppl_stats_pupil_terciles.pk")
df_bhvr_pupil["N"] = df_bhvr_pupil["avgspikecounts"].apply(lambda x: np.asarray(x, dtype=float).ravel().size)
df_bhvr_pupil = df_bhvr_pupil[df_bhvr_pupil["N"] >= 30].copy()
df_bhvr_pupil = df_bhvr_pupil[df_bhvr_pupil["area"].isin(area_order)].copy()
df_bhvr_pupil['mean_firing_rate'] = df_bhvr_pupil['avgspikecounts'].apply(
    lambda x: np.asarray(x, dtype=float).ravel().mean() / 0.05)
df_bhvr_pupil["TR_emp"] = df_bhvr_pupil["avgspikecounts"].apply(compute_treves_rolls) * df_bhvr_pupil["N"]
df_bhvr_pupil = df_bhvr_pupil.drop(
    columns=["session_number", "experience_level", "n_bins_used", "behavior_metric", "view_flag", "avgspikecounts"],
    errors="ignore",
)

stat_cols = ['mii', 'mij', 'sii', 'sij', 'TV_emp', 'PR_emp', 'PR_fit', 'TR_emp', 'mean_firing_rate']
df_pupil_avg = (
    df_bhvr_pupil.groupby(tercile_pair_keys, as_index=False)[stat_cols].mean()
)

# PR increases with pupil size in active block

# %%
# ---------------------------------------------------------
#  SPEED DOES NOT AFFECT METRIC TRENDS ACROSS ACTIVE BLOCKS
# ---------------------------------------------------------
speed_file = DATA_DIR / "suppl_stats_speed_terciles.pk"
df_bhvr_speed = pd.read_pickle(speed_file)
_require_image_name(df_bhvr_speed, "suppl_stats_speed_terciles.pk")
df_bhvr_speed["N"] = df_bhvr_speed["avgspikecounts"].apply(lambda x: np.asarray(x, dtype=float).ravel().size)
df_bhvr_speed = df_bhvr_speed[df_bhvr_speed["N"] >= 30].copy()
df_bhvr_speed = df_bhvr_speed[df_bhvr_speed["area"].isin(area_order)].copy()
df_bhvr_speed['mean_firing_rate'] = df_bhvr_speed['avgspikecounts'].apply(
    lambda x: np.asarray(x, dtype=float).ravel().mean() / 0.05)
df_bhvr_speed["TR_emp"] = df_bhvr_speed["avgspikecounts"].apply(compute_treves_rolls) * df_bhvr_speed["N"]
df_bhvr_speed = df_bhvr_speed.drop(
    columns=["session_number", "experience_level", "n_bins_used", "behavior_metric", "view_flag", "avgspikecounts"],
    errors="ignore",
)

stat_cols = ['mii', 'mij', 'sii', 'sij', 'TV_emp', 'PR_emp', 'PR_fit', 'TR_emp', 'mean_firing_rate']
df_speed_avg = (
    df_bhvr_speed.groupby(tercile_pair_keys, as_index=False)[stat_cols].mean()
)

# Shared x/y limits per metric across pupil and speed, and across active/passive.
tercile_ylims = {
    y_col: _shared_tercile_ylim([df_pupil_avg, df_speed_avg], y_col)
    for y_col, _ylabel, _stem in TERCILE_METRIC_SPECS
}
print("Shared tercile y-limits (pupil + speed, active + passive):")
for y_col, ylim in tercile_ylims.items():
    print(f"  {y_col}: {ylim}")

plot_tercile_pointplots(
    df_pupil_avg,
    "Pupil terciles",
    "suppl_covstats_with_bhvr_pupil_terciles",
    TERCILE_METRIC_SPECS,
    palette=PUPIL_TERCILE_PALETTE,
    ylims=tercile_ylims,
)
plot_tercile_pointplots(
    df_speed_avg,
    "Speed terciles",
    "suppl_covstats_with_bhvr_speed_terciles",
    TERCILE_METRIC_SPECS,
    palette=SPEED_TERCILE_PALETTE,
    ylims=tercile_ylims,
)

# PR increases with speed in active (and passive?) block

# %%
# ---------------------------------------------------------
#  MULTIPLE LINEAR REGRESSION (FROM suppl_behavior)
# ---------------------------------------------------------

try:
    smf = importlib.import_module("statsmodels.formula.api")
except ModuleNotFoundError:
    smf = None


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


def zscore_within_group(series, groups):
    grouped = series.groupby(groups)
    centered = series - grouped.transform("mean")
    std = grouped.transform(lambda x: x.std(ddof=0))
    z = centered / std.replace(0, np.nan)
    z = pd.to_numeric(z, errors="coerce")
    return z.astype(float).fillna(0.0)


def standardize_within_session(df_sess):
    df_sess = df_sess.copy()
    # Center and scale behavior predictors within (session, active) groups so
    # condition-level shifts do not drive slope estimates.
    df_sess["speed_mean_z"] = zscore_within_group(df_sess["speed_mean"], df_sess["active"])
    df_sess["pupil_mean_z"] = zscore_within_group(df_sess["pupil_mean"], df_sess["active"])
    # Keep outcome standardized at session level for coefficient comparability.
    df_sess["PR_emp_z"] = zscore(df_sess["PR_emp"])
    return df_sess


def build_design_matrix(df_sess):
    X = pd.DataFrame(
        {
            "active": df_sess["active"].astype(float),
            "speed_mean_z": df_sess["speed_mean_z"].astype(float),
            "pupil_mean_z": df_sess["pupil_mean_z"].astype(float),
        }
    )
    X["active:speed_mean_z"] = X["active"] * X["speed_mean_z"]
    X["active:pupil_mean_z"] = X["active"] * X["pupil_mean_z"]
    return X


def compute_session_collinearity(df_sess):
    X = build_design_matrix(df_sess)
    records = []
    for term in X.columns:
        y = X[term]
        X_other = X.drop(columns=[term])
        if y.std(ddof=0) == 0:
            vif = np.inf
        else:
            r2 = LinearRegression().fit(X_other, y).score(X_other, y)
            vif = np.inf if np.isclose(1.0 - r2, 0.0) else 1.0 / (1.0 - r2)
        records.append({"term": term, "vif": float(vif)})
    max_vif = np.nanmax([r["vif"] for r in records]) if records else np.nan
    return records, max_vif


def fit_session_regression(df_sess, smf_module=None):
    X = build_design_matrix(df_sess)
    if smf_module is not None:
        model = smf_module.ols(formula=REGRESSION_FORMULA, data=df_sess).fit()
        coef_map = model.params.drop(labels=["Intercept"], errors="ignore").to_dict()
        return coef_map, model.rsquared

    y = df_sess["PR_emp_z"].astype(float)
    reg = LinearRegression().fit(X, y)
    coef_map = dict(zip(X.columns, reg.coef_))
    return coef_map, reg.score(X, y)


stats_file = DATA_DIR / "df_stats_area.pk"
df_stats_all = pd.read_pickle(stats_file)

data_filter = (
    (df_stats_all.session_number == 1)
    & (df_stats_all.experience_level == "Familiar")
    & (df_stats_all.image_name != "omitted")
    & (df_stats_all.is_image_novel == False)
    & (df_stats_all.is_change == False)
)
df_stats = df_stats_all[data_filter].drop(
    ["session_number", "experience_level", "is_image_novel", "is_change"],
    axis=1,
    errors="ignore",
)
if "N" not in df_stats.columns:
    df_stats["N"] = df_stats["avgspikecounts"].apply(lambda x: np.asarray(x, dtype=float).ravel().size)
df_stats = df_stats[df_stats["N"] >= 30].copy()
df_stats = df_stats.drop(
    ["PR_fit", "ciisq", "cijsq", "TS_emp", "avgspikecounts", "is_peak"],
    axis=1,
    errors="ignore",
)
if "subsample_id" not in df_stats.columns:
    df_stats["subsample_id"] = (
        df_stats.groupby(["session_id", "active", "image_set", "image_name", "area"], dropna=False).cumcount()
    )


df_model = df_stats[df_stats["area"].isin(area_order)].copy()
df_model["speed_arr"] = df_model["speed_values"].apply(unpack_list_of_one)
df_model["pupil_arr"] = df_model["pupil_values"].apply(unpack_list_of_one)

df_model["speed_mean"] = df_model["speed_arr"].apply(np.nanmean)
df_model["pupil_mean"] = df_model["pupil_arr"].apply(np.nanmean)
df_model["active"] = df_model["active"].astype(int)

# Average 200-bin chunks so each row is a (session_id, image_name) pair.
mlr_pair_keys = ["session_id", "image_name", "active", "area"]
df_model = (
    df_model.groupby(mlr_pair_keys, as_index=False)
    .agg(
        PR_emp=("PR_emp", "mean"),
        speed_mean=("speed_mean", "mean"),
        pupil_mean=("pupil_mean", "mean"),
        N=("N", "first"),
    )
)
n_mlr_pairs = df_model.drop_duplicates(subset=["session_id", "image_name"]).shape[0]
print(
    f"PR~behavior MLR rows: {len(df_model)} "
    f"({n_mlr_pairs} unique session_id × image_name pairs)"
)

MIN_ROWS_PER_SESSION = 8
VIF_FLAG_THRESHOLD = 5.0
REGRESSION_FORMULA = (
    "PR_emp_z ~ active + speed_mean_z + pupil_mean_z "
    "+ active:speed_mean_z + active:pupil_mean_z"
)

coef_session_records = []
r2_session_records = []
vif_session_records = []
skipped_sessions = 0

if smf is None:
    print("statsmodels is not installed: using sklearn fallback without per-term p-values.")

for area in area_order:
    df_area = df_model[df_model["area"] == area].copy()
    for session_id, df_sess in df_area.groupby("session_id"):
        if len(df_sess) < MIN_ROWS_PER_SESSION:
            skipped_sessions += 1
            continue
        df_sess = standardize_within_session(df_sess)
        vif_records, max_vif = compute_session_collinearity(df_sess)
        is_problematic = (np.isfinite(max_vif) and max_vif >= VIF_FLAG_THRESHOLD) or (not np.isfinite(max_vif))
        for record in vif_records:
            vif_session_records.append(
                {
                    "area": area,
                    "session_id": session_id,
                    "n_rows": len(df_sess),
                    "term": record["term"],
                    "vif": record["vif"],
                    "max_vif": max_vif,
                    "is_problematic": is_problematic,
                }
            )

        coef_map, r2 = fit_session_regression(df_sess, smf_module=smf)
        r2_session_records.append(
            {"area": area, "session_id": session_id, "r2": r2, "n_rows": len(df_sess)}
        )
        for term, coef in coef_map.items():
            coef_session_records.append(
                {
                    "area": area,
                    "session_id": session_id,
                    "term": term,
                    "coef": coef,
                }
            )

if skipped_sessions > 0:
    print(f"Skipped {skipped_sessions} area-session fits with fewer than {MIN_ROWS_PER_SESSION} rows.")

df_coef_sessions = pd.DataFrame(coef_session_records)
df_r2_sessions = pd.DataFrame(r2_session_records)
df_vif_sessions = pd.DataFrame(vif_session_records)

if not df_coef_sessions.empty:
    df_coef = (
        df_coef_sessions.groupby(["area", "term"], as_index=False)
        .agg(
            coef_mean=("coef", "mean"),
            coef_std=("coef", "std"),
            n_sessions=("session_id", "nunique"),
        )
    )
    df_coef["coef_sem"] = df_coef["coef_std"] / np.sqrt(df_coef["n_sessions"].clip(lower=1))
else:
    df_coef = pd.DataFrame(columns=["area", "term", "coef_mean", "coef_sem", "n_sessions"])

if not df_coef_sessions.empty:
    df_terms = (
        df_coef_sessions.pivot_table(
            index=["area", "session_id"],
            columns="term",
            values="coef",
            aggfunc="first",
        )
        .reset_index()
    )
    required_terms = {
        "active",
        "speed_mean_z",
        "active:speed_mean_z",
        "pupil_mean_z",
        "active:pupil_mean_z",
    }
    if required_terms.issubset(set(df_terms.columns)):
        df_terms["speed_active_full_z"] = df_terms["speed_mean_z"] + df_terms["active:speed_mean_z"]
        df_terms["pupil_active_full_z"] = df_terms["pupil_mean_z"] + df_terms["active:pupil_mean_z"]
        df_coef_full_active_sessions = df_terms[
            [
                "area",
                "session_id",
                "active",
                "speed_mean_z",
                "speed_active_full_z",
                "pupil_mean_z",
                "pupil_active_full_z",
            ]
        ].melt(
            id_vars=["area", "session_id"],
            var_name="term",
            value_name="coef",
        )
        df_coef_full_active = (
            df_coef_full_active_sessions.groupby(["area", "term"], as_index=False)
            .agg(
                coef_mean=("coef", "mean"),
                coef_std=("coef", "std"),
                n_sessions=("session_id", "nunique"),
            )
        )
        df_coef_full_active["coef_sem"] = df_coef_full_active["coef_std"] / np.sqrt(
            df_coef_full_active["n_sessions"].clip(lower=1)
        )
    else:
        df_coef_full_active_sessions = pd.DataFrame(columns=["area", "session_id", "term", "coef"])
        df_coef_full_active = pd.DataFrame(
            columns=["area", "term", "coef_mean", "coef_sem", "n_sessions"]
        )
else:
    df_coef_full_active_sessions = pd.DataFrame(columns=["area", "session_id", "term", "coef"])
    df_coef_full_active = pd.DataFrame(columns=["area", "term", "coef_mean", "coef_sem", "n_sessions"])

if not df_r2_sessions.empty:
    df_r2 = (
        df_r2_sessions.groupby("area", as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            n_sessions=("session_id", "nunique"),
        )
    )
    df_r2["r2_sem"] = df_r2["r2_std"] / np.sqrt(df_r2["n_sessions"].clip(lower=1))
else:
    df_r2 = pd.DataFrame(columns=["area", "r2_mean", "r2_std", "r2_sem", "n_sessions"])

if not df_vif_sessions.empty:
    df_problematic_sessions = (
        df_vif_sessions[df_vif_sessions["is_problematic"]]
        .groupby(["area", "session_id"], as_index=False)
        .agg(
            n_rows=("n_rows", "first"),
            max_vif=("max_vif", "first"),
        )
    )
else:
    df_problematic_sessions = pd.DataFrame(
        columns=["area", "session_id", "n_rows", "max_vif"]
    )

if not df_r2_sessions.empty:
    df_fitted_area_sessions = (
        df_r2_sessions[["area", "session_id", "n_rows"]]
        .drop_duplicates()
        .sort_values(["area", "session_id"])
        .reset_index(drop=True)
    )
    total_fitted_area_sessions = len(df_fitted_area_sessions)
    total_fitted_unique_sessions = df_fitted_area_sessions["session_id"].nunique()
else:
    df_fitted_area_sessions = pd.DataFrame(columns=["area", "session_id", "n_rows"])
    total_fitted_area_sessions = 0
    total_fitted_unique_sessions = 0

print(
    "Initial fitted sessions available: "
    f"{total_fitted_area_sessions} area-session fits "
    f"({total_fitted_unique_sessions} unique session IDs)."
)
initial_unique_session_ids = sorted(df_fitted_area_sessions["session_id"].unique().tolist())
print(f"Unique session_id before VIF filtering ({len(initial_unique_session_ids)}): {initial_unique_session_ids}")

if not df_problematic_sessions.empty:
    df_problematic_sessions = df_problematic_sessions.sort_values(["area", "session_id"]).reset_index(
        drop=True
    )
    flagged_unique_session_ids = sorted(df_problematic_sessions["session_id"].unique().tolist())
    remaining_unique_session_ids = sorted(set(initial_unique_session_ids) - set(flagged_unique_session_ids))
    print("Flagged sessions (area-session fits):")
    print(
        df_problematic_sessions.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )
    remaining_after_flagging = total_fitted_area_sessions - len(df_problematic_sessions)
    print(
        "Flagged "
        f"{len(df_problematic_sessions)} area-session fits with max VIF >= {VIF_FLAG_THRESHOLD} "
        "(or non-finite VIF)."
    )
    print(
        "Remaining sessions after flagging: "
        f"{remaining_after_flagging} area-session fits."
    )
    print(
        f"Unique session_id after VIF filtering ({len(remaining_unique_session_ids)}): "
        f"{remaining_unique_session_ids}"
    )
else:
    print("Flagged sessions (area-session fits): none")
    print(
        "Remaining sessions after flagging: "
        f"{total_fitted_area_sessions} area-session fits."
    )
    print(f"Unique session_id after VIF filtering ({len(initial_unique_session_ids)}): {initial_unique_session_ids}")

# Coefficients: mean ± SEM across sessions
if not df_coef.empty:
    term_order = [
        "active",
        "speed_mean_z",
        "active:speed_mean_z",
        "pupil_mean_z",
        "active:pupil_mean_z",
    ]
    df_coef["term"] = pd.Categorical(df_coef["term"], categories=term_order, ordered=True)
    term_labels = {
        "active": "active",
        "speed_mean_z": "speed",
        "active:speed_mean_z": "speed x active",
        "pupil_mean_z": "pupil",
        "active:pupil_mean_z": "pupil x active",
    }

    for area in area_order:
        df_a = df_coef[df_coef["area"] == area].sort_values("term")
        if df_a.empty:
            continue

        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        y = np.arange(len(df_a))
        ax.errorbar(
            df_a["coef_mean"],
            y,
            xerr=df_a["coef_sem"],
            fmt="o",
            color="C0",
            capsize=3,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([term_labels.get(t, t) for t in df_a["term"]])
        ax.set_xlabel("Coefficient (z-scored PR_emp)")
        ax.set_ylabel("Predictor")
        ax.set_title(f"Regression coefficients - {area}")
        ax.set_xlim(-1.1, 1.1)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        df_sess_area = df_coef_sessions[df_coef_sessions["area"] == area]
        plot_terms = df_a["term"].tolist()
        _annotate_forest_vs_zero(ax, df_sess_area, plot_terms)
        _print_coef_ttests(df_sess_area, f"PR~behavior {area}", plot_terms)
        fig.tight_layout()

        if SAVE_PLOTS:
            fig.savefig(
                RESULTS_DIR / f"suppl_covstats_with_bhvr_regression_coefficients_{area}.pdf",
                format="pdf",
                bbox_inches="tight",
            )
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)

# Coefficients (full active slopes): mean ± SEM across sessions
if not df_coef_full_active.empty:
    term_order_full_active = [
        "active",
        "speed_mean_z",
        "speed_active_full_z",
        "pupil_mean_z",
        "pupil_active_full_z",
    ]
    df_coef_full_active["term"] = pd.Categorical(
        df_coef_full_active["term"], categories=term_order_full_active, ordered=True
    )
    term_labels_full_active = {
        "active": "active",
        "speed_mean_z": "speed (passive)",
        "speed_active_full_z": "speed (active full)",
        "pupil_mean_z": "pupil (passive)",
        "pupil_active_full_z": "pupil (active full)",
    }

    for area in area_order:
        df_a = df_coef_full_active[df_coef_full_active["area"] == area].sort_values("term")
        if df_a.empty:
            continue

        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        y = np.arange(len(df_a))
        ax.errorbar(
            df_a["coef_mean"],
            y,
            xerr=df_a["coef_sem"],
            fmt="o",
            color="C1",
            capsize=3,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([term_labels_full_active.get(t, t) for t in df_a["term"]])
        ax.set_xlabel("Coefficient (z-scored PR_emp)")
        ax.set_ylabel("Predictor")
        ax.set_title(f"Regression coefficients (full active slopes) - {area}")
        ax.set_xlim(-1.1, 1.1)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        df_sess_area = df_coef_full_active_sessions[df_coef_full_active_sessions["area"] == area]
        plot_terms = df_a["term"].tolist()
        _annotate_forest_vs_zero(ax, df_sess_area, plot_terms)
        _print_coef_ttests(df_sess_area, f"PR~behavior full-active {area}", plot_terms)
        fig.tight_layout()

        if SAVE_PLOTS:
            fig.savefig(
                RESULTS_DIR / f"suppl_covstats_with_bhvr_regression_coefficients_full_active_{area}.pdf",
                format="pdf",
                bbox_inches="tight",
            )
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)

    df_scmrn = df_coef_full_active[df_coef_full_active["area"] == "SCMRN"].copy()
    if not df_scmrn.empty:
        plot_forest_negative_xaxis_cut(
            df_scmrn,
            df_coef_full_active_sessions[df_coef_full_active_sessions["area"] == "SCMRN"],
            term_labels_full_active,
            "suppl_covstats_with_bhvr_regression_coefficients_full_active_SCMRN_xaxis_cut.pdf",
            "Regression coefficients (full active slopes) - SCMRN",
            color="C1",
        )

# R^2 distribution across sessions by area
if not df_r2_sessions.empty:
    df_r2_plot = df_r2_sessions[df_r2_sessions["area"].isin(area_order)].copy()
    df_r2_plot["area"] = pd.Categorical(df_r2_plot["area"], categories=area_order, ordered=True)

    g_r2 = sns.catplot(
        data=df_r2_plot,
        x="area",
        y="r2",
        kind="box",
        order=area_order,
        height=4.0,
        aspect=1.6,
    )
    g_r2.set_axis_labels("Area", "R^2")
    g_r2.fig.suptitle("Model R^2 distribution across sessions by area", y=1.02)
    g_r2.ax.set_ylim(0, 1.0)

    if SAVE_PLOTS:
        g_r2.figure.savefig(
            RESULTS_DIR / "suppl_covstats_with_bhvr_regression_r2_by_area.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(g_r2.figure)

if SAVE_PLOTS and not df_coef.empty:
    df_coef.to_csv(RESULTS_DIR / "suppl_covstats_with_bhvr_regression_coefficients.csv", index=False)
if SAVE_PLOTS and not df_coef_full_active.empty:
    df_coef_full_active.to_csv(
        RESULTS_DIR / "suppl_covstats_with_bhvr_regression_coefficients_full_active.csv", index=False
    )
if SAVE_PLOTS and not df_r2.empty:
    df_r2.to_csv(RESULTS_DIR / "suppl_covstats_with_bhvr_regression_r2.csv", index=False)
if SAVE_PLOTS and not df_vif_sessions.empty:
    df_vif_sessions.to_csv(
        RESULTS_DIR / "suppl_covstats_with_bhvr_regression_vif_by_session.csv", index=False
    )
if SAVE_PLOTS and not df_problematic_sessions.empty:
    df_problematic_sessions.to_csv(
        RESULTS_DIR / "suppl_covstats_with_bhvr_regression_problematic_sessions.csv", index=False
    )

# %%
# ---------------------------------------------------------
#  DOES DIMENSIONALITY STILL DRIVE?
# ---------------------------------------------------------

stats_file = DATA_DIR / "df_stats_area.pk"
df_stats_all = pd.read_pickle(stats_file)

data_filter = (
    (df_stats_all.session_number == 1)
    & (df_stats_all.experience_level == "Familiar")
    & (df_stats_all.image_name != "omitted")
    & (df_stats_all.is_image_novel == False)
    & (df_stats_all.is_change == False)
    & (df_stats_all.area.isin(['LGd', 'VISpl', 'VISm', 'SCMRN']))
    & (df_stats_all.N >= 30)
)

df_stats = df_stats_all[data_filter].drop(
    ["session_number", "experience_level", "is_image_novel", "is_change", "image_set"],
    axis=1,
    errors="ignore",
).copy()

df_stats = df_stats.drop(
    ["speed_values", "pupil_values", "avgspikecounts",
     "mii", "mij", "sii", "sij", "ciisq", "cijsq", "TS_emp", "PR_fit", "variances", "is_peak"],
    axis=1,
    errors="ignore",
).copy()

if "subsample_id" not in df_stats.columns:
    df_stats["subsample_id"] = (
        df_stats.groupby(["session_id", "active", "image_name", "area"], dropna=False).cumcount()
    )

# Average 200-bin chunks so channel rows are (session_id, image_name) pairs.
channel_pair_keys = ["session_id", "active", "image_name", "area"]
df_stats = (
    df_stats.groupby(channel_pair_keys, as_index=False)
    .agg(
        TV_emp=("TV_emp", "mean"),
        PR_emp=("PR_emp", "mean"),
        N=("N", "first"),
    )
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

    group_cols = ["session_id", "active", "image_name"]
    role_size = df_channel.groupby(group_cols)["area_role"].transform("size")
    role_unique = df_channel.groupby(group_cols)["area_role"].transform("nunique")
    df_channel = df_channel[(role_size == 2) & (role_unique == 2)].copy()

    df_channel = (
        df_channel.pivot(
            index=group_cols,
            columns="area_role",
            values=["area", "TV_emp", "PR_emp", "N"],
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
            "TV_emp_input": "input_TV",
            "TV_emp_target": "target_TV",
            "PR_emp_input": "input_PR",
            "PR_emp_target": "target_PR",
            "N_input": "N_input",
            "N_target": "N_target",
        }
    )

    df_channel = df_channel[
        [
            "session_id",
            "active",
            "image_name",
            "input_area",
            "target_area",
            "input_TV",
            "target_TV",
            "input_PR",
            "target_PR",
            "N_input",
            "N_target",
        ]
    ].copy()

    channel_dfs[f"{input_area}_to_{target_area}"] = df_channel

n_channel_pairs = {
    name: df.drop_duplicates(subset=["session_id", "image_name"]).shape[0]
    for name, df in channel_dfs.items()
}
print("Channel MLR unique session_id × image_name pairs:")
for name, n_pairs in n_channel_pairs.items():
    print(f"  {name}: {len(channel_dfs[name])} rows ({n_pairs} pairs)")

df_channel_lgd_vispl = channel_dfs["LGd_to_VISpl"].copy()
df_channel_vispl_vism = channel_dfs["VISpl_to_VISm"].copy()
df_channel_vism_scmrn = channel_dfs["VISm_to_SCMRN"].copy()
# %%
# ---------------------------------------------------------
#  CHANNEL MLR: DRIVERS OF TARGET PR / TARGET TV
# ---------------------------------------------------------
CHANNEL_MIN_ROWS_PER_SESSION = 8
CHANNEL_VIF_FLAG_THRESHOLD = 5.0
channel_order = ["LGd_to_VISpl", "VISpl_to_VISm", "VISm_to_SCMRN"]
channel_display_labels = {
    "LGd_to_VISpl": "LGd -> VISpl",
    "VISpl_to_VISm": "VISpl -> VISm",
    "VISm_to_SCMRN": "VISm -> SCMRN",
}
CHANNEL_MODEL_TERMS = [
    "active",
    "TV_input_z",
    "PR_input_z",
    "active:TV_input_z",
    "active:PR_input_z",
]
channel_outcomes = {
    "PR_target": "target_PR",
    "TV_target": "target_TV",
}

channel_coef_records = []
channel_r2_records = []
channel_vif_records = []
channel_session_counts = []

print("Running channel MLR analysis for PR_target and TV_target.")

for channel_name, df_channel in channel_dfs.items():
    df_channel_model = df_channel.copy()
    df_channel_model["active"] = df_channel_model["active"].astype(int)

    for outcome_name, outcome_col in channel_outcomes.items():
        skipped_sessions = 0
        fitted_sessions = []

        for session_id, df_sess in df_channel_model.groupby("session_id"):
            if len(df_sess) < CHANNEL_MIN_ROWS_PER_SESSION:
                skipped_sessions += 1
                continue

            df_sess = df_sess.copy()
            df_sess["TV_input_z"] = zscore_within_group(df_sess["input_TV"], df_sess["active"])
            df_sess["PR_input_z"] = zscore_within_group(df_sess["input_PR"], df_sess["active"])
            df_sess["target_z"] = zscore(df_sess[outcome_col])

            X = pd.DataFrame(
                {
                    "active": df_sess["active"].astype(float),
                    "TV_input_z": df_sess["TV_input_z"].astype(float),
                    "PR_input_z": df_sess["PR_input_z"].astype(float),
                }
            )
            X["active:TV_input_z"] = X["active"] * X["TV_input_z"]
            X["active:PR_input_z"] = X["active"] * X["PR_input_z"]
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
                        "session_id": session_id,
                        "n_rows": len(df_sess),
                        "term": term,
                        "vif": float(vif),
                        "max_vif": float(max_vif),
                        "is_problematic": bool(is_problematic),
                    }
                )

            if smf is not None:
                formula = (
                    "target_z ~ active + TV_input_z + PR_input_z "
                    "+ active:TV_input_z + active:PR_input_z"
                )
                model = smf.ols(formula=formula, data=df_sess).fit()
                coef_map = model.params.drop(labels=["Intercept"], errors="ignore").to_dict()
                r2_value = model.rsquared
            else:
                reg = LinearRegression().fit(X, y)
                coef_map = dict(zip(X.columns, reg.coef_))
                r2_value = reg.score(X, y)

            channel_r2_records.append(
                {
                    "channel": channel_name,
                    "outcome": outcome_name,
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
        df_channel_coef_sessions.groupby(["channel", "outcome", "term"], as_index=False)
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
        columns=["channel", "outcome", "term", "coef_mean", "coef_std", "coef_sem", "n_sessions"]
    )

if not df_channel_coef_sessions.empty:
    df_channel_terms = (
        df_channel_coef_sessions.pivot_table(
            index=["channel", "outcome", "session_id"],
            columns="term",
            values="coef",
            aggfunc="first",
        )
        .reset_index()
    )
    required_channel_terms = {
        "active",
        "TV_input_z",
        "active:TV_input_z",
        "PR_input_z",
        "active:PR_input_z",
    }
    if required_channel_terms.issubset(set(df_channel_terms.columns)):
        df_channel_terms["TV_input_active_full_z"] = (
            df_channel_terms["TV_input_z"] + df_channel_terms["active:TV_input_z"]
        )
        df_channel_terms["PR_input_active_full_z"] = (
            df_channel_terms["PR_input_z"] + df_channel_terms["active:PR_input_z"]
        )
        df_channel_coef_full_active_sessions = df_channel_terms[
            [
                "channel",
                "outcome",
                "session_id",
                "active",
                "TV_input_z",
                "TV_input_active_full_z",
                "PR_input_z",
                "PR_input_active_full_z",
            ]
        ].melt(
            id_vars=["channel", "outcome", "session_id"],
            var_name="term",
            value_name="coef",
        )
        df_channel_coef_full_active = (
            df_channel_coef_full_active_sessions.groupby(["channel", "outcome", "term"], as_index=False)
            .agg(
                coef_mean=("coef", "mean"),
                coef_std=("coef", "std"),
                n_sessions=("session_id", "nunique"),
            )
        )
        df_channel_coef_full_active["coef_sem"] = df_channel_coef_full_active["coef_std"] / np.sqrt(
            df_channel_coef_full_active["n_sessions"].clip(lower=1)
        )
    else:
        df_channel_coef_full_active_sessions = pd.DataFrame(
            columns=["channel", "outcome", "session_id", "term", "coef"]
        )
        df_channel_coef_full_active = pd.DataFrame(
            columns=["channel", "outcome", "term", "coef_mean", "coef_std", "coef_sem", "n_sessions"]
        )
else:
    df_channel_coef_full_active_sessions = pd.DataFrame(
        columns=["channel", "outcome", "session_id", "term", "coef"]
    )
    df_channel_coef_full_active = pd.DataFrame(
        columns=["channel", "outcome", "term", "coef_mean", "coef_std", "coef_sem", "n_sessions"]
    )

if not df_channel_r2_sessions.empty:
    df_channel_r2 = (
        df_channel_r2_sessions.groupby(["channel", "outcome"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            n_sessions=("session_id", "nunique"),
        )
    )
    df_channel_r2["r2_sem"] = df_channel_r2["r2_std"] / np.sqrt(df_channel_r2["n_sessions"].clip(lower=1))
else:
    df_channel_r2 = pd.DataFrame(columns=["channel", "outcome", "r2_mean", "r2_std", "r2_sem", "n_sessions"])

if not df_channel_vif_sessions.empty:
    df_channel_problematic_sessions = (
        df_channel_vif_sessions[df_channel_vif_sessions["is_problematic"]]
        .groupby(["channel", "outcome", "session_id"], as_index=False)
        .agg(
            n_rows=("n_rows", "first"),
            max_vif=("max_vif", "first"),
        )
    )
else:
    df_channel_problematic_sessions = pd.DataFrame(
        columns=["channel", "outcome", "session_id", "n_rows", "max_vif"]
    )

for channel_name in channel_dfs.keys():
    for outcome_name in channel_outcomes.keys():
        df_fit = df_channel_r2_sessions[
            (df_channel_r2_sessions["channel"] == channel_name)
            & (df_channel_r2_sessions["outcome"] == outcome_name)
        ].copy()
        df_flag = df_channel_problematic_sessions[
            (df_channel_problematic_sessions["channel"] == channel_name)
            & (df_channel_problematic_sessions["outcome"] == outcome_name)
        ].copy()

        initial_unique_session_ids = sorted(df_fit["session_id"].unique().tolist())
        flagged_unique_session_ids = sorted(df_flag["session_id"].unique().tolist())
        remaining_unique_session_ids = sorted(set(initial_unique_session_ids) - set(flagged_unique_session_ids))

        print(
            f"[{channel_name} | {outcome_name}] Initial fitted sessions available: "
            f"{len(df_fit)} area-session fits ({len(initial_unique_session_ids)} unique session IDs)."
        )
        print(
            f"[{channel_name} | {outcome_name}] Unique session_id before VIF filtering "
            f"({len(initial_unique_session_ids)}): {initial_unique_session_ids}"
        )
        if not df_flag.empty:
            print(f"[{channel_name} | {outcome_name}] Flagged sessions (area-session fits):")
            print(df_flag.sort_values("session_id").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            print(
                f"[{channel_name} | {outcome_name}] Flagged {len(df_flag)} area-session fits "
                f"with max VIF >= {CHANNEL_VIF_FLAG_THRESHOLD} (or non-finite VIF)."
            )
            print(
                f"[{channel_name} | {outcome_name}] Remaining sessions after flagging: "
                f"{len(df_fit) - len(df_flag)} area-session fits."
            )
            print(
                f"[{channel_name} | {outcome_name}] Unique session_id after VIF filtering "
                f"({len(remaining_unique_session_ids)}): {remaining_unique_session_ids}"
            )
        else:
            print(f"[{channel_name} | {outcome_name}] Flagged sessions (area-session fits): none")
            print(
                f"[{channel_name} | {outcome_name}] Remaining sessions after flagging: "
                f"{len(df_fit)} area-session fits."
            )
            print(
                f"[{channel_name} | {outcome_name}] Unique session_id after VIF filtering "
                f"({len(initial_unique_session_ids)}): {initial_unique_session_ids}"
            )

if not df_channel_coef.empty:
    channel_term_order = [
        "active",
        "TV_input_z",
        "active:TV_input_z",
        "PR_input_z",
        "active:PR_input_z",
    ]
    channel_term_labels = {
        "active": "active",
        "TV_input_z": "input TV",
        "active:TV_input_z": "input TV x active",
        "PR_input_z": "input PR",
        "active:PR_input_z": "input PR x active",
    }
    df_channel_coef_plot = df_channel_coef.copy()
    df_channel_coef_plot["term"] = pd.Categorical(
        df_channel_coef_plot["term"], categories=channel_term_order, ordered=True
    )

    for channel_name in channel_order:
        for outcome_name in channel_outcomes.keys():
            df_plot = df_channel_coef_plot[
                (df_channel_coef_plot["channel"] == channel_name)
                & (df_channel_coef_plot["outcome"] == outcome_name)
            ].sort_values("term")
            if df_plot.empty:
                continue

            fig, ax = plt.subplots(figsize=(5.4, 4.0))
            y = np.arange(len(df_plot))
            ax.errorbar(
                df_plot["coef_mean"],
                y,
                xerr=df_plot["coef_sem"],
                fmt="o",
                color="C0",
                capsize=3,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels([channel_term_labels.get(t, t) for t in df_plot["term"]])
            ax.set_xlabel(f"Coefficient (z-scored {outcome_name})")
            ax.set_ylabel("Predictor")
            ax.set_title(f"{channel_display_labels[channel_name]} - {outcome_name}")
            ax.set_xlim(-1.1, 1.1)
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            df_sess_ch = df_channel_coef_sessions[
                (df_channel_coef_sessions["channel"] == channel_name)
                & (df_channel_coef_sessions["outcome"] == outcome_name)
            ]
            plot_terms = df_plot["term"].tolist()
            _annotate_forest_vs_zero(ax, df_sess_ch, plot_terms)
            _print_coef_ttests(
                df_sess_ch,
                f"channel {channel_name} {outcome_name}",
                plot_terms,
            )
            fig.tight_layout()

            if SAVE_PLOTS:
                fig.savefig(
                    RESULTS_DIR / f"suppl_covstats_with_bhvr_channel_regression_coefficients_{channel_name}_{outcome_name}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )
            if SHOW_PLOTS:
                plt.show()
            else:
                plt.close(fig)

if not df_channel_coef_full_active.empty:
    channel_term_order_full_active = [
        "active",
        "TV_input_z",
        "TV_input_active_full_z",
        "PR_input_z",
        "PR_input_active_full_z",
    ]
    channel_term_labels_full_active = {
        "active": "active",
        "TV_input_z": "input TV (passive)",
        "TV_input_active_full_z": "input TV (active full)",
        "PR_input_z": "input PR (passive)",
        "PR_input_active_full_z": "input PR (active full)",
    }
    df_channel_coef_full_active_plot = df_channel_coef_full_active.copy()
    df_channel_coef_full_active_plot["term"] = pd.Categorical(
        df_channel_coef_full_active_plot["term"],
        categories=channel_term_order_full_active,
        ordered=True,
    )

    for channel_name in channel_order:
        for outcome_name in channel_outcomes.keys():
            df_plot = df_channel_coef_full_active_plot[
                (df_channel_coef_full_active_plot["channel"] == channel_name)
                & (df_channel_coef_full_active_plot["outcome"] == outcome_name)
            ].sort_values("term")
            if df_plot.empty:
                continue

            fig, ax = plt.subplots(figsize=(5.8, 4.0))
            y = np.arange(len(df_plot))
            ax.errorbar(
                df_plot["coef_mean"],
                y,
                xerr=df_plot["coef_sem"],
                fmt="o",
                color="C1",
                capsize=3,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels([channel_term_labels_full_active.get(t, t) for t in df_plot["term"]])
            ax.set_xlabel(f"Coefficient (z-scored {outcome_name})")
            ax.set_ylabel("Predictor")
            ax.set_title(f"{channel_display_labels[channel_name]} - {outcome_name} (full active slopes)")
            ax.set_xlim(-1.1, 1.1)
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            df_sess_ch = df_channel_coef_full_active_sessions[
                (df_channel_coef_full_active_sessions["channel"] == channel_name)
                & (df_channel_coef_full_active_sessions["outcome"] == outcome_name)
            ]
            plot_terms = df_plot["term"].tolist()
            _annotate_forest_vs_zero(ax, df_sess_ch, plot_terms)
            _print_coef_ttests(
                df_sess_ch,
                f"channel full-active {channel_name} {outcome_name}",
                plot_terms,
            )
            fig.tight_layout()

            if SAVE_PLOTS:
                fig.savefig(
                    RESULTS_DIR
                    / f"suppl_covstats_with_bhvr_channel_regression_coefficients_full_active_{channel_name}_{outcome_name}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )
            if SHOW_PLOTS:
                plt.show()
            else:
                plt.close(fig)

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
        kind="box",
        order=[channel_display_labels[ch] for ch in channel_order],
        sharey=True,
        height=4.0,
        aspect=1.2,
    )
    g_channel_r2.set_axis_labels("Channel", "R^2")
    g_channel_r2.fig.suptitle("Channel model R^2 distribution by outcome", y=1.03)
    for ax in g_channel_r2.axes.flat:
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=20)

    if SAVE_PLOTS:
        g_channel_r2.figure.savefig(
            RESULTS_DIR / "suppl_covstats_with_bhvr_channel_regression_r2_by_channel.pdf",
            format="pdf",
            bbox_inches="tight",
        )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(g_channel_r2.figure)

if SAVE_PLOTS and not df_channel_coef.empty:
    df_channel_coef.to_csv(RESULTS_DIR / "suppl_covstats_with_bhvr_channel_regression_coefficients.csv", index=False)
if SAVE_PLOTS and not df_channel_coef_full_active.empty:
    df_channel_coef_full_active.to_csv(
        RESULTS_DIR / "suppl_covstats_with_bhvr_channel_regression_coefficients_full_active.csv", index=False
    )
if SAVE_PLOTS and not df_channel_r2.empty:
    df_channel_r2.to_csv(RESULTS_DIR / "suppl_covstats_with_bhvr_channel_regression_r2.csv", index=False)
if SAVE_PLOTS and not df_channel_vif_sessions.empty:
    df_channel_vif_sessions.to_csv(
        RESULTS_DIR / "suppl_covstats_with_bhvr_channel_regression_vif_by_session.csv", index=False
    )
if SAVE_PLOTS and not df_channel_problematic_sessions.empty:
    df_channel_problematic_sessions.to_csv(
        RESULTS_DIR / "suppl_covstats_with_bhvr_channel_regression_problematic_sessions.csv", index=False
    )
# %%