# %%
"""Plots for covariance stats split by unengagement and speed/pupil terciles,
plus session-level MLR of PR on behavior and channel MLR.

Reads data/suppl_stats_*.pk and df_stats_area.pk.
Writes figures to results/suppl_analysis_behavior/.
"""
import sys
from pathlib import Path
import importlib
import pickle

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

if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

import config

DATA_DIR = config.DATA_DIR
RESULTS_DIR = config.results_dir("suppl_analysis_behavior")
SAVE_PLOTS = True
SHOW_PLOTS = True
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

area_order = ['LGd', 'VISpl', 'VISm', 'SCMRN']
unengagement_area_order = ['VISp', 'VISrl', 'VISl', 'VISal', 'VISpm', 'VISam']
tercile_order = ['low', 'mid', 'high']
active_order = [True, False]
tercile_pair_keys = ["session_id", "image_name", "active", "area", "tercile"]


def read_pickle(path):
    """Load a DataFrame, including pandas 3 string-column pickles."""
    try:
        return pd.read_pickle(path)
    except NotImplementedError as exc:
        if "StringDtype" not in str(exc):
            raise
        return _read_pickle_pandas3_strings(path)


def _read_pickle_pandas3_strings(path):
    """Rebuild pandas 3 StringArrays, whose pickle state is (dtype, ndarray).

    pandas 2.3 NDArrayBacked.__setstate__ only accepts a 3-tuple, so those
    columns raise NotImplementedError under pd.read_pickle.
    """
    from pandas._libs.arrays import NDArrayBacked
    from pandas.core.arrays.string_ import StringArray

    class _CompatStringArray(StringArray):
        def __new__(cls):
            return NDArrayBacked.__new__(cls)

        def __setstate__(self, state):
            if not (isinstance(state, tuple) and len(state) == 2):
                raise NotImplementedError(state)
            left, right = state
            data = right if isinstance(right, np.ndarray) else left
            rebuilt = StringArray(np.asarray(data, dtype=object))
            NDArrayBacked.__init__(self, rebuilt._ndarray, rebuilt.dtype)

    def _unpickle_ndarray_backed(cls, checksum, state):
        if getattr(cls, "__name__", "") == "StringArray":
            obj = _CompatStringArray.__new__(_CompatStringArray)
            if state is not None:
                obj.__setstate__(state)
            return obj
        from pandas._libs import arrays as arrays_mod
        return arrays_mod.__pyx_unpickle_NDArrayBacked(cls, checksum, state)

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "__pyx_unpickle_NDArrayBacked":
                return _unpickle_ndarray_backed
            return super().find_class(module, name)

    with open(path, "rb") as handle:
        return _Unpickler(handle).load()


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
AREA_COLORS = {
    area: AREA_PALETTE[i % len(AREA_PALETTE)]
    for i, area in enumerate(unengagement_area_order)
}


def _ratio_box_data(df_metric, areas):
    box_data = []
    box_colors = []
    tick_labels = []
    for area in areas:
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


def plot_filled_ratio_boxplots(df_ratios, metric, ylabel, filename, areas):
    """Filled boxplots (vis_circuit_metrics supplementary style) of a ratio vs area."""
    df_metric = df_ratios[df_ratios["metric"] == metric].copy()
    if df_metric.empty:
        return
    box_data, box_colors, tick_labels = _ratio_box_data(df_metric, areas)
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


def plot_empty_ratio_box_strip(df_ratios, metric, ylabel, filename, areas):
    """Empty colored boxplots plus rasterized strip points (one per session_id, image_name)."""
    df_metric = df_ratios[df_ratios["metric"] == metric].copy()
    if df_metric.empty:
        return
    box_data, box_colors, tick_labels = _ratio_box_data(df_metric, areas)
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
df_bhvr_engaged = read_pickle(unengagement_file)

df_bhvr_engaged = df_bhvr_engaged[df_bhvr_engaged.session_has_unengaged_state==True]
df_bhvr_engaged["N"] = df_bhvr_engaged["avgspikecounts"].apply(lambda x: np.asarray(x, dtype=float).ravel().size)
df_bhvr_engaged = df_bhvr_engaged[df_bhvr_engaged["N"] >= 30].copy()
df_bhvr_engaged = df_bhvr_engaged[df_bhvr_engaged["area"].isin(unengagement_area_order)].copy()

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
        df_uneng_ratios["area"], categories=unengagement_area_order, ordered=True
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
            unengagement_area_order,
        )
        plot_empty_ratio_box_strip(
            df_uneng_ratios,
            metric,
            metric_ylabels[metric],
            f"suppl_covstats_with_bhvr_unengagement_{metric}_ratio_box_strip.pdf",
            unengagement_area_order,
        )
# %%
# --------------------------------------------------------------
#  PUPIL SIZE DOES NOT AFFECT METRIC TRENDS ACROSS ACTIVE BLOCKS
# --------------------------------------------------------------
pupil_file = DATA_DIR / "suppl_stats_pupil_terciles.pk"
df_bhvr_pupil = read_pickle(pupil_file)
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
df_bhvr_speed = read_pickle(speed_file)
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
df_stats_all = read_pickle(stats_file)

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
        # Passive slope is the main effect. Full active slope is main effect + interaction.
        df_terms["speed_active_full_z"] = df_terms["speed_mean_z"] + df_terms["active:speed_mean_z"]
        df_terms["pupil_active_full_z"] = df_terms["pupil_mean_z"] + df_terms["active:pupil_mean_z"]
        df_coef_full_sessions = df_terms[
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
        df_coef_full = (
            df_coef_full_sessions.groupby(["area", "term"], as_index=False)
            .agg(
                coef_mean=("coef", "mean"),
                coef_std=("coef", "std"),
                n_sessions=("session_id", "nunique"),
            )
        )
        df_coef_full["coef_sem"] = df_coef_full["coef_std"] / np.sqrt(
            df_coef_full["n_sessions"].clip(lower=1)
        )
    else:
        df_coef_full_sessions = pd.DataFrame(columns=["area", "session_id", "term", "coef"])
        df_coef_full = pd.DataFrame(
            columns=["area", "term", "coef_mean", "coef_sem", "n_sessions"]
        )
else:
    df_coef_full_sessions = pd.DataFrame(columns=["area", "session_id", "term", "coef"])
    df_coef_full = pd.DataFrame(columns=["area", "term", "coef_mean", "coef_sem", "n_sessions"])

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

# Full active and full passive slopes, with active kept as a predictor.
if not df_coef_full.empty:
    term_order_full = [
        "active",
        "speed_mean_z",
        "speed_active_full_z",
        "pupil_mean_z",
        "pupil_active_full_z",
    ]
    df_coef_full["term"] = pd.Categorical(
        df_coef_full["term"], categories=term_order_full, ordered=True
    )
    term_labels_full = {
        "active": "active",
        "speed_mean_z": "speed (passive)",
        "speed_active_full_z": "speed (active full)",
        "pupil_mean_z": "pupil (passive)",
        "pupil_active_full_z": "pupil (active full)",
    }

    for area in area_order:
        df_a = df_coef_full[df_coef_full["area"] == area].sort_values("term")
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
        ax.set_yticklabels([term_labels_full.get(t, t) for t in df_a["term"]])
        ax.set_xlabel("Coefficient (z-scored PR_emp)")
        ax.set_ylabel("Predictor")
        ax.set_title(f"Regression coefficients (full active slopes) - {area}")
        ax.set_xlim(-1.1, 1.1)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        df_sess_area = df_coef_full_sessions[df_coef_full_sessions["area"] == area]
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

    df_scmrn = df_coef_full[df_coef_full["area"] == "SCMRN"].copy()
    if not df_scmrn.empty:
        plot_forest_negative_xaxis_cut(
            df_scmrn,
            df_coef_full_sessions[df_coef_full_sessions["area"] == "SCMRN"],
            term_labels_full,
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

if SAVE_PLOTS and not df_coef_full.empty:
    df_coef_full.to_csv(
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