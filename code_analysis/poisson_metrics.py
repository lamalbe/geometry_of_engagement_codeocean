# %%
"""
Analysis of area-level Poisson / sparsity statistics (df_poisson_stats).

Loads full and subsampled dataframes from clean_scripts/data
(df_poisson_stats.pk, df_poisson_stats_subsampled.pk). Derives Treves–Rolls
sparsity of mean rates (TR_emp), of spike-count variances
(spikecount_var_sparsity), Fano-like sparsity (a_sparsity), and mean network
rate (R_emp). Diagnostic plots use the non-subsampled table. Ratio boxplots
average those metrics over subsample_id (200-bin chunks), then take
active/passive ratios, matching vis_circuit_metrics.py. Figures go to
clean_scripts/results.

Sparsity uses (mean x)^2 / mean(x^2), equivalent to (sum x)^2 / sum(x^2) after
normalizing by the number of neurons.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

BINSIZE = 0.05  # s; under Poisson, mean count variance ≈ R_emp (Hz) * BINSIZE
AREA_ORDER = ["LGd", "VISpl", "VISm", "SCMRN"]
POISSON_AREA_ORDER = ["LGd", "VISpl", "VISm", "SCMRN"]
MULTI_AREA_SESSION_AREAS = ["LGd", "VISpl", "VISm", "SCMRN"]
VISUAL_CORTEX_AREAS = {"VISpl", "VISm"}
POISSON_SESSION_PREFERRED = "1118327332"
MIN_NEURONS_PER_AREA = 70
FANO_MIN_NEURONS = 50
N_EXAMPLE_NEURON_PLOTS = 20
EXAMPLE_NEURON_RNG_SEED = 42

SAVE_PLOTS = True
SAVE_DPI = 300

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

DATA_DIR = _SCRIPT_DIR.parent / "data"
RESULTS_DIR = _SCRIPT_DIR.parent / "results"

# %%
# hue='active' colors by area — use palette=palette_for_area(area) in seaborn
palette_map = {
    "LGd": {True: "#D679AD", False: "#E4B9D1"},
    "VISpl": {True: "#53BBF7", False: "#A7D8F4"},
    "VISm": {True: "#0079BC", False: "#7AB6D7"},
    "SCMRN": {True: "#F0A600", False: "#F2CD7A"},
}
CIRCUIT_BOX_COLORS = {area: palette_map[area][True] for area in AREA_ORDER}
VISP_METRIC_COLORS = {
    "a_sparsity_ratio": "#53BBF7",
    "R_emp_ratio": "#0079BC",
    "gain_power_law": "#D679AD",
    "gain_exponential": "#F0A600",
}
VISP_METRIC_LABELS = {
    "a_sparsity_ratio": "a_sparsity",
    "R_emp_ratio": "mean activity",
    "gain_power_law": "gain power-law",
    "gain_exponential": "gain exponential",
}
VISP_METRIC_ORDER = [
    "a_sparsity_ratio",
    "R_emp_ratio",
    "gain_power_law",
    "gain_exponential",
]
VISP_METRIC_BOX_COLORS = {
    VISP_METRIC_LABELS[m]: VISP_METRIC_COLORS[m] for m in VISP_METRIC_ORDER
}


def palette_for_area(area):
    """Return {True: color, False: color} for sns.*(..., hue='active', palette=...)."""
    return palette_map.get(area, {True: "#333333", False: "#999999"})


def get_asterisk(p_value):
    """Convert p-value to asterisk notation (same thresholds as MLR_analysis_raw)."""
    if np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def areas_present(df, area_col="area", order=AREA_ORDER):
    """Area labels present in df, in canonical order."""
    present = set(df[area_col].dropna())
    return [a for a in order if a in present]


def areas_present_poisson(df, area_col="area"):
    """Area order for Poisson Part A / Part B plots."""
    return areas_present(df, area_col=area_col, order=POISSON_AREA_ORDER)


def _rasterize_scatter_collections(fig):
    """Rasterize scatter/strip point layers only (keeps lines/text vector)."""
    for ax in fig.axes:
        for coll in ax.collections:
            if hasattr(coll, "set_rasterized"):
                coll.set_rasterized(True)


def save_plot(fig, filename, *, rasterize_scatter=False):
    """Save figure to clean_scripts/results when SAVE_PLOTS is True."""
    if not SAVE_PLOTS:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if rasterize_scatter:
        _rasterize_scatter_collections(fig)
    fig.savefig(RESULTS_DIR / filename, dpi=SAVE_DPI, bbox_inches="tight")


def add_poisson_derived_metrics(df):
    """Add TR_emp, a_sparsity, R_emp, PR/N, RHS, and angular coefficients."""
    out = df.copy()
    out["TR_emp"] = out.apply(
        lambda row: (np.mean(row["mean_firing_rates"]) ** 2)
        / np.mean(row["mean_firing_rates"] ** 2),
        axis=1,
    )
    out["spikecount_var_sparsity"] = out.apply(
        lambda row: (np.mean(row["spikecount_variances"]) ** 2)
        / np.mean(row["spikecount_variances"] ** 2),
        axis=1,
    )
    out["a_sparsity"] = out.apply(
        lambda row: (
            np.nan
            if np.all(row["mean_firing_rates"] == 0)
            else (
                lambda a: (np.mean(a) ** 2) / np.mean(a**2)
            )(
                np.divide(
                    row["spikecount_variances"],
                    row["mean_firing_rates"],
                    out=np.zeros_like(row["mean_firing_rates"], dtype=float),
                    where=row["mean_firing_rates"] > 0,
                )
            )
        ),
        axis=1,
    )
    out["R_emp"] = out.apply(
        lambda row: np.mean(row["mean_firing_rates"]),
        axis=1,
    )
    out["PR_emp"] = out["PR_emp"] / out["N_neurons"]
    out["RHS"] = out["TR_emp"] * out["a_sparsity"]
    out["var_sparsity_ratio"] = np.divide(
        out["spikecount_var_sparsity"],
        out["RHS"],
        out=np.full(len(out), np.nan, dtype=float),
        where=np.isfinite(out["RHS"].to_numpy()) & (out["RHS"].to_numpy() != 0),
    )
    out["tian_angular_coefficient"] = out["PR_emp"] / out["TR_emp"]
    out["angular_coefficient"] = out["PR_emp"] / (out["TR_emp"] * out["a_sparsity"])
    return out


def _y_offset_for_annotations(y_series, frac=0.03):
    y = y_series.dropna()
    if y.empty:
        return 0.1
    span = y.max() - y.min()
    return frac * span if span > 0 else 0.1


def _annotate_two_sample_active_passive(ax, data, y_col, areas, area_col="area"):
    """Black asterisk between active/passive (Welch two-sample t-test per area)."""
    y_offset = _y_offset_for_annotations(data[y_col])
    for i, area in enumerate(areas):
        sub = data[data[area_col] == area]
        active = sub.loc[sub["active"] == True, y_col].dropna()
        passive = sub.loc[sub["active"] == False, y_col].dropna()
        p = np.nan
        if len(active) > 1 and len(passive) > 1:
            _, p = stats.ttest_ind(active, passive, equal_var=False)
        ast = get_asterisk(p)
        if not ast:
            continue
        active_y = active.quantile(0.75) if len(active) else 0
        passive_y = passive.quantile(0.75) if len(passive) else 0
        mid_y = max(active_y, passive_y) + y_offset
        ax.text(
            i, mid_y, ast,
            ha="center", va="bottom", color="black",
            fontsize=10, fontweight="bold", clip_on=True,
        )


def _annotate_one_sample_vs_mu(ax, data, y_col, categories, mu=1.0, category_col="area"):
    """Black asterisk above each category (two-sided one-sample t-test vs mu)."""
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
    """Filled boxes with black edges (vis_circuit by-image boxplot style)."""
    for patch, color in zip(bp["boxes"], box_colors):
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


def _ratio_ylim(vals, pad_frac=0.1):
    vals = pd.Series(vals).dropna()
    if vals.empty:
        return 0.0, 2.0
    y_min, y_max = float(vals.min()), float(vals.max())
    y_min = min(y_min, 1.0)
    y_max = max(y_max, 1.0)
    span = y_max - y_min
    pad = pad_frac * span if span > 0 else 0.1
    return y_min - pad, y_max + pad


def plot_filled_ratio_boxplot(
    df,
    y_col,
    categories,
    colors,
    xlabel,
    ylabel,
    title,
    filename,
    category_col="area",
    figsize=(8, 5),
):
    """One box per category; observations are rows of *df* (session × image)."""
    box_data = []
    box_colors = []
    tick_labels = []
    for category in categories:
        vals = df.loc[df[category_col] == category, y_col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        box_data.append(vals)
        box_colors.append(colors.get(category, "#888888") if isinstance(colors, dict) else colors[len(tick_labels)])
        tick_labels.append(category)
    if not box_data:
        return None

    y_lim = _ratio_ylim(np.concatenate(box_data))
    fig, ax = plt.subplots(figsize=figsize)
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_lim)
    ax.set_xlim(-0.5, len(box_data) - 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8, alpha=0.5, zorder=0)
    _annotate_one_sample_vs_mu(
        ax, df, y_col, tick_labels, mu=1.0, category_col=category_col,
    )
    _pad_ylim_for_asterisks(ax)
    ax.set_title(title)
    fig.tight_layout()
    save_plot(fig, filename)
    return fig


def catplot_area_stats(data, y, kind, test, marker=None, mu=1.0, area_order=None, **catplot_kw):
    """
    Catplot with fixed area order and significance asterisks.

    test: 'two_sample' (active vs passive) or 'one_sample' (vs mu, default 1).
    area_order: optional override of AREA_ORDER for this plot only.
    """
    order = AREA_ORDER if area_order is None else list(area_order)
    plot_df = data[data["area"].isin(order)].copy()
    areas = areas_present(plot_df, order=order)
    plot_df["area"] = pd.Categorical(plot_df["area"], categories=areas, ordered=True)

    base_kw = dict(
        data=plot_df, x="area", y=y, order=areas, kind=kind, **catplot_kw,
    )
    if marker is not None:
        base_kw["marker"] = marker

    if test == "two_sample":
        g = sns.catplot(hue="active", **base_kw)
        _annotate_two_sample_active_passive(g.axes.flat[0], plot_df, y, areas)
    elif test == "one_sample":
        g = sns.catplot(**base_kw)
        _annotate_one_sample_vs_mu(g.axes.flat[0], plot_df, y, areas, mu=mu)
        for ax in g.axes.flat:
            ax.axhline(mu, color="black", linestyle="--", linewidth=0.8, zorder=0)
    else:
        raise ValueError(f"Unknown test: {test}")

    ax = g.axes.flat[0]
    ymin, ymax = ax.get_ylim()
    pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.1
    ax.set_ylim(ymin, ymax + pad)
    return g


def sessions_with_all_areas(df, areas, area_col="area", session_col="session_id"):
    """Session IDs that contain every area in *areas*."""
    required = set(areas)
    session_areas = df.groupby(session_col)[area_col].apply(set)
    return session_areas[session_areas.apply(required.issubset)].index.tolist()


def sessions_with_min_neurons(df, areas, min_neurons):
    """Sessions that contain every area and have at least *min_neurons* in each."""
    complete = sessions_with_all_areas(df, areas)
    if not complete:
        return []
    ntab = (
        df[df["area"].isin(areas)]
        .groupby(["session_id", "area"])["N_neurons"]
        .first()
        .unstack()
    )
    ntab.index = ntab.index.map(str)
    out = []
    for sid in complete:
        key = str(sid)
        if key not in ntab.index:
            continue
        min_n = float(ntab.loc[key, list(areas)].min())
        if min_n >= min_neurons:
            out.append(sid)
    return out


def resolve_poisson_session(
    df,
    areas,
    preferred=POISSON_SESSION_PREFERRED,
    min_neurons=MIN_NEURONS_PER_AREA,
):
    """Session with all *areas* and enough neurons per area."""
    complete = sessions_with_all_areas(df, areas)
    if not complete:
        raise ValueError(f"No sessions contain all of {list(areas)}")

    ntab = (
        df[df["area"].isin(areas)]
        .groupby(["session_id", "area"])["N_neurons"]
        .first()
        .unstack()
    )
    ntab.index = ntab.index.map(str)

    scored = []
    for sid in complete:
        key = str(sid)
        if key not in ntab.index:
            continue
        min_n = float(ntab.loc[key, list(areas)].min())
        scored.append((min_n, sid))
    if not scored:
        raise ValueError(f"No complete sessions with N_neurons for {list(areas)}")

    scored.sort(key=lambda x: x[0], reverse=True)
    qualifying = [(n, s) for n, s in scored if n >= min_neurons]
    pool = qualifying if qualifying else scored[:1]
    pool_str = [str(s) for _, s in pool]
    if str(preferred) in pool_str:
        return pool[pool_str.index(str(preferred))][1]
    print(
        f"Poisson session {preferred} skipped "
        f"(min N_neurons < {min_neurons}); using {pool[0][1]} (min N={pool[0][0]:.0f})"
    )
    return pool[0][1]


def firing_rates_long(df):
    """One row per neuron rate; columns session_id, area, active, mean_firing_rate."""
    long_df = (
        df[["session_id", "area", "active", "mean_firing_rates"]]
        .explode("mean_firing_rates")
        .rename(columns={"mean_firing_rates": "mean_firing_rate"})
    )
    long_df["mean_firing_rate"] = pd.to_numeric(
        long_df["mean_firing_rate"], errors="coerce",
    )
    return long_df.dropna(subset=["mean_firing_rate"])


# %%
df_poisson_stats = pd.read_pickle(DATA_DIR / "df_poisson_stats.pk")
df_poisson_stats_subsampled = pd.read_pickle(DATA_DIR / "df_poisson_stats_subsampled.pk")

drop_cols = ["image_set", "session_number", "experience_level"]
df_poisson_stats = df_poisson_stats.drop(columns=drop_cols, errors="ignore")
df_poisson_stats_subsampled = df_poisson_stats_subsampled.drop(
    columns=drop_cols, errors="ignore"
)

df_poisson_stats = df_poisson_stats[df_poisson_stats.N_neurons >= 45]
# %%
df_poisson_stats = add_poisson_derived_metrics(df_poisson_stats)

# Ratio boxplots: derived metrics on 200-bin chunks, then mean over subsample_id
# (same order as vis_circuit_metrics.py).
df_poisson_sub = df_poisson_stats_subsampled[
    df_poisson_stats_subsampled.N_neurons >= 45
].copy()
df_poisson_sub = add_poisson_derived_metrics(df_poisson_sub)

group_keys = ["area", "session_id", "image_name"]
avg_group_cols = group_keys + ["active"]
ratio_metrics = [
    "PR_emp", "TR_emp", "a_sparsity", "R_emp", "RHS",
    "angular_coefficient", "tian_angular_coefficient",
]
df_poisson_sub_avg = (
    df_poisson_sub[avg_group_cols + ratio_metrics]
    .groupby(avg_group_cols, as_index=False)[ratio_metrics]
    .mean()
)

wide = (
    df_poisson_sub_avg.set_index(group_keys + ["active"])[ratio_metrics]
    .unstack("active")
)

df_stats_ap_ratio = wide.index.to_frame(index=False)
for metric in ratio_metrics:
    df_stats_ap_ratio[f"{metric}_ratio"] = (
        wide[(metric, True)] / wide[(metric, False)]
    ).values

df_stats_ap_ratio["gain_power_law"] = (
    df_stats_ap_ratio["R_emp_ratio"] ** (4 / 3)
    / df_stats_ap_ratio["TR_emp_ratio"] ** (2 / 9)
)
df_stats_ap_ratio["gain_exponential"] = (
    df_stats_ap_ratio["R_emp_ratio"] ** 2
    / df_stats_ap_ratio["TR_emp_ratio"]
)

# %%
g_cat_angular = catplot_area_stats(
    df_poisson_stats,
    y="angular_coefficient",
    kind="point",
    test="two_sample",
    marker="o",
    area_order=AREA_ORDER,
)
save_plot(g_cat_angular.fig, "catplot_angular_coefficient.pdf")

df_ang_ap = (
    df_poisson_stats[df_poisson_stats["area"].isin(AREA_ORDER)]
    .pivot_table(
        index=["session_id", "image_name", "area"],
        columns="active",
        values="angular_coefficient",
        aggfunc="mean",
    )
    .reset_index()
    .rename(columns={False: "passive", True: "active"})
)
df_ang_ap = df_ang_ap.dropna(subset=["passive", "active"])
_ang_areas = [a for a in AREA_ORDER if a in set(df_ang_ap["area"])]

g_ang_scatter = sns.FacetGrid(
    df_ang_ap,
    col="area",
    col_order=_ang_areas,
    height=3.2,
    aspect=1.0,
)
_ang_vals = pd.concat([df_ang_ap["passive"], df_ang_ap["active"]], ignore_index=True).dropna()
if _ang_vals.empty:
    _ang_low, _ang_high = 0.0, 1.0
else:
    _ang_pad = 0.05 * (_ang_vals.max() - _ang_vals.min()) if _ang_vals.max() > _ang_vals.min() else 0.1
    _ang_low = float(_ang_vals.min()) - _ang_pad
    _ang_high = float(_ang_vals.max()) + _ang_pad

for ax, area in zip(g_ang_scatter.axes.flat, _ang_areas):
    sub = df_ang_ap[df_ang_ap["area"] == area]
    sns.scatterplot(
        data=sub,
        x="passive",
        y="active",
        ax=ax,
        color=palette_for_area(area)[True],
        alpha=0.7,
        s=28,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.plot([_ang_low, _ang_high], [_ang_low, _ang_high], ls="--", c="k", lw=1, zorder=0)
    ax.set_xlim(_ang_low, _ang_high)
    ax.set_ylim(_ang_low, _ang_high)
    ax.set_xlabel("Passive")
    ax.set_ylabel("Active")
    ax.set_title(area)
    ax.set_aspect("equal", adjustable="box")

g_ang_scatter.fig.subplots_adjust(top=0.82)
g_ang_scatter.fig.suptitle(
    "Angular coefficient: active vs passive (dashed y = x)",
    y=1.02,
)
save_plot(
    g_ang_scatter.fig,
    "scatter_angular_coefficient_active_vs_passive.pdf",
    rasterize_scatter=True,
)

# sns.displot(data=df_poisson_stats, x="angular_coefficient", hue="active", kind="kde", row="area")
# g_cat_a_sparsity = catplot_area_stats(df_poisson_stats, y="a_sparsity", kind="box", test="two_sample")
# save_plot(g_cat_a_sparsity.fig, "catplot_a_sparsity.pdf")

# %%
for area_id in areas_present(df_poisson_stats):
    df_stats_area = df_poisson_stats[df_poisson_stats.area == area_id]
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_stats_area, x="RHS", y="PR_emp", hue="active", ax=ax)

    # Get the min/max of both x and y to set shared range
    x_min, x_max = df_stats_area["TR_emp"].min(), df_stats_area["TR_emp"].max()
    y_min, y_max = df_stats_area["PR_emp"].min(), df_stats_area["PR_emp"].max()
    shared_min = min(x_min, y_min)
    shared_max = max(x_max, y_max)
    # Optional: Some padding, e.g. 5% of the range
    pad = 0.05 * (shared_max - shared_min)
    shared_min -= pad
    shared_max += pad
    ax.set_xlim(shared_min, shared_max)
    ax.set_ylim(shared_min, shared_max)

    # Add y=x dashed black line
    ax.plot(
        [shared_min, shared_max],
        [shared_min, shared_max],
        linestyle="--",
        color="black",
        linewidth=1,
        label="y = x"
    )

    ax.set_title(f"Area: {area_id}")
    plt.show()

# %%
# Filled ratio boxplots: session × image after averaging subsample_id.
plot_filled_ratio_boxplot(
    df_stats_ap_ratio, "angular_coefficient_ratio", AREA_ORDER, CIRCUIT_BOX_COLORS,
    xlabel="Area",
    ylabel="Angular coefficient (active/passive)",
    title="Angular coefficient (active/passive)",
    filename="poisson_angular_coefficient_ratio_boxplot.pdf",
)

df_visp_ratio = df_stats_ap_ratio[df_stats_ap_ratio["area"] == "VISp"].copy()
visp_long = df_visp_ratio.melt(
    value_vars=VISP_METRIC_ORDER,
    var_name="metric",
    value_name="ratio",
)
visp_long["metric_label"] = visp_long["metric"].map(VISP_METRIC_LABELS)

plot_filled_ratio_boxplot(
    visp_long, "ratio", [VISP_METRIC_LABELS[m] for m in VISP_METRIC_ORDER],
    VISP_METRIC_BOX_COLORS,
    xlabel="",
    ylabel="Active / passive",
    title="VISp Poisson metric ratios (session × image)",
    filename="poisson_visp_metric_ratios_boxplot.pdf",
    category_col="metric_label",
    figsize=(8, 5),
)

plot_filled_ratio_boxplot(
    df_stats_ap_ratio, "gain_power_law", AREA_ORDER, CIRCUIT_BOX_COLORS,
    xlabel="Area",
    ylabel="Gain (power-law, active/passive)",
    title="Gain (power-law, active/passive)",
    filename="poisson_circuit_gain_power_law_boxplot.pdf",
)
plot_filled_ratio_boxplot(
    df_stats_ap_ratio, "gain_exponential", AREA_ORDER, CIRCUIT_BOX_COLORS,
    xlabel="Area",
    ylabel="Gain (exponential, active/passive)",
    title="Gain (exponential, active/passive)",
    filename="poisson_circuit_gain_exponential_boxplot.pdf",
)

# %%
# Toy: PR/(d(a)D(r)) RHS = (1 - g^2 * r_bar)^2 (active r_bar < passive r_bar => active curve above)
R_BAR_ACTIVE_TOY = 5.0
R_BAR_PASSIVE_TOY = 9.0
_g_max_toy = 1.0 / np.sqrt(R_BAR_PASSIVE_TOY)
_g_toy = np.linspace(0, _g_max_toy, 300)

fig_rhs_toy, ax_rhs_toy = plt.subplots(figsize=(5, 4))
ax_rhs_toy.plot(
    _g_toy,
    (1 - _g_toy**2 * R_BAR_ACTIVE_TOY) ** 2,
    color=palette_for_area("VISpl")[True],
    label=rf"active ($\bar{{r}}$ = {R_BAR_ACTIVE_TOY:g})",
)
ax_rhs_toy.plot(
    _g_toy,
    (1 - _g_toy**2 * R_BAR_PASSIVE_TOY) ** 2,
    color=palette_for_area("VISpl")[False],
    label=rf"passive ($\bar{{r}}$ = {R_BAR_PASSIVE_TOY:g})",
)
ax_rhs_toy.set_xlim(0, _g_max_toy)
ax_rhs_toy.set_ylim(0, 1.02)
ax_rhs_toy.set_xlabel("g")
ax_rhs_toy.set_ylabel(r"$(1 - g^2 \bar{r})^2$")
ax_rhs_toy.set_title("Angular coefficient RHS (toy)")
ax_rhs_toy.legend()
save_plot(fig_rhs_toy, "angular_coefficient_RHS_toy.pdf")

# %%
# ------------------------
# POISSON CONSISTENCY — Part A & B (complete LGd / VISpl / VISm / SCMRN session)
# ------------------------
# Trial = (active, image_name, subsample_id). Part B: per-neuron Fano over pooled trials.

POISSON_SESSION = resolve_poisson_session(
    df_poisson_stats, POISSON_AREA_ORDER,
)
print(f"Poisson session: {POISSON_SESSION}")

META_COLS = [
    "session_id",
    "area",
    "active",
    "image_name",
    "subsample_id",
    "N_neurons",
]


def mean_count_and_fano(mean_firing_rates, spikecount_variances):
    """Per-neuron mean spike count (50 ms bin) and Fano = var / mean count."""
    rates = np.asarray(mean_firing_rates, dtype=float)
    vars_ = np.asarray(spikecount_variances, dtype=float)
    mean_count = rates * BINSIZE
    fano = np.divide(vars_, mean_count, out=np.full_like(vars_, np.nan), where=mean_count > 0)
    return mean_count, vars_, fano


def expand_row_population(row):
    """One subsampled row → long table with one row per neuron."""
    mean_count, neuron_var, fano = mean_count_and_fano(
        row["mean_firing_rates"], row["spikecount_variances"]
    )
    n = mean_count.shape[0]
    out = {col: row[col] for col in META_COLS if col in row.index}
    out.update(
        {
            "neuron_idx": np.arange(n, dtype=int),
            "mean_count": mean_count,
            "neuron_var": neuron_var,
            "fano": fano,
        }
    )
    return pd.DataFrame(out)


def _rank_near_poisson_neurons(df_pop, areas, min_points_per_block=10, max_log10_fano=0.15):
    """
    Per area, Poisson-like neurons with a usable mean-count range.

    Returns {area: [( -span, |log10 Fano|, neuron_idx), ...]} sorted best-first.
    """
    ranked_by_area = {}
    for area in areas:
        sub = df_pop[df_pop["area"] == area]
        ranked = []
        fallback = []
        for neuron_idx, g in sub.groupby("neuron_idx"):
            act = g.loc[g["active"] == True]
            pas = g.loc[g["active"] == False]
            if len(act) < min_points_per_block or len(pas) < min_points_per_block:
                continue
            fano = pd.concat([act["fano"], pas["fano"]], ignore_index=True).dropna()
            fano = fano[fano > 0]
            means = pd.concat([act["mean_count"], pas["mean_count"]], ignore_index=True).dropna()
            if fano.empty or len(means) < min_points_per_block:
                continue
            span = float(means.quantile(0.9) - means.quantile(0.1))
            if span <= 0.05:
                continue
            score = float(np.mean(np.abs(np.log10(fano.to_numpy()))))
            item = (-span, score, int(neuron_idx))
            fallback.append(item)
            if score <= max_log10_fano:
                ranked.append(item)
        use = ranked if ranked else fallback
        if not use:
            print(f"Warning: no near-Poisson neuron found for {area}")
            continue
        use.sort()
        ranked_by_area[area] = use
    return ranked_by_area


def sample_example_neuron_sets(ranked_by_area, n_plots, rng_seed=EXAMPLE_NEURON_RNG_SEED):
    """Random one-neuron-per-area draws; first set is the top-ranked choice."""
    areas = [a for a in POISSON_AREA_ORDER if a in ranked_by_area]
    if not areas:
        return []
    rng = np.random.default_rng(rng_seed)
    sets = []
    seen = set()
    top = {area: ranked_by_area[area][0][2] for area in areas}
    sets.append(top)
    seen.add(tuple(top[a] for a in areas))
    pool = {area: [item[2] for item in ranked_by_area[area]] for area in areas}
    max_tries = n_plots * 50
    tries = 0
    while len(sets) < n_plots and tries < max_tries:
        tries += 1
        choice = {
            area: int(rng.choice(pool[area]))
            for area in areas
        }
        key = tuple(choice[a] for a in areas)
        if key in seen:
            continue
        seen.add(key)
        sets.append(choice)
    while len(sets) < n_plots:
        choice = {
            area: int(rng.choice(pool[area]))
            for area in areas
        }
        sets.append(choice)
    return sets


def plot_example_neurons(df_pop, chosen, areas, session_id, choice_id, row_order):
    """One mean–variance panel per area for the selected example cells."""
    example_areas = [a for a in areas if a in chosen]
    parts = []
    for area in example_areas:
        neuron_idx = chosen[area]
        sub = df_pop[
            (df_pop["area"] == area) & (df_pop["neuron_idx"] == neuron_idx)
        ]
        if sub.empty:
            print(
                f"Warning: no data for {area} neuron {neuron_idx} "
                f"in session {session_id}"
            )
            continue
        parts.append(sub)
    if not parts:
        print(f"Warning: no example neuron data for choice {choice_id}; skipping")
        return None

    df_ex = pd.concat(parts, ignore_index=True)
    plot_areas = [a for a in example_areas if a in set(df_ex["area"])]

    g = sns.FacetGrid(
        df_ex,
        col="area",
        col_order=plot_areas,
        height=3.2,
        aspect=1.0,
        sharex=False,
        sharey=False,
        margin_titles=True,
    )
    for col_i, area in enumerate(plot_areas):
        ax = g.axes.flat[col_i]
        neuron_idx = chosen[area]
        sub_n = df_ex[df_ex["area"] == area]
        pal = palette_for_area(area)
        for active in row_order:
            s = sub_n[sub_n["active"] == active]
            if s.empty:
                continue
            sns.regplot(
                data=s,
                x="mean_count",
                y="neuron_var",
                ax=ax,
                color=pal[active],
                scatter_kws={"alpha": 0.55, "s": 14},
                line_kws={"lw": 1},
                label="active" if active else "passive",
                truncate=False,
            )
        vals = pd.concat(
            [sub_n["mean_count"], sub_n["neuron_var"]], ignore_index=True
        ).dropna()
        if vals.empty:
            low, high = 0.0, 1.0
        else:
            low = max(0.0, float(vals.min()))
            high = float(vals.max())
            if high <= low:
                high = low + 0.1
        ax.plot([low, high], [low, high], ls="--", c="k", lw=1, zorder=0)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("mean count (50 ms bin)")
        ax.set_ylabel("spike count variance")
        ax.set_title(f"{area}  n={neuron_idx}", fontsize=9)
        ax.legend(fontsize=7, loc="best")

    g.fig.subplots_adjust(top=0.90)
    g.fig.suptitle(
        f"Part A: example neurons choice {choice_id} "
        f"(session {session_id}; Poisson: dashed y = x)",
        y=1.02,
    )
    for ax in g.axes.flat:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        low = min(x0, y0)
        high = max(x1, y1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
    save_plot(
        g.fig,
        f"poisson_partA_example_neurons_session_{session_id}_choice_{choice_id:02d}.pdf",
        rasterize_scatter=True,
    )
    plt.close(g.fig)
    return g


def _poisson_mv_equal_axes(ax, x, y):
    """Square limits and y = x reference for mean-count vs variance panels."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return
    x, y = x[mask], y[mask]
    low = max(0, min(x.min(), y.min()))
    high = max(x.max(), y.max())
    ax.plot([low, high], [low, high], ls="--", c="k", lw=1, zorder=0)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)


def _axis_lim_from_series(series, default=(0.0, 1.0)):
    vals = series.dropna()
    if vals.empty:
        return default
    return max(0.0, float(vals.min())), float(vals.max())


def _poisson_ref_line_in_axes(ax):
    """Dashed y = x segment visible within current x/y limits."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    low = max(x0, y0, 0.0)
    high = min(x1, y1)
    if high > low:
        ax.plot([low, high], [low, high], ls="--", c="k", lw=1, zorder=0)


df_sub_poisson = df_poisson_stats_subsampled.drop(columns=["PR_emp"], errors="ignore")
df_sub_poisson = df_sub_poisson[
    (df_sub_poisson["session_id"].astype(str) == str(POISSON_SESSION))
    & (df_sub_poisson["area"].isin(POISSON_AREA_ORDER))
].copy()
_poisson_areas = areas_present_poisson(df_sub_poisson)
print(f"Poisson areas (plot order): {_poisson_areas}")

df_poisson_population = pd.concat(
    (expand_row_population(row) for _, row in df_sub_poisson.iterrows()),
    ignore_index=True,
)

# %%
# Part A — all neurons in session: mean_count vs neuron_var across trials
df_poisson_mv_plot = df_poisson_population.dropna(subset=["mean_count", "neuron_var"]).copy()

_row_order = [True, False]
_vis_areas_plot = [a for a in _poisson_areas if a in VISUAL_CORTEX_AREAS]
_area_xlim = {
    area: _axis_lim_from_series(
        df_poisson_mv_plot.loc[df_poisson_mv_plot["area"] == area, "mean_count"]
    )
    for area in _poisson_areas
}
_area_ylim = {
    area: _axis_lim_from_series(
        df_poisson_mv_plot.loc[df_poisson_mv_plot["area"] == area, "neuron_var"]
    )
    for area in _poisson_areas
}
if _vis_areas_plot:
    _vis_ylim = _axis_lim_from_series(
        df_poisson_mv_plot.loc[
            df_poisson_mv_plot["area"].isin(_vis_areas_plot), "neuron_var"
        ]
    )
else:
    _vis_ylim = None

g_a_mv = sns.FacetGrid(
    df_poisson_mv_plot,
    row="active",
    col="area",
    row_order=_row_order,
    col_order=_poisson_areas,
    height=3.2,
    aspect=1.05,
    sharex=False,
    sharey=False,
    margin_titles=True,
)

for row_i, active in enumerate(_row_order):
    for col_i, area in enumerate(_poisson_areas):
        ax = g_a_mv.axes[row_i, col_i]
        sub = df_poisson_mv_plot[
            (df_poisson_mv_plot["area"] == area)
            & (df_poisson_mv_plot["active"] == active)
        ]
        if sub.empty:
            continue
        sns.scatterplot(
            data=sub,
            x="mean_count",
            y="neuron_var",
            ax=ax,
            color=palette_for_area(area)[active],
            alpha=0.25,
            s=8,
        )
        ylim = _vis_ylim if area in VISUAL_CORTEX_AREAS else _area_ylim[area]
        ax.set_xlim(_area_xlim[area])
        ax.set_ylim(ylim)
        _poisson_ref_line_in_axes(ax)
        ax.set_xlabel("mean count (50 ms bin)")
        ax.set_ylabel("spike count variance")

g_a_mv.fig.subplots_adjust(top=0.92)
g_a_mv.fig.suptitle(
    f"Part A: mean–variance, all neurons (session {POISSON_SESSION}; Poisson: dashed y = x)",
    y=1.02,
)
save_plot(
    g_a_mv.fig,
    f"poisson_partA_all_neurons_session_{POISSON_SESSION}.pdf",
    rasterize_scatter=True,
)

# Part A — example neurons: one cell per area, 10 random draws (choice 00 = top-ranked)
print("Selecting near-Poisson example-neuron candidates:")
_ranked_examples = _rank_near_poisson_neurons(df_poisson_mv_plot, _poisson_areas)
for _area, _ranked in _ranked_examples.items():
    print(f"  {_area}: {len(_ranked)} candidate neurons")
_example_sets = sample_example_neuron_sets(
    _ranked_examples, N_EXAMPLE_NEURON_PLOTS, rng_seed=EXAMPLE_NEURON_RNG_SEED,
)
for _choice_id, _chosen in enumerate(_example_sets):
    print(f"Example neuron choice {_choice_id:02d}: {_chosen}")
    plot_example_neurons(
        df_poisson_mv_plot,
        _chosen,
        POISSON_AREA_ORDER,
        POISSON_SESSION,
        _choice_id,
        _row_order,
    )

# %%
# Part B — population Fano for every session with ≥50 neurons in all circuit areas
_fano_sessions = sessions_with_min_neurons(
    df_poisson_stats, POISSON_AREA_ORDER, min_neurons=FANO_MIN_NEURONS,
)
print(
    f"Part B sessions with ≥{FANO_MIN_NEURONS} neurons in {POISSON_AREA_ORDER}: "
    f"{len(_fano_sessions)} found"
)
for _sid in _fano_sessions:
    print(f"  session_id={_sid}")

for _sid in _fano_sessions:
    df_full_poisson = df_poisson_stats[
        (df_poisson_stats["session_id"].astype(str) == str(_sid))
        & (df_poisson_stats["area"].isin(POISSON_AREA_ORDER))
    ].copy()
    df_fano_expanded = pd.concat(
        (expand_row_population(row) for _, row in df_full_poisson.iterrows()),
        ignore_index=True,
    )
    df_poisson_fano = (
        df_fano_expanded.dropna(subset=["fano"])
        .groupby(["area", "active", "neuron_idx"], as_index=False)["fano"]
        .mean()
    )
    _n_neurons_area = (
        df_full_poisson.groupby("area")["N_neurons"].first().to_dict()
    )
    _poisson_areas_fano = [a for a in POISSON_AREA_ORDER if a in _n_neurons_area]
    print(
        f"Part B session {_sid} neuron counts:",
        {a: int(_n_neurons_area.get(a, 0)) for a in _poisson_areas_fano},
    )

    g_b_fano = sns.FacetGrid(
        df_poisson_fano,
        row="active",
        col="area",
        row_order=_row_order,
        col_order=_poisson_areas_fano,
        height=3.0,
        aspect=1.05,
        sharex=True,
        margin_titles=True,
    )
    for row_i, active in enumerate(_row_order):
        for col_i, area in enumerate(_poisson_areas_fano):
            ax = g_b_fano.axes[row_i, col_i]
            sub = df_poisson_fano[
                (df_poisson_fano["area"] == area)
                & (df_poisson_fano["active"] == active)
            ]
            n_area = int(_n_neurons_area.get(area, sub["neuron_idx"].nunique()))
            if row_i == 0:
                ax.set_title(f"{area} (n={n_area})")
            if sub.empty:
                continue
            sns.histplot(
                data=sub,
                x="fano",
                ax=ax,
                color=palette_for_area(area)[active],
                alpha=0.45,
                stat="density",
                bins=min(30, max(10, int(np.sqrt(len(sub))))),
            )
            ax.axvline(1.0, ls="--", c="k", lw=1)
            ax.set_xlim(0, 4)
            ax.set_xlabel("Fano factor")
            ax.set_ylabel("density")

    g_b_fano.fig.subplots_adjust(top=0.90)
    g_b_fano.fig.suptitle(
        f"Part B: population Fano (session {_sid}, one value per neuron)",
        y=1.02,
    )
    save_plot(
        g_b_fano.fig,
        f"poisson_partB_fano_session_{_sid}.pdf",
    )
    plt.close(g_b_fano.fig)

# %%
# ------------------------
# Firing-rate distributions — sessions with LGd, VISpl, VISm, SCMRN
# ------------------------

multi_area_sessions = sessions_with_all_areas(
    df_poisson_stats, MULTI_AREA_SESSION_AREAS,
)
print(
    f"Sessions with {MULTI_AREA_SESSION_AREAS}: "
    f"{len(multi_area_sessions)} found"
)
for _sid in multi_area_sessions:
    print(f"  session_id={_sid}")

for _sid in multi_area_sessions:
    _df_rates = firing_rates_long(
        df_poisson_stats[
            (df_poisson_stats["session_id"] == _sid)
            & (df_poisson_stats["area"].isin(MULTI_AREA_SESSION_AREAS))
        ]
    )
    _df_rates = _df_rates[_df_rates["mean_firing_rate"] > 0].copy()
    _df_rates["log10_mean_firing_rate"] = np.log10(
        _df_rates["mean_firing_rate"].to_numpy(dtype=float),
    )
    _g_rates = sns.displot(
        data=_df_rates,
        x="log10_mean_firing_rate",
        row="active",
        col="area",
        row_order=[True, False],
        col_order=MULTI_AREA_SESSION_AREAS,
        kind="kde",
        height=2.8,
        aspect=1.1,
        facet_kws={"margin_titles": True},
    )
    _g_rates.set(xlabel=r"$\log_{10}$(mean firing rate / Hz)")
    _g_rates.fig.subplots_adjust(top=0.90)
    _g_rates.fig.suptitle(
        f"Session {_sid}: log10 firing rates (lognormal ⇒ bell-shaped)",
        y=1.02,
    )
    save_plot(_g_rates.fig, f"firing_rate_log10_dist_session_{_sid}.pdf")
# %%
