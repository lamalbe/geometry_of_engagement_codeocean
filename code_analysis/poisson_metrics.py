# %%
"""
Analysis of area-level Poisson / sparsity statistics (df_poisson_stats).

Loads full and subsampled dataframes from clean_scripts/data
(df_poisson_stats.pk, df_poisson_stats_subsampled.pk). Derives Treves–Rolls
sparsity of mean rates (TR_emp), of spike-count variances
(spikecount_var_sparsity), Fano-like sparsity (a_sparsity), and mean network
rate (R_emp) on the non-subsampled table. Figures are written to
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
# AREA_ORDER = ["VISp", "LGd", "VISpl", "VISm", "SCMRN"]
# MULTI_AREA_SESSION_AREAS = ["VISp", "LGd", "VISpl", "VISm", "SCMRN"]
POISSON_AREA_ORDER = ["VISp", "VISlat", "LGd", "VISpl", "VISm", "SCMRN"]
AREA_ORDER = ["LGd", "VISpl", "VISm", "SCMRN"]
MULTI_AREA_SESSION_AREAS = ["LGd", "VISpl", "VISm", "SCMRN"]
VISUAL_CORTEX_AREAS = {"VISp", "VISlat", "VISpl", "VISm"}
POISSON_SESSION = 1118327332
POISSON_EXAMPLE_NEURONS = {
    "LGd": 20,
    "VISlat": 41,
    "SCMRN": 2,
    "VISm": 7,
    "VISp": 43,
}

SAVE_PLOTS = True  # Set to True to save plots as PDF in results folder
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
    "VISp": {True: "#34A0A4", False: "#48CAE4"},
    "LGd": {True: "#80B918", False: "#BFD200"},
    "VISpl": {True: "#34A0A4", False: "#48CAE4"},
    "VISlat": {True: "#34A0A4", False: "#48CAE4"},
    "VISm": {True: "#34A0A4", False: "#48CAE4"},
    "SCMRN": {True: "#C36F09", False: "#EEBA0B"},
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


def _annotate_one_sample_vs_mu(ax, data, y_col, areas, mu=1.0, area_col="area"):
    """Black asterisk above each area (two-sided one-sample t-test vs mu)."""
    y_offset = _y_offset_for_annotations(data[y_col])
    for i, area in enumerate(areas):
        vals = data.loc[data[area_col] == area, y_col].dropna()
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


def catplot_area_stats(data, y, kind, test, marker=None, mu=1.0, **catplot_kw):
    """
    Catplot with fixed area order and significance asterisks.

    test: 'two_sample' (active vs passive) or 'one_sample' (vs mu, default 1).
    """
    plot_df = data[data["area"].isin(AREA_ORDER)].copy()
    areas = areas_present(plot_df)
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
# Non-subsampled: TR_emp, a_sparsity (C_ii / r_i* per neuron), R_emp

df_poisson_stats["TR_emp"] = df_poisson_stats.apply(
    lambda row: (np.mean(row["mean_firing_rates"]) ** 2)
    / np.mean(row["mean_firing_rates"] ** 2),
    axis=1,
)

df_poisson_stats["spikecount_var_sparsity"] = df_poisson_stats.apply(
    lambda row: (np.mean(row["spikecount_variances"]) ** 2)
    / np.mean(row["spikecount_variances"] ** 2),
    axis=1,
)

df_poisson_stats["a_sparsity"] = df_poisson_stats.apply(
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

df_poisson_stats["R_emp"] = df_poisson_stats.apply(
    lambda row: np.mean(row["mean_firing_rates"]),
    axis=1,
)
df_poisson_stats["PR_emp"] = df_poisson_stats["PR_emp"] / df_poisson_stats["N_neurons"]
df_poisson_stats["RHS"] = df_poisson_stats["TR_emp"] * df_poisson_stats["a_sparsity"]
df_poisson_stats["var_sparsity_ratio"] = np.divide(
    df_poisson_stats["spikecount_var_sparsity"],
    df_poisson_stats["RHS"],
    out=np.full(len(df_poisson_stats), np.nan, dtype=float),
    where=np.isfinite(df_poisson_stats["RHS"].to_numpy())
    & (df_poisson_stats["RHS"].to_numpy() != 0),
)
df_poisson_stats["tian_angular_coefficient"] = df_poisson_stats["PR_emp"] / (df_poisson_stats["TR_emp"])
df_poisson_stats["angular_coefficient"] = df_poisson_stats["PR_emp"] / (df_poisson_stats["TR_emp"]* df_poisson_stats["a_sparsity"])

# Active / passive ratios per (area, session_id, image_name)
group_keys = ["area", "session_id", "image_name"]
ratio_metrics = ["PR_emp", "TR_emp", "a_sparsity", "R_emp", "RHS", "angular_coefficient", "tian_angular_coefficient"]

wide = (
    df_poisson_stats.set_index(group_keys + ["active"])[ratio_metrics]
    .unstack("active")
)

df_stats_ap_ratio = wide.index.to_frame(index=False)
for metric in ratio_metrics:
    df_stats_ap_ratio[f"{metric}_ratio"] = (
        wide[(metric, True)] / wide[(metric, False)]
    ).values

df_stats_ap_ratio["R_TR_combo_1"] = (
    df_stats_ap_ratio["R_emp_ratio"] ** (4 / 3)
    / df_stats_ap_ratio["TR_emp_ratio"] ** (2 / 9)
)
df_stats_ap_ratio["R_TR_combo_2"] = (
    df_stats_ap_ratio["R_emp_ratio"] ** 2
    / df_stats_ap_ratio["TR_emp_ratio"]
)
# %%
sns.catplot(data=df_stats_ap_ratio, x="area", y="R_emp_ratio", kind="box")
sns.catplot(data=df_stats_ap_ratio, x="area", y="R_TR_combo_1", kind="box")
sns.catplot(data=df_stats_ap_ratio, x="area", y="R_TR_combo_2", kind="box")

sns.displot(data=df_stats_ap_ratio, x="R_TR_combo_1", hue="area", kind="kde")
sns.displot(data=df_stats_ap_ratio, x="R_TR_combo_2", hue="area", kind="kde")
# %%
g_cat_angular = catplot_area_stats(
    df_poisson_stats, y="angular_coefficient", kind="point", test="two_sample", marker="o",
)
save_plot(g_cat_angular.fig, "catplot_angular_coefficient.pdf")

sns.displot(data=df_poisson_stats, x="angular_coefficient", hue="active", kind="kde", row="area")

# g_cat_tian = catplot_area_stats(
#     df_poisson_stats, y="tian_angular_coefficient", kind="point", test="two_sample", marker="D",
# )
#save_plot(g_cat_tian.fig, "catplot_tian_angular_coefficient.pdf")

g_cat_a_sparsity = catplot_area_stats(df_poisson_stats, y="a_sparsity", kind="box", test="two_sample")
save_plot(g_cat_a_sparsity.fig, "catplot_a_sparsity.pdf")

_df_var_sparsity = df_poisson_stats[df_poisson_stats["area"].isin(AREA_ORDER)].dropna(
    subset=["var_sparsity_ratio"]
).copy()
_var_sparsity_areas = areas_present(_df_var_sparsity)
_df_var_sparsity["area"] = pd.Categorical(
    _df_var_sparsity["area"], categories=_var_sparsity_areas, ordered=True
)
# Per-area palette_map: one stripplot per area (catplot uses a single hue palette).
_fig_var_sparsity, _ax_var_sparsity = plt.subplots(
    figsize=(max(7.0, 1.15 * len(_var_sparsity_areas)), 4)
)
for _area in _var_sparsity_areas:
    _sub_area = _df_var_sparsity[_df_var_sparsity["area"] == _area]
    if _sub_area.empty:
        continue
    sns.stripplot(
        data=_sub_area,
        x="area",
        y="var_sparsity_ratio",
        hue="active",
        hue_order=[True, False],
        order=_var_sparsity_areas,
        palette=palette_for_area(_area),
        dodge=True,
        ax=_ax_var_sparsity,
        alpha=0.6,
        size=4,
        jitter=True,
        legend=False,
    )
_ax_var_sparsity.axhline(1.0, ls="--", c="k", lw=1, zorder=0)
_ax_var_sparsity.set_ylabel(
    r"spikecount_var_sparsity / ($TR_{emp} \times a_{sparsity}$)"
)
_fig_var_sparsity.suptitle(
    r"Variance sparsity ratio (theory: $y = 1$)",
    y=1.02,
)
_fig_var_sparsity.tight_layout()
save_plot(
    _fig_var_sparsity,
    "catplot_var_sparsity_ratio.pdf",
    rasterize_scatter=True,
)

# sns.displot(data=df_poisson_stats, x="angular_coefficient", hue="active", kind="hist", row="area")
# sns.catplot(data=df_poisson_stats, x="area", y="R_emp", hue="active", kind="box")
# sns.displot(data=df_poisson_stats, x="R_emp", hue="active", kind="kde", row="area")
# %%
for area_id in df_poisson_stats.area.unique():
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
g_cat_tian_ratio = catplot_area_stats(
    df_stats_ap_ratio, y="angular_coefficient_ratio", kind="box", test="one_sample", mu=1.0,
)
save_plot(g_cat_tian_ratio.fig, "catplot_angular_coefficient_ratio.pdf")

g_cat_a_sparsity_ratio = catplot_area_stats(
    df_stats_ap_ratio, y="a_sparsity_ratio", kind="box", test="one_sample", mu=1.0,
)
save_plot(g_cat_a_sparsity_ratio.fig, "catplot_a_sparsity_ratio.pdf")

g_cat_r_tr_combo_1 = catplot_area_stats(
    df_stats_ap_ratio, y="R_TR_combo_1", kind="box", test="one_sample", mu=1.0,
)
save_plot(g_cat_r_tr_combo_1.fig, "catplot_R_TR_combo_1.pdf")

g_cat_r_tr_combo_2 = catplot_area_stats(
    df_stats_ap_ratio, y="R_TR_combo_2", kind="box", test="one_sample", mu=1.0,
)
save_plot(g_cat_r_tr_combo_2.fig, "catplot_R_TR_combo_2.pdf")

sns.displot(data=df_stats_ap_ratio, x="angular_coefficient_ratio", hue="area", kind="kde")

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
    color=palette_for_area("VISp")[True],
    label=rf"active ($\bar{{r}}$ = {R_BAR_ACTIVE_TOY:g})",
)
ax_rhs_toy.plot(
    _g_toy,
    (1 - _g_toy**2 * R_BAR_PASSIVE_TOY) ** 2,
    color=palette_for_area("VISp")[False],
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
ax = sns.scatterplot(
    data=df_poisson_stats[df_poisson_stats.area.isin(['VISp'])],
    y="PR_emp",
    x="RHS",
    hue="active",
)

# Set same range for x and y axis
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
low = min(xmin, ymin)
high = max(xmax, ymax)
ax.set_xlim(low, high)
ax.set_ylim(low, high)
# Add y=x line
ax.plot([low, high], [low, high], ls="--", c="k", lw=1, zorder=0)

ax = sns.displot(
    data=df_poisson_stats[df_poisson_stats.area.isin(['VISp'])],
    y="PR_emp",
    x="TR_emp",
    hue="active",
    kind="kde",
)

# %%
# ------------------------
# POISSON CONSISTENCY — Part A & B (fixed session, POISSON_AREA_ORDER)
# ------------------------
# Trial = (active, image_name, subsample_id). Part B: per-neuron Fano over pooled trials.

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
    out = {col: row[col] for col in META_COLS}
    out.update(
        {
            "neuron_idx": np.arange(n, dtype=int),
            "mean_count": mean_count,
            "neuron_var": neuron_var,
            "fano": fano,
        }
    )
    return pd.DataFrame(out)


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
    df_sub_poisson["session_id"].astype(str) == str(POISSON_SESSION)
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

# Part A — example neurons (one panel per area, shared x/y across areas)
_example_areas = [a for a in POISSON_AREA_ORDER if a in POISSON_EXAMPLE_NEURONS]
_example_parts = []
for area, neuron_idx in POISSON_EXAMPLE_NEURONS.items():
    sub = df_poisson_mv_plot[
        (df_poisson_mv_plot["area"] == area)
        & (df_poisson_mv_plot["neuron_idx"] == neuron_idx)
    ]
    if sub.empty:
        print(f"Warning: no data for {area} neuron {neuron_idx} in session {POISSON_SESSION}")
    else:
        _example_parts.append(sub)
df_poisson_examples = pd.concat(_example_parts, ignore_index=True)

_ex_x = df_poisson_examples["mean_count"].dropna()
_ex_y = df_poisson_examples["neuron_var"].dropna()
_ex_vals = pd.concat([_ex_x, _ex_y], ignore_index=True)
if _ex_vals.empty:
    _ex_low, _ex_high = 0.0, 1.0
else:
    _ex_low = max(0.0, float(_ex_vals.min()))
    _ex_high = float(_ex_vals.max())

g_a_lm = sns.FacetGrid(
    df_poisson_examples,
    col="area",
    col_order=_example_areas,
    height=3.2,
    aspect=1.0,
    sharex=True,
    sharey=True,
    margin_titles=True,
)
for col_i, area in enumerate(_example_areas):
    ax = g_a_lm.axes[0, col_i]
    neuron_idx = POISSON_EXAMPLE_NEURONS[area]
    sub_n = df_poisson_examples[df_poisson_examples["area"] == area]
    pal = palette_for_area(area)
    for active in _row_order:
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
            label=f"active={active}",
        )
    ax.plot([_ex_low, _ex_high], [_ex_low, _ex_high], ls="--", c="k", lw=1, zorder=0)
    ax.set_xlim(_ex_low, _ex_high)
    ax.set_ylim(_ex_low, _ex_high)
    ax.set_xlabel("mean count (50 ms bin)")
    ax.set_ylabel("spike count variance")
    ax.set_title(f"{area}  n={neuron_idx}", fontsize=9)
    ax.legend(fontsize=7, loc="best")

g_a_lm.fig.subplots_adjust(top=0.90)
g_a_lm.fig.suptitle(
    f"Part A: example neurons (session {POISSON_SESSION}; Poisson: dashed y = x)",
    y=1.02,
)
save_plot(
    g_a_lm.fig,
    f"poisson_partA_example_neurons_session_{POISSON_SESSION}.pdf",
    rasterize_scatter=True,
)

# %%
# Part B — population Fano distribution (all neurons, all areas)
df_poisson_population_plot = df_poisson_population.dropna(subset=["fano"]).copy()

g_b_fano = sns.FacetGrid(
    df_poisson_population_plot,
    row="active",
    col="area",
    row_order=_row_order,
    col_order=_poisson_areas,
    height=3.0,
    aspect=1.05,
    sharex=True,
    margin_titles=True,
)
for row_i, active in enumerate(_row_order):
    for col_i, area in enumerate(_poisson_areas):
        ax = g_b_fano.axes[row_i, col_i]
        sub = df_poisson_population_plot[
            (df_poisson_population_plot["area"] == area)
            & (df_poisson_population_plot["active"] == active)
        ]
        if sub.empty:
            continue
        sns.histplot(
            data=sub,
            x="fano",
            ax=ax,
            color=palette_for_area(area)[active],
            alpha=0.45,
            stat="density",
            bins=30,
        )
        ax.axvline(1.0, ls="--", c="k", lw=1)
        ax.set_xlim(0, 4)
        ax.set_xlabel("Fano factor")
        ax.set_ylabel("density")

g_b_fano.fig.subplots_adjust(top=0.92)
g_b_fano.fig.suptitle(
    f"Part B: population Fano (session {POISSON_SESSION}, all units, pooled trials)",
    y=1.02,
)
save_plot(
    g_b_fano.fig,
    f"poisson_partB_fano_session_{POISSON_SESSION}.pdf",
)

# %%
# ------------------------
# Firing-rate distributions — sessions with VISp, LGd, VISpl, VISm, SCMRN
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
