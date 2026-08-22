# Needs dataframes visp_cloud_shape.pk and df_stats_area.pk in clean_scripts/data
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats
# %%
# ---------------------------------------------------------------
# CONFIGURATION FLAG
# ---------------------------------------------------------------
# Set to True to save plots as PDF in results folder, False to only show plots
SAVE_PLOTS = True


def get_asterisk(p_value):
    """Convert p-value to asterisk notation."""
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


def _annotate_one_sample_vs_mu(
    ax, data, y_col, categories, mu=1.0, category_col="area", x_positions=None,
):
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
        x = x_positions[i] if x_positions is not None else i
        ax.text(
            x, box_y, ast,
            ha="center", va="bottom", color="black",
            fontsize=10, fontweight="bold", clip_on=True,
        )


def _pad_ylim_for_asterisks(ax):
    ymin, ymax = ax.get_ylim()
    pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.1
    ax.set_ylim(ymin, ymax + pad)


def _style_filled_boxes(bp, box_colors):
    """Filled boxes with black edges (vis_circuit by-area boxplot style)."""
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


def _twin_ylims_sharing_reference(left_vals, right_vals, y_ref=1.0, pad_frac=0.12):
    """Y-limits for twin axes so *y_ref* sits at the same vertical position on both."""
    left = np.asarray(left_vals, dtype=float)
    right = np.asarray(right_vals, dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0:
        left = np.array([y_ref - 0.5, y_ref + 0.5])
    if right.size == 0:
        right = np.array([y_ref - 0.5, y_ref + 0.5])
    lmin, lmax = float(left.min()), float(left.max())
    rmin, rmax = float(right.min()), float(right.max())
    lmin, lmax = min(lmin, y_ref), max(lmax, y_ref)
    rmin, rmax = min(rmin, y_ref), max(rmax, y_ref)
    lpad = pad_frac * (lmax - lmin) if lmax > lmin else 0.1
    rpad = pad_frac * (rmax - rmin) if rmax > rmin else 0.1
    lmin, lmax = lmin - lpad, lmax + lpad
    rmin, rmax = rmin - rpad, rmax + rpad
    l_below, l_above = y_ref - lmin, lmax - y_ref
    r_below, r_above = y_ref - rmin, rmax - y_ref
    below = max(
        l_below / (l_below + l_above) if (l_below + l_above) > 0 else 0.5,
        r_below / (r_below + r_above) if (r_below + r_above) > 0 else 0.5,
    )
    above = max(
        l_above / (l_below + l_above) if (l_below + l_above) > 0 else 0.5,
        r_above / (r_below + r_above) if (r_below + r_above) > 0 else 0.5,
    )
    total = below + above
    below, above = below / total, above / total
    l_span = max(
        l_below / below if below > 0 else 0.0,
        l_above / above if above > 0 else 0.0,
    )
    r_span = max(
        r_below / below if below > 0 else 0.0,
        r_above / above if above > 0 else 0.0,
    )
    return (
        (y_ref - below * l_span, y_ref + above * l_span),
        (y_ref - below * r_span, y_ref + above * r_span),
    )


N_VOLUME_PCS = 5


def _ellipsoid_volume_from_eigs(eig_cell, n_pcs=N_VOLUME_PCS):
    """sqrt(prod of the leading n_pcs eigenvalues); log-sum for numerical stability."""
    if eig_cell is None or len(eig_cell) == 0:
        return np.nan
    eigs = np.asarray(eig_cell[0], dtype=float).ravel()
    if eigs.size < n_pcs:
        return np.nan
    eigs = eigs[:n_pcs]
    if np.any(eigs <= 0) or not np.all(np.isfinite(eigs)):
        return np.nan
    return float(np.exp(0.5 * np.sum(np.log(eigs))))


try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

DATA_DIR = _SCRIPT_DIR.parent / 'data'
RESULTS_DIR = _SCRIPT_DIR.parent / 'results'
# %%
# ---------------------------------------------------------------
# LOAD DATA AND CREATE DATAFRAMES: ONLY VISp AREA FOR SUMMARY PLOT IN FIGURE 1
# ---------------------------------------------------------------

stats_file = DATA_DIR / 'df_stats_area.pk'
geometry_file = DATA_DIR / 'visp_cloud_shape.pk'
df_stats_all = pd.read_pickle(stats_file)
df_geometry_stats = pd.read_pickle(geometry_file)

area_choice = 'VISp'
stats_filter = ((df_stats_all.session_number==1)&(df_stats_all.experience_level == 'Familiar')&(df_stats_all.image_name != 'omitted')
                &(df_stats_all.is_image_novel==False)&(df_stats_all.is_change==False)&(df_stats_all.area == area_choice))
geometry_filter = ((df_geometry_stats.session_number==1)&(df_geometry_stats.experience_level == 'Familiar')&(df_geometry_stats.image_name != 'omitted')
                &(df_geometry_stats.is_image_novel==False)&(df_geometry_stats.is_change==False)&(df_geometry_stats.area == area_choice))
df_stats = df_stats_all[stats_filter].drop(['session_number', 'experience_level', 'is_image_novel', 'is_change'], axis=1)
df_geometry = df_geometry_stats[geometry_filter].drop(['session_number', 'experience_level', 'is_image_novel', 'is_change'], axis=1)

# Compute volume_10PC from the leading 10 covariance eigenvalues.
# Stored volume / volume_95 use a variable number of axes (dim_95 ~ 15–78) and
# underflow to ~0; a fixed 10-PC product is comparable across sessions.
df_geometry['volume_10PC'] = df_geometry['eigenvalues_95'].apply(_ellipsoid_volume_from_eigs)

# Compute missing summary plot metrics: rbar and TR_emp
df_stats['rates'] = df_stats['avgspikecounts'].apply(lambda x: x[0] / 0.05)
df_stats['rbar'] = df_stats['rates'].apply(lambda x: np.mean(x))
df_stats['TR_emp'] = df_stats.apply(lambda row: (np.mean(row['rates'])**2) / np.mean(row['rates']**2), axis=1)
df_stats = df_stats.drop(['avgspikecounts', 'rates'], axis=1)

if "subsample_id" not in df_stats.columns:
    df_stats["subsample_id"] = (df_stats.groupby(["session_id", "active", "image_set", "image_name", "area"], dropna=False).cumcount())
if "subsample_id" not in df_geometry.columns:
    df_geometry["subsample_id"] = (df_geometry.groupby(["session_id", "active", "image_set", "image_name", "area"], dropna=False).cumcount())

# Filter columns for df_stats and df_geometry
df_stats = df_stats[['session_id', 'active', 'area', 'N', 'image_name', 'image_set', 'subsample_id', 'rbar', 'PR_emp', 'TR_emp']].copy()
df_geometry = df_geometry[['session_id', 'active', 'area', 'N', 'image_name', 'image_set', 'subsample_id', 'volume_10PC', 'cloud_separation']].copy()

# Sort both dataframes by the same keys to ensure matching order
sort_keys = ['session_id', 'active', 'area', 'N', 'image_name', 'image_set', 'subsample_id']
df_stats = df_stats.sort_values(sort_keys).reset_index(drop=True)
df_geometry = df_geometry.sort_values(sort_keys).reset_index(drop=True)

df_metrics = df_stats.copy()
df_metrics['volume_10PC'] = df_geometry['volume_10PC']
df_metrics['cloud_separation'] = df_geometry['cloud_separation']

# Example Plots DataFrames: keep image_name and image_set (1:1), average over subsample_id
df_metrics_images = df_metrics[['session_id', 'active', 'area', 'N', 'image_name', 'image_set', 
                                'rbar', 'PR_emp', 'TR_emp', 'volume_10PC', 'cloud_separation']].copy()
df_metrics_images = df_metrics_images.groupby(['session_id', 'active', 'area', 'N', 'image_name', 'image_set'], 
                                            as_index=False).mean()

# Summary Plots DataFrames: average over image_name, image_set (1:1), and subsample_id
df_metrics_session = df_metrics[['session_id', 'active', 'area', 'N', 
                                  'rbar', 'PR_emp', 'TR_emp', 'volume_10PC', 'cloud_separation']].copy()
df_metrics_session = df_metrics_session.groupby(['session_id', 'active', 'area', 'N'], as_index=False).mean()

# Summary Plots DataFrames: active vs. passive ratio
df_dummy_active = df_metrics[df_metrics['active'] == True].sort_values(sort_keys).reset_index(drop=True)
df_dummy_passive = df_metrics[df_metrics['active'] == False].sort_values(sort_keys).reset_index(drop=True)
df_ratio = df_dummy_active[['session_id', 'area', 'N', 'image_name', 'image_set']].copy()
df_ratio['rbar'] = df_dummy_active['rbar'] / df_dummy_passive['rbar']
df_ratio['PR_emp'] = df_dummy_active['PR_emp'] / df_dummy_passive['PR_emp']
df_ratio['TR_emp'] = df_dummy_active['TR_emp'] / df_dummy_passive['TR_emp']
df_ratio['volume_10PC'] = df_dummy_active['volume_10PC'] / df_dummy_passive['volume_10PC']
df_ratio['cloud_separation'] = df_dummy_active['cloud_separation'] / df_dummy_passive['cloud_separation']

df_ratio  = df_ratio[['session_id','area', 'N', 'rbar', 'PR_emp', 'TR_emp', 'volume_10PC', 'cloud_separation']].copy()
df_metrics_ratio = df_ratio.groupby(['session_id', 'area', 'N'], as_index=False).mean()
# %%
# ---------------------------------------------------------------
# SUMMARY PLOT: RATIOS ACROSS METRICS
# ---------------------------------------------------------------
ratio_metric_groups = [
    (
        ['rbar', 'TR_emp', 'PR_emp'],
        'summary_ratios_firing_stats_boxplot.pdf',
        'Summary Plot: Active vs Passive Ratios (Firing Statistics)',
    ),
    (
        ['volume_10PC', 'cloud_separation'],
        'summary_ratios_geometry_boxplot.pdf',
        'Summary Plot: Active vs Passive Ratios (Geometry)',
    ),
]

output_dir = RESULTS_DIR
if SAVE_PLOTS:
    output_dir.mkdir(parents=True, exist_ok=True)

for value_vars, filename, title in ratio_metric_groups:
    df_plot = df_metrics_ratio.melt(
        id_vars=['session_id', 'area', 'N'],
        value_vars=value_vars,
        var_name='metric',
        value_name='ratio',
    )
    g_ratio = sns.catplot(
        data=df_plot,
        x='metric',
        y='ratio',
        kind='box',
        order=value_vars,
        height=6,
        aspect=1.5,
    )
    g_ratio.set_ylabels("Ratios Active / Passive")
    g_ratio.set_xlabels("Metric")
    g_ratio.figure.suptitle(title, y=1.02)
    g_ratio.set_xticklabels(rotation=45, ha='right')
    for ax in g_ratio.axes.flat:
        ax.axhline(1, color='black', linestyle='--', linewidth=0.8, zorder=0)
        _annotate_one_sample_vs_mu(
            ax, df_plot, 'ratio', value_vars, mu=1.0, category_col='metric',
        )
        _pad_ylim_for_asterisks(ax)

    if SAVE_PLOTS:
        g_ratio.figure.savefig(output_dir / filename, format="pdf", bbox_inches="tight")
        plt.close(g_ratio.figure)
    else:
        plt.show()

# Combined 5-metric boxplot: firing stats on the left axis, geometry on the right.
left_ratio_metrics = ['rbar', 'TR_emp', 'PR_emp']
right_ratio_metrics = ['volume_10PC', 'cloud_separation']
all_ratio_metrics = left_ratio_metrics + right_ratio_metrics
ratio_metric_colors = {
    'rbar': '#03045E',
    'TR_emp': '#0077B6',
    'PR_emp': '#48CAE4',
    'volume_10PC': '#0B6E4F',
    'cloud_separation': '#F0A600',
}
df_plot_all = df_metrics_ratio.melt(
    id_vars=['session_id', 'area', 'N'],
    value_vars=all_ratio_metrics,
    var_name='metric',
    value_name='ratio',
)

left_box_data, left_box_colors = [], []
right_box_data, right_box_colors = [], []
for metric in left_ratio_metrics:
    left_box_data.append(
        df_plot_all.loc[df_plot_all['metric'] == metric, 'ratio'].dropna().to_numpy()
    )
    left_box_colors.append(ratio_metric_colors[metric])
for metric in right_ratio_metrics:
    right_box_data.append(
        df_plot_all.loc[df_plot_all['metric'] == metric, 'ratio'].dropna().to_numpy()
    )
    right_box_colors.append(ratio_metric_colors[metric])

left_positions = list(range(len(left_ratio_metrics)))
right_positions = list(range(len(left_ratio_metrics), len(all_ratio_metrics)))

fig_all, ax_left = plt.subplots(figsize=(9, 5))
ax_right = ax_left.twinx()
boxplot_kw = dict(widths=0.6, patch_artist=True, showfliers=True, zorder=2)
bp_left = ax_left.boxplot(left_box_data, positions=left_positions, **boxplot_kw)
bp_right = ax_right.boxplot(right_box_data, positions=right_positions, **boxplot_kw)
_style_filled_boxes(bp_left, left_box_colors)
_style_filled_boxes(bp_right, right_box_colors)

left_ylim, right_ylim = _twin_ylims_sharing_reference(
    np.concatenate(left_box_data) if left_box_data else np.array([1.0]),
    np.concatenate(right_box_data) if right_box_data else np.array([1.0]),
    y_ref=1.0,
    pad_frac=0.15,
)
ax_left.set_ylim(*left_ylim)
ax_right.set_ylim(*right_ylim)

n_all = len(all_ratio_metrics)
ax_left.set_xticks(range(n_all))
ax_left.set_xticklabels(all_ratio_metrics, rotation=45, ha='right')
ax_left.set_xlim(-0.5, n_all - 0.5)
ax_right.set_xlim(-0.5, n_all - 0.5)
ax_right.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax_left.set_xlabel('Metric')
ax_left.set_ylabel('Ratios Active / Passive (firing stats)')
ax_right.set_ylabel('Ratios Active / Passive (geometry)')
ax_left.grid(axis='y', linestyle='--', alpha=0.2)
ax_left.axvline(len(left_ratio_metrics) - 0.5, color='0.7', linestyle=':', linewidth=0.8, zorder=0)
ax_left.axhline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
_annotate_one_sample_vs_mu(
    ax_left,
    df_plot_all[df_plot_all['metric'].isin(left_ratio_metrics)],
    'ratio', left_ratio_metrics, mu=1.0, category_col='metric', x_positions=left_positions,
)
_annotate_one_sample_vs_mu(
    ax_right,
    df_plot_all[df_plot_all['metric'].isin(right_ratio_metrics)],
    'ratio', right_ratio_metrics, mu=1.0, category_col='metric', x_positions=right_positions,
)
fig_all.suptitle('Summary Plot: Active vs Passive Ratios Across All Metrics', y=1.02)
fig_all.tight_layout()

if SAVE_PLOTS:
    fig_all.savefig(
        output_dir / 'summary_ratios_all_metrics_twin_boxplot.pdf',
        format='pdf',
        bbox_inches='tight',
    )
    plt.close(fig_all)
else:
    plt.show()
# %%
# ---------------------------------------------------------------
# SPAGHETTI PLOTS: ACTIVE VS PASSIVE PER SESSION
# ---------------------------------------------------------------
# One figure per available session (image-averaged metrics).

metric_names = ['rbar', 'PR_emp', 'TR_emp', 'volume_10PC', 'cloud_separation']
metric_labels = {
    'rbar': 'Mean Firing Rate (rbar)',
    'PR_emp': 'Participation Ratio (PR_emp)',
    'TR_emp': 'Trial Ratio (TR_emp)',
    'volume_10PC': 'Volume (10PC)',
    'cloud_separation': 'Cloud Separation'
}
x_positions = {True: 0, False: 1}  # Active first, then passive


def _plot_session_spaghetti(df_session, session_id):
    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 6), sharey=True)
    if len(metric_names) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_names):
        df_metric = df_session[df_session[metric].notna()].copy()
        if df_metric.empty:
            ax.set_visible(False)
            continue

        pivot = df_metric.pivot_table(
            index='image_name',
            columns='active',
            values=metric,
            aggfunc='mean',
        )
        if False not in pivot.columns or True not in pivot.columns:
            ax.set_visible(False)
            continue

        pivot = pivot.dropna(subset=[False, True])
        pivot_mean = (pivot[False] + pivot[True]) / 2
        pivot_normalized = pivot.div(pivot_mean, axis=0)

        ax.axhline(1, color='black', linestyle='--', linewidth=0.8, zorder=0)
        for _, row in pivot_normalized.iterrows():
            ax.plot(
                [x_positions[True], x_positions[False]],
                [row[True], row[False]],
                color='black',
                linewidth=1.0,
                alpha=0.65,
                zorder=2,
            )
        for active_state in [True, False]:
            normalized_values = [
                pivot_normalized.loc[img_name, active_state]
                for img_name in pivot_normalized.index
            ]
            ax.scatter(
                np.full(len(normalized_values), x_positions[active_state]),
                normalized_values,
                s=70,
                facecolors='white',
                edgecolors='black',
                linewidths=0.8,
                alpha=0.7,
                zorder=3,
                rasterized=False,
            )

        ax.set_xticks([x_positions[True], x_positions[False]])
        ax.set_xticklabels(['Active', 'Passive'])
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(metric_labels[metric])
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylabel(f"{metric_labels[metric]} (Normalized)")

    fig.suptitle(f"Session {session_id}: Active vs Passive Trajectories", y=0.98)
    fig.tight_layout()
    return fig


session_ids = sorted(df_metrics_images['session_id'].astype(str).unique())
for session_id in session_ids:
    df_session = df_metrics_images[df_metrics_images['session_id'].astype(str) == session_id].copy()
    if df_session.empty:
        continue
    fig = _plot_session_spaghetti(df_session, session_id)
    if SAVE_PLOTS:
        output_dir = RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"spaghetti_metrics_{session_id}.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)
    else:
        plt.show()
# %%



















































# %%