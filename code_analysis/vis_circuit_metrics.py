# %%
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats
# %%
# -------------
#  LOAD DATASET
# -------------
"""
session number: indicates whether a session was the first or second recording day for a given mouse.
image set: indicates whether image set G or H was shown during a particular session.
experience level: indicates whether the image set shown was familiar to the mouse from training or novel.
is_change: Indicates whether the image identity changed for this stimulus presentation.
           When both this value and 'active' are TRUE, the mouse was rewarded for licking within the response window.
is_image_novel: Indicates whether this image has been shown to the mouse in previous training sessions.
                If True, then this image is novel to the mouse.
"""

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

DATA_DIR = _SCRIPT_DIR.parent / 'data'
RESULTS_DIR = _SCRIPT_DIR.parent / 'results'

stats_file = DATA_DIR / 'df_stats_area.pk'
df_stats_all = pd.read_pickle(stats_file)

data_filter = ((df_stats_all.session_number==1)&(df_stats_all.experience_level == 'Familiar')&(df_stats_all.image_name != 'omitted')
                &(df_stats_all.is_image_novel==False)&(df_stats_all.is_change==False))
df_stats = df_stats_all[data_filter].drop(['session_number', 'experience_level', 'is_image_novel', 'is_change'], axis=1)

df_stats = df_stats.drop(['speed_values', 'pupil_values'], axis=1, errors='ignore')
df_stats = df_stats.drop(['PR_fit'], axis=1, errors='ignore')

df_stats['rates'] = df_stats['avgspikecounts'].apply(lambda x: x[0] / 0.05)
df_stats['rbar'] = df_stats['rates'].apply(lambda x: np.mean(x))
df_stats['TR_emp'] = df_stats.apply(lambda row: (np.mean(row['rates'])**2) / np.mean(row['rates']**2), axis=1)


def _a_sparsity_from_rates_variances(rates, variances):
    """Treves–Rolls sparsity of per-neuron C_ii / r_i (poisson_metrics a_sparsity)."""
    rates = np.asarray(rates, dtype=float).ravel()
    if isinstance(variances, (list, tuple)):
        variances = variances[0] if len(variances) else np.array([])
    variances = np.asarray(variances, dtype=float).ravel()
    if rates.size == 0 or np.all(rates == 0):
        return np.nan
    a = np.divide(
        variances, rates,
        out=np.zeros_like(rates, dtype=float),
        where=rates > 0,
    )
    denom = np.mean(a ** 2)
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float((np.mean(a) ** 2) / denom)


df_stats['a_sparsity'] = df_stats.apply(
    lambda row: _a_sparsity_from_rates_variances(row['rates'], row['variances']),
    axis=1,
)
df_stats = df_stats.drop(['avgspikecounts', 'rates', 'variances'], axis=1)

if "subsample_id" not in df_stats.columns:
    df_stats["subsample_id"] = (df_stats.groupby(["session_id", "active", "image_set", "image_name", "area"], dropna=False).cumcount())

# Reorder dataframe
# re_order = [
#     "session_id", "area", "N", "active", 
#     "image_set", "image_name", "subsample_id",
#     "mii", "mij", "sii", "sij", 
#     "TV_emp", "PR_emp", "speed", "pupil"]
# df_stats = df_stats.reindex(columns=re_order)

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
    ax, data, y_col, categories, mu=1.0, category_col="area", y_offset=None,
):
    """Black asterisk above each category (two-sided one-sample t-test vs mu)."""
    if y_offset is None:
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
    """Filled boxes with black edges (visual-cortex by-area boxplot style)."""
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


def _tukey_inliers(vals, whis=1.5):
    """Keep values inside the matplotlib boxplot whiskers (Q1/Q3 ± whis * IQR)."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return vals
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - whis * iqr, q3 + whis * iqr
    return vals[(vals >= lo) & (vals <= hi)]


def _box_ylim_values(box_data, cut_outliers=False):
    """Values that should set the y-axis: all points, or Tukey inliers only."""
    if not cut_outliers:
        return np.concatenate(box_data)
    inliers = [_tukey_inliers(v) for v in box_data]
    inliers = [v for v in inliers if v.size]
    if not inliers:
        return np.concatenate(box_data)
    return np.concatenate(inliers)

# %%
# -------------------------------------------------------------------------------------------------
# ACTIVE/PASSIVE RATIO BY AREA (LGd, VISpl, VISm, SCMRN)
# -------------------------------------------------------------------------------------------------
# When True, all ratio boxplots in this section share a common y-axis.
SHARE_Y_AXIS = False
# When True, omit Tukey outliers from the plots and set y-limits from inliers.
# Shared y-limits (SHARE_Y_AXIS) are then computed from those inliers.
CUT_OUTLIERS = False

box_area_order = ['LGd', 'VISpl', 'VISm', 'SCMRN']
metrics_ratio = ['rbar', 'TR_emp', 'PR_emp', 'a_sparsity']
ratio_keys = ['session_id', 'image_name', 'area']
avg_group_cols = ['session_id', 'image_name', 'active', 'area']
circuit_box_colors = {
    'LGd': '#D679AD',
    'VISpl': '#53BBF7',
    'VISm': '#0079BC',
    'SCMRN': '#F0A600',
}

df_ratio_areas = df_stats.copy()
df_ratio_areas = df_ratio_areas[
    (df_ratio_areas['area'].isin(box_area_order))
    & (df_ratio_areas['N'] >= 30)
    & (df_ratio_areas['image_name'] != 'omitted')
]

if not df_ratio_areas.empty:
    numeric_cols = df_ratio_areas.select_dtypes(include=np.number).columns.tolist()
    avg_cols = [col for col in numeric_cols if col not in avg_group_cols + ['subsample_id']]

    df_ratio_avg = (
        df_ratio_areas[avg_group_cols + avg_cols]
        .groupby(avg_group_cols, as_index=False)[avg_cols]
        .mean()
    )

    output_dir = RESULTS_DIR
    if SAVE_PLOTS:
        output_dir.mkdir(parents=True, exist_ok=True)

    ratio_plot_cases = []
    for metric in metrics_ratio:
        df_pivot = df_ratio_avg.pivot_table(
            index=ratio_keys,
            columns='active',
            values=metric,
            aggfunc='mean',
        ).reset_index()
        df_pivot['ratio'] = df_pivot[True] / df_pivot[False]
        df_plot_ratio = df_pivot[ratio_keys + ['ratio']].copy()
        df_plot_ratio = df_plot_ratio[
            df_plot_ratio['ratio'].notna() & np.isfinite(df_plot_ratio['ratio'])
        ]
        df_plot_ratio['area'] = pd.Categorical(
            df_plot_ratio['area'],
            categories=box_area_order,
            ordered=True,
        )

        box_data, box_colors, tick_labels = [], [], []
        for area in box_area_order:
            vals = df_plot_ratio.loc[df_plot_ratio['area'] == area, 'ratio'].dropna().to_numpy()
            if len(vals) == 0:
                continue
            box_data.append(vals)
            box_colors.append(circuit_box_colors[area])
            tick_labels.append(area)
        if not box_data:
            continue
        ratio_plot_cases.append((metric, df_plot_ratio, box_data, box_colors, tick_labels))

    shared_ylim = None
    if SHARE_Y_AXIS and ratio_plot_cases:
        shared_ylim = _ratio_ylim(
            np.concatenate([
                _box_ylim_values(case[2], CUT_OUTLIERS) for case in ratio_plot_cases
            ])
        )

    for metric, df_plot_ratio, box_data, box_colors, tick_labels in ratio_plot_cases:
        ylim_vals = _box_ylim_values(box_data, CUT_OUTLIERS)
        y_lim = shared_ylim if shared_ylim is not None else _ratio_ylim(ylim_vals)
        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(
            box_data,
            positions=range(len(box_data)),
            widths=0.6,
            patch_artist=True,
            showfliers=not CUT_OUTLIERS,
            zorder=2,
        )
        _style_filled_boxes(bp, box_colors)
        ax.set_xticks(range(len(box_data)))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_xlabel('Area')
        ax.set_ylabel(f'Active / Passive Ratio ({metric})')
        ax.set_ylim(y_lim)
        ax.set_xlim(-0.5, len(box_data) - 0.5)
        ax.grid(axis='y', linestyle='--', alpha=0.2)
        ax.axhline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
        _annotate_one_sample_vs_mu(
            ax, df_plot_ratio, 'ratio', tick_labels, mu=1.0, category_col='area',
            y_offset=_y_offset_for_annotations(pd.Series(ylim_vals)),
        )
        _pad_ylim_for_asterisks(ax)
        fig.suptitle(
            f'Active vs Passive Ratio: {metric} (averaged over subsamples)',
            y=1.02,
        )
        fig.tight_layout()

        if SAVE_PLOTS:
            fig.savefig(
                output_dir / f"ratio_box_{'-'.join(box_area_order)}_{metric}.pdf",
                format='pdf',
                bbox_inches='tight',
                dpi=300,
            )
            plt.close(fig)
        else:
            plt.show()

# %%
# -------------------------------------------------------------------------------------------------
# ACTIVE/PASSIVE RATIO BY VISUAL CORTEX AREA
# -------------------------------------------------------------------------------------------------

visual_cortex_areas = ['VISp', 'VISpm', 'VISam', 'VISrl', 'VISl', 'VISal']
metrics_ratio_vis = ['rbar', 'TR_emp', 'PR_emp']
ratio_keys_vis = ['session_id', 'image_name', 'area']
avg_group_cols_vis = ['session_id', 'image_name', 'active', 'area']

df_ratio_vis_areas = df_stats.copy()
df_ratio_vis_areas = df_ratio_vis_areas[
    (df_ratio_vis_areas['area'].isin(visual_cortex_areas))
    & (df_ratio_vis_areas['N'] >= 30)
    & (df_ratio_vis_areas['image_name'] != 'omitted')
]

if not df_ratio_vis_areas.empty:
    numeric_cols = df_ratio_vis_areas.select_dtypes(include=np.number).columns.tolist()
    avg_cols = [col for col in numeric_cols if col not in avg_group_cols_vis + ['subsample_id']]

    df_ratio_vis_avg = (
        df_ratio_vis_areas[avg_group_cols_vis + avg_cols]
        .groupby(avg_group_cols_vis, as_index=False)[avg_cols]
        .mean()
    )

    output_dir = RESULTS_DIR
    if SAVE_PLOTS:
        output_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics_ratio_vis:
        df_pivot = df_ratio_vis_avg.pivot_table(
            index=ratio_keys_vis,
            columns='active',
            values=metric,
            aggfunc='mean',
        ).reset_index()
        df_pivot['ratio'] = df_pivot[True] / df_pivot[False]
        df_plot_ratio = df_pivot[ratio_keys_vis + ['ratio']].copy()
        df_plot_ratio['area'] = pd.Categorical(
            df_plot_ratio['area'],
            categories=visual_cortex_areas,
            ordered=True,
        )

        g_ratio = sns.catplot(
            data=df_plot_ratio,
            x='area',
            y='ratio',
            kind='box',
            order=visual_cortex_areas,
            height=4,
            aspect=1.4,
        )
        g_ratio.set_ylabels(f'Active / Passive Ratio ({metric})')
        g_ratio.set_xlabels('Area')
        g_ratio.figure.suptitle(
            f'Active vs Passive Ratio: {metric} (visual cortex, averaged over subsamples)',
            y=1.02,
        )
        for ax in g_ratio.axes.flat:
            ax.axhline(1, color='black', linestyle='--', linewidth=0.8, zorder=0)
            _annotate_one_sample_vs_mu(
                ax, df_plot_ratio, 'ratio', visual_cortex_areas, mu=1.0, category_col='area',
            )
            _pad_ylim_for_asterisks(ax)

        if SAVE_PLOTS:
            g_ratio.figure.savefig(
                output_dir / f"ratio_box_visual_cortex_{metric}.pdf",
                format="pdf",
                bbox_inches="tight",
            )
            plt.close(g_ratio.figure)
        else:
            plt.show()

# %%
# -------------------------------------------------------------------------------------------------
# VISUAL CORTEX ACTIVE/PASSIVE SPAGHETTI PLOTS WITH SUBSAMPLE AVERAGING
# -------------------------------------------------------------------------------------------------

df_visual_spaghetti = df_stats.copy()
df_visual_spaghetti = df_visual_spaghetti[df_visual_spaghetti['area'].isin(visual_cortex_areas)].copy()
df_visual_spaghetti = df_visual_spaghetti[df_visual_spaghetti['image_name'] != 'omitted']

if not df_visual_spaghetti.empty:
    df_visual_spaghetti['PR_emp_norm'] = df_visual_spaghetti['PR_emp'] / df_visual_spaghetti['N']

    group_cols = ['session_id', 'active', 'area']
    numeric_cols = df_visual_spaghetti.select_dtypes(include=np.number).columns.tolist()
    avg_cols = [col for col in numeric_cols if col not in group_cols + ['subsample_id']]

    df_visual_spaghetti_avg = (
        df_visual_spaghetti[group_cols + avg_cols]
        .groupby(group_cols, as_index=False)[avg_cols]
        .mean()
    )

    df_visual_spaghetti_avg['area'] = pd.Categorical(
        df_visual_spaghetti_avg['area'],
        categories=visual_cortex_areas,
        ordered=True
    )

    metric_configs = [
        ('PR_emp_norm', 'Normalized Participation Ratio'),
        ('rbar', 'Population Mean Activity')
    ]

    rng = np.random.default_rng(seed=0)
    palette = {False: '#1f77b4', True: '#ff7f0e'}

    for metric, y_label in metric_configs:
        if metric not in df_visual_spaghetti_avg.columns:
            continue

        df_metric = df_visual_spaghetti_avg[df_visual_spaghetti_avg[metric].notna()].copy()
        if df_metric.empty:
            continue

        areas_present = [area for area in visual_cortex_areas if area in df_metric['area'].unique()]
        if not areas_present:
            continue

        n_cols = 3
        n_rows = math.ceil(len(areas_present) / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 4, n_rows * 4),
            sharey=False,
            squeeze=False
        )

        for idx, area in enumerate(areas_present):
            ax = axes[idx // n_cols, idx % n_cols]
            area_data = df_metric[df_metric['area'] == area]

            # Draw connecting lines between passive and active for each session
            pivot = (
                area_data.pivot_table(
                    index='session_id',
                    columns='active',
                    values=metric,
                    aggfunc='mean'
                )
            )
            if True in pivot.columns and False in pivot.columns:
                pivot = pivot.dropna(subset=[True, False])
                for _, row in pivot.iterrows():
                    ax.plot(
                        [0, 1],
                        [row[True], row[False]],
                        color='black',
                        linewidth=1,
                        alpha=0.6,
                        zorder=2
                    )

            # Scatter points for each active state (active first)
            for x_pos, active_val in enumerate([True, False]):
                subset = area_data[area_data['active'] == active_val]
                if subset.empty:
                    continue

                jitter = rng.uniform(-0.08, 0.08, len(subset))
                ax.scatter(
                    np.full(len(subset), x_pos) + jitter,
                    subset[metric],
                    s=70,
                    facecolors='white',
                    edgecolors=palette[active_val],
                    linewidths=1.5,
                    alpha=0.95,
                    zorder=3
                )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Active', 'Passive'])
            ax.set_xlim(-0.4, 1.4)
            ax.set_title(area)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            if idx % n_cols == 0:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel('')

        # Hide unused axes
        total_axes = n_rows * n_cols
        for idx in range(len(areas_present), total_axes):
            axes[idx // n_cols, idx % n_cols].axis('off')

        fig.suptitle(f'Active vs Passive {y_label} (Session Trajectories)', y=0.98)
        fig.tight_layout()

        if SAVE_PLOTS:
            output_dir = RESULTS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_dir / f"{metric}_spaghetti_visual_cortex.pdf",
                format="pdf",
                bbox_inches="tight"
            )
            plt.close(fig)
        else:
            plt.show()

# %%
# ------------------------------
# INTER-AREA - PLOTS PER SESSION
# ------------------------------

# Define palette for each area combination
inter_area_combinations = [
    ['LGd', 'VISpl'],
    ['VISpl', 'VISm'],
    ['VISm', 'SCMRN']
]

# Map each combination to its specific palette
palette_map = {
    tuple(['LGd', 'VISpl']): {True: '#2A7FFF', False: '#AA0000'},
    tuple(['VISpl', 'VISm']): {True: '#2A7FFF', False: '#AA0000'},
    tuple(['VISm', 'SCMRN']): {True: '#2A7FFF', False: '#AA0000'}
}

for inter_area in inter_area_combinations:
    # Get the palette for this specific combination
    palette_inter = palette_map[tuple(inter_area)]
    df_plot = df_stats.copy()
    df_plot = df_plot[df_plot.area.isin(inter_area)]
    df_plot = df_plot[df_plot.N >= 30]
    df_plot = df_plot[df_plot.image_name != 'omitted']
    df_plot['PR_emp'] = df_plot['PR_emp'] / df_plot['N']
    df_plot['TV_emp'] = df_plot['TV_emp'] / df_plot['N']

    valid_sessions = (df_plot.groupby('session_id')['area'].apply(lambda x: set(inter_area).issubset(set(x))))
    df_plot = df_plot[df_plot.session_id.isin(valid_sessions[valid_sessions].index)]

    keys = ['session_id', 'active', 'image_name', 'subsample_id']
    metrics = ['mii', 'mij', 'sii', 'sij', 'TV_emp', 'PR_emp', 'rbar']

    # Pivot directly: one column per (metric, area)
    df_pivot = df_plot.pivot(index=keys, columns='area', values=metrics)

    # Flatten multi-index column names (metric, area) → "metric_area"
    df_pivot.columns = [f"{metric}_{area}" for metric, area in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    # Metric pairs to plot - leave space for future additions
    metric_pairs = [
        ('TV_emp', 'mii'),
        # ('TV_emp', 'PR_emp'),
        ('PR_emp', 'PR_emp'),
        ('rbar', 'rbar'),
        # Add more metric pairs here in the future
    ]

    # loop over sessions
    for session in df_pivot['session_id'].unique():
        df_sess = df_pivot[df_pivot['session_id'] == session]
        
        # Create figure with subplots - one column per metric pair
        n_cols = len(metric_pairs)
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4), sharey=False)
        
        # Handle single subplot case (when n_cols == 1)
        if n_cols == 1:
            axes = [axes]
        
        # loop over metric pairs and plot in columns
        for col_idx, (metric_base_x, metric_base_y) in enumerate(metric_pairs):
            ax = axes[col_idx]
            metric_x = f'{metric_base_x}_{inter_area[0]}'
            metric_y = f'{metric_base_y}_{inter_area[1]}'
            
            # Skip if metrics missing in this session
            if metric_x not in df_sess.columns or metric_y not in df_sess.columns:
                ax.axis('off')
                continue
            
            df_xy = df_sess[[metric_x, metric_y, 'active']].dropna()
            if df_xy.empty:
                ax.axis('off')
                continue
            
            # Plot using seaborn regplot on the specific axis
            for active_val in [False, True]:
                df_xy_subset = df_xy[df_xy['active'] == active_val]
                if df_xy_subset.empty:
                    continue
                
                sns.regplot(
                    data=df_xy_subset,
                    x=metric_x,
                    y=metric_y,
                    scatter=True,
                    ax=ax,
                    color=palette_inter[active_val],
                    label=f"{'Active' if active_val else 'Passive'}",
                    scatter_kws={"alpha": 1, "s": 25, "edgecolor": "black", "linewidths": 0.5}
                )
            
            # Rasterize scatter points
            for collection in ax.collections:
                if hasattr(collection, 'set_rasterized'):
                    collection.set_rasterized(True)
            
            ax.set_xlabel(metric_x)
            ax.set_ylabel(metric_y)
            ax.set_title(f"{metric_base_x} vs {metric_base_y}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f"Session: {session} | {inter_area[0]} vs {inter_area[1]}", y=1.02)
        plt.tight_layout()
        
        if SAVE_PLOTS:
            output_dir = RESULTS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            area_label = "_".join(inter_area)
            out_name = f"{session}__{area_label}_inter_area.pdf".replace('/', '-')
            fig.savefig(output_dir / out_name, format="pdf", bbox_inches="tight", dpi=300)
            plt.close(fig)
        else:
            plt.show()

# %%
# ------------------------------
# INTRA-AREA - PLOTS PER SESSION
# ------------------------------

# choose areas to show as columns
areas = ['LGd', 'VISp', 'SCMRN']

palette_inter = {False: '#ff7f0e', True: '#1f77b4'}

# Metric pairs to plot - leave space for future additions
metric_pairs = [
    ('TV_emp', 'rbar'),
    # Add more metric pairs here in the future
]

# Normalize TV_emp once for all metric pairs that might need it
df_plot = df_stats.copy()
df_plot['TV_emp'] = df_plot['TV_emp'] / df_plot['N']

# loop over metric pairs
for metric_x, metric_y in metric_pairs:
    # loop over all sessions
    for sess in df_stats["session_id"].unique():
        df_sess = df_plot[(df_plot["session_id"] == sess) & (df_plot["area"].isin(areas)) & (df_plot["N"]>30)]
        if df_sess.empty:
            continue

        g = sns.lmplot(
            data=df_sess,
            x=metric_x,
            y=metric_y,
            hue="active",
            palette=palette_inter,
            col="area",
            col_order=areas,
            scatter_kws={"alpha": 1, "s": 25, "edgecolor": "black", "linewidths": 0.5},
            line_kws={"linewidth": 2},
            height=3,
            aspect=1,
            facet_kws={'sharex': False, 'sharey': False}
        )

        g.figure.suptitle(f"Session {sess}: {metric_y} vs {metric_x}", y=1.05)
        
        # Rasterize scatter points in all facets
        for ax in g.axes.flat:
            for collection in ax.collections:
                if hasattr(collection, 'set_rasterized'):
                    collection.set_rasterized(True)
        
        # annotate each facet - show only slope at bottom right, slightly above x-axis
        for j, area in enumerate(g.col_names):  # columns = areas present
            ax = g.axes[0, j]
            # Get unique active values present in the data for this area
            active_vals = sorted(df_sess[df_sess["area"] == area]["active"].unique(), reverse=True)
            
            slopes_text = []
            for i, active_val in enumerate(active_vals):
                sub = df_sess[(df_sess["area"] == area) & (df_sess["active"] == active_val)]
                if len(sub) < 2:
                    continue

                lr = stats.linregress(sub[metric_x].values, sub[metric_y].values)
                slope = lr.slope
                slopes_text.append(f"m={slope:.3f}")

                print(f"Session {sess} | {area} | active={active_val} | "
                    f"slope={slope:.6f}")
            
            # Position at bottom right, slightly above x-axis - show all slopes stacked
            if slopes_text:
                combined_text = "\n".join(slopes_text)
                ax.text(
                    0.98, 0.05,
                    combined_text,
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.8, alpha=0.9)
                )
        
        if SAVE_PLOTS:
            output_dir = RESULTS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"intra_{sess}_{metric_y}_vs_{metric_x}.pdf".replace('/', '-')
            g.figure.savefig(output_dir / out_name, format="pdf", bbox_inches="tight", dpi=300)
            plt.close(g.figure)
        else:
            plt.show()

# %%
# -------------------------------------------------------------------------------------------------
# BOXPLOT+JOINTPLOT FOR VISUAL CORTEX LAYERS ONLY (GOOD SUPPLEMENTARY PLOT)
# -------------------------------------------------------------------------------------------------

df_ratio = df_stats.copy()
df_ratio = df_ratio[df_ratio.N >= 30]
df_ratio = df_ratio[df_ratio.image_name != 'omitted']
df_ratio = df_ratio[df_ratio.image_set == 'G']
df_ratio['PR_emp'] = df_ratio['PR_emp'] / df_ratio['N']
metrics_ratio = ['PR_emp', 'TR_emp', 'rbar']

visual_cortex_layers = ['VISp', 'VISrl', 'VISl', 'VISal', 'VISpm', 'VISam']

# Filter ratio data for visual cortex only
df_ratio_vis = df_ratio[df_ratio['area'].isin(visual_cortex_layers)].copy()

# Create ratio dataframe for visual cortex
ratio_data_vis = []

# Group keys for computing ratios (finest granularity) and averaging
group_keys_ratio_vis = ['session_id', 'image_name', 'image_set', 'subsample_id', 'area']
group_keys_avg_vis = ['session_id', 'image_name', 'area']  # Keys for averaging ratios

for metric in metrics_ratio:
    # First compute ratios at finest granularity (including subsample_id, image_name, image_set)
    df_pivot_vis = df_ratio_vis.pivot_table(
        index=group_keys_ratio_vis,
        columns='active',
        values=metric,
        aggfunc='mean'
    ).reset_index()
    
    # Check if we have both True and False
    if True not in df_pivot_vis.columns or False not in df_pivot_vis.columns:
        continue
    
    # Compute ratio (avoid division by zero)
    df_pivot_vis['ratio'] = df_pivot_vis[True] / df_pivot_vis[False]
    df_pivot_vis = df_pivot_vis[df_pivot_vis['ratio'].notna() & np.isfinite(df_pivot_vis['ratio'])]
    
    # Average over subsample_id (and image_set); keep (session_id, image_name, area)
    df_pivot_vis_avg = (
        df_pivot_vis[group_keys_avg_vis + ['ratio']]
        .groupby(group_keys_avg_vis, as_index=False)['ratio']
        .mean()
    )
    
    # Add metadata
    df_pivot_vis_avg['metric'] = metric
    
    # Keep only necessary columns
    ratio_data_vis.append(df_pivot_vis_avg[group_keys_avg_vis + ['ratio', 'metric']])


# Combine ratio data for visual cortex
if ratio_data_vis:
    df_ratios_vis = pd.concat(ratio_data_vis, ignore_index=True)

    df_ratios_vis['area'] = pd.Categorical(
        df_ratios_vis['area'],
        categories=visual_cortex_layers,
        ordered=True,
    )
    image_order = sorted(df_ratios_vis['image_name'].astype(str).unique())
    df_ratios_vis['image_name'] = pd.Categorical(
        df_ratios_vis['image_name'],
        categories=image_order,
        ordered=True,
    )

    # Area plots: original blue / light-cyan palette
    area_palette = ['#03045E', '#023E8A', '#0077B6', '#0096C7', '#00B4D8', '#48CAE4']
    area_colors = {
        area: area_palette[i] for i, area in enumerate(visual_cortex_layers)
    }

    # Image plots: water-green only (no shared hex with the area blues)
    image_palette_base = [
        '#0B6E4F', '#148F77', '#1ABC9C', '#2ECC71',
        '#52D1A8', '#7DDEBF', '#A8E6CF', '#C8F0DC',
    ]
    n_images = len(image_order)
    if n_images <= len(image_palette_base):
        image_colors_list = image_palette_base[:n_images]
    else:
        image_colors_list = [
            image_palette_base[i % len(image_palette_base)] for i in range(n_images)
        ]
    image_colors = {name: image_colors_list[i] for i, name in enumerate(image_order)}

    y_min = df_ratios_vis['ratio'].min()
    y_max = df_ratios_vis['ratio'].max()
    y_margin = (y_max - y_min) * 0.1
    y_lim = (y_min - y_margin, y_max + y_margin)

    def _ratio_box_jointplot(
        df_ratios,
        category_col,
        category_order,
        colors,
        xlabel,
        title_by,
        file_stem,
    ):
        for metric in metrics_ratio:
            df_metric_vis = df_ratios[df_ratios['metric'] == metric].copy()
            if df_metric_vis.empty:
                continue

            positions = {name: i for i, name in enumerate(category_order)}
            df_metric_vis['cat_index'] = df_metric_vis[category_col].map(positions).astype(float)
            rng = np.random.default_rng(seed=0)
            df_metric_vis['cat_jitter'] = (
                df_metric_vis['cat_index'] + rng.uniform(-0.1, 0.1, len(df_metric_vis))
            )

            joint_box = sns.jointplot(
                data=df_metric_vis,
                x='cat_jitter',
                y='ratio',
                hue=category_col,
                palette=colors,
                kind='scatter',
                marginal_kws={'fill': True, 'common_norm': False, 'alpha': 0.4},
                height=6,
                alpha=0.25,
                s=10,
            )

            scatter_collections = list(joint_box.ax_joint.collections)

            box_data = []
            box_colors_list = []
            tick_labels = []
            for name in category_order:
                vals = df_metric_vis[df_metric_vis[category_col] == name]['ratio'].values
                if len(vals) > 0:
                    box_data.append(vals)
                    box_colors_list.append(colors[name])
                    tick_labels.append(name)

            if box_data:
                bp = joint_box.ax_joint.boxplot(
                    box_data,
                    positions=range(len(box_data)),
                    widths=0.6,
                    patch_artist=True,
                    showfliers=False,
                    zorder=1,
                )
                for i, (patch, color) in enumerate(zip(bp['boxes'], box_colors_list)):
                    patch.set_facecolor('white')
                    patch.set_edgecolor(color)
                    patch.set_linewidth(1.5)
                    bp['whiskers'][i * 2].set_color(color)
                    bp['whiskers'][i * 2 + 1].set_color(color)
                    bp['whiskers'][i * 2].set_linewidth(1.5)
                    bp['whiskers'][i * 2 + 1].set_linewidth(1.5)
                    bp['caps'][i * 2].set_color(color)
                    bp['caps'][i * 2 + 1].set_color(color)
                    bp['caps'][i * 2].set_linewidth(1.5)
                    bp['caps'][i * 2 + 1].set_linewidth(1.5)
                    bp['medians'][i].set_color(color)
                    bp['medians'][i].set_linewidth(1.5)

            for collection in scatter_collections:
                collection.set_rasterized(True)
                collection.set_edgecolors('none')
                collection.set_linewidths(0)
                collection.set_zorder(3)

            n_xticks = len(tick_labels)
            joint_box.ax_joint.set_xticks(range(n_xticks))
            joint_box.ax_joint.set_xticklabels(tick_labels, rotation=45, ha='right')
            joint_box.ax_joint.set_xlabel(xlabel)
            joint_box.ax_joint.set_ylabel('Active/Passive Ratio')
            joint_box.ax_joint.set_ylim(y_lim)
            joint_box.ax_joint.set_xlim(-0.5, max(n_xticks - 0.5, 0.5))
            joint_box.ax_joint.grid(axis='y', linestyle='--', alpha=0.2)
            joint_box.ax_joint.axhline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            _annotate_one_sample_vs_mu(
                joint_box.ax_joint, df_metric_vis, 'ratio', tick_labels,
                mu=1.0, category_col=category_col,
            )
            _pad_ylim_for_asterisks(joint_box.ax_joint)

            joint_box.ax_marg_x.set_visible(False)
            joint_box.ax_marg_y.set_xlabel('Density')
            joint_box.ax_marg_y.set_ylabel('')
            joint_box.ax_marg_y.grid(axis='y', linestyle='--', alpha=0.2)
            joint_box.ax_marg_y.set_ylim(joint_box.ax_joint.get_ylim())

            joint_box.fig.suptitle(
                f'Visual Cortex: Active/Passive Ratio ({metric}) by {title_by} — Boxplot + Jointplot',
                y=1.02,
            )
            joint_box.fig.tight_layout()

            if SAVE_PLOTS:
                output_dir = RESULTS_DIR
                output_dir.mkdir(parents=True, exist_ok=True)
                joint_box.fig.savefig(
                    output_dir / f"{file_stem}_jointplot_{metric}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close(joint_box.fig)
            else:
                plt.show()

            fig, ax = plt.subplots(figsize=(8, 5))
            if not box_data:
                plt.close(fig)
                continue

            bp = ax.boxplot(
                box_data,
                positions=range(len(box_data)),
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                zorder=2,
            )
            for i, (patch, color) in enumerate(zip(bp['boxes'], box_colors_list)):
                patch.set_facecolor(color)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)
                patch.set_alpha(0.7)
            for i in range(len(box_data)):
                bp['whiskers'][i * 2].set_color('black')
                bp['whiskers'][i * 2 + 1].set_color('black')
                bp['whiskers'][i * 2].set_linewidth(1.5)
                bp['whiskers'][i * 2 + 1].set_linewidth(1.5)
                bp['caps'][i * 2].set_color('black')
                bp['caps'][i * 2 + 1].set_color('black')
                bp['caps'][i * 2].set_linewidth(1.5)
                bp['caps'][i * 2 + 1].set_linewidth(1.5)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.5)
            for flier in bp['fliers']:
                flier.set_markerfacecolor('white')
                flier.set_markeredgecolor('black')
                flier.set_markersize(5)
                flier.set_alpha(0.8)
                flier.set_marker('o')

            ax.set_xticks(range(len(box_data)))
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Active/Passive Ratio')
            ax.set_ylim(y_lim)
            ax.set_xlim(-0.5, len(box_data) - 0.5)
            ax.grid(axis='y', linestyle='--', alpha=0.2)
            ax.axhline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            _annotate_one_sample_vs_mu(
                ax, df_metric_vis, 'ratio', tick_labels,
                mu=1.0, category_col=category_col,
            )
            _pad_ylim_for_asterisks(ax)
            fig.suptitle(
                f'Visual Cortex: Active/Passive Ratio ({metric}) by {title_by} — Boxplot',
                y=1.02,
            )
            fig.tight_layout()

            if SAVE_PLOTS:
                output_dir = RESULTS_DIR
                output_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    output_dir / f"{file_stem}_boxplot_{metric}.pdf",
                    format="pdf",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close(fig)
            else:
                plt.show()

    _ratio_box_jointplot(
        df_ratios_vis,
        category_col='area',
        category_order=visual_cortex_layers,
        colors=area_colors,
        xlabel='Area',
        title_by='area',
        file_stem='ratio_box_visual_cortex_by_area',
    )
    _ratio_box_jointplot(
        df_ratios_vis,
        category_col='image_name',
        category_order=image_order,
        colors=image_colors,
        xlabel='Image',
        title_by='image',
        file_stem='ratio_box_visual_cortex_by_image',
    )


# %%
