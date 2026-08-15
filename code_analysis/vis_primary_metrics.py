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
SAVE_PLOTS = False


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

# Compute volume_10PC from eigenvalues_95 (first 10 eigenvalues)
# Eigenvalues are stored as [array(...)], so extract array first
# Volume of ellipsoid is proportional to product of square roots of eigenvalues
df_geometry['volume_10PC'] = df_geometry['eigenvalues_95'].apply(
    lambda eig: np.sqrt(np.prod(eig[0][:3])) if len(eig) > 0 and len(eig[0]) >= 3 else np.nan
)

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
# %%
# ---------------------------------------------------------------
# SPAGHETTI PLOTS: CHERRY-PICKED SESSION
# ---------------------------------------------------------------
# Create spaghetti plots for cherry-picked session_id='1139846596'

metric_names = ['rbar', 'PR_emp', 'TR_emp', 'volume_10PC', 'cloud_separation']
metric_labels = {
    'rbar': 'Mean Firing Rate (rbar)',
    'PR_emp': 'Participation Ratio (PR_emp)',
    'TR_emp': 'Trial Ratio (TR_emp)',
    'volume_10PC': 'Volume (10PC)',
    'cloud_separation': 'Cloud Separation'
}
x_positions = {True: 0, False: 1}  # Active first, then passive

cherry_picked_session = '1139846596'
df_session_cherry = df_metrics_images[df_metrics_images['session_id'] == cherry_picked_session].copy()

if not df_session_cherry.empty:
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 6), sharey=True)
    if len(metric_names) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metric_names):
        df_metric = df_session_cherry[df_session_cherry[metric].notna()].copy()
        
        if df_metric.empty:
            ax.set_visible(False)
            continue
        
        # Pivot to get active and passive values side by side
        pivot = df_metric.pivot_table(
            index='image_name',
            columns='active',
            values=metric,
            aggfunc='mean'
        )
        
        # Normalize: divide each value by the average of active + passive for that image
        if False in pivot.columns and True in pivot.columns:
            pivot = pivot.dropna(subset=[False, True])
            pivot_mean = (pivot[False] + pivot[True]) / 2
            pivot_normalized = pivot.div(pivot_mean, axis=0)
            
            # Add reference line at y=1 (lower layer)
            ax.axhline(1, color='black', linestyle='--', linewidth=0.8, zorder=0)
            
            # Plot lines connecting active and passive for each image
            for _, row in pivot_normalized.iterrows():
                ax.plot(
                    [x_positions[True], x_positions[False]],
                    [row[True], row[False]],
                    color='black',
                    linewidth=1.0,
                    alpha=0.65,
                    zorder=2
                )
            
            # Plot scatter points for active and passive (in order: active first, then passive)
            for active_state in [True, False]:
                # Get normalized values for this active state
                normalized_values = []
                for img_name in pivot_normalized.index:
                    normalized_values.append(pivot_normalized.loc[img_name, active_state])
                
                ax.scatter(
                    np.full(len(normalized_values), x_positions[active_state]),
                    normalized_values,
                    s=70,
                    facecolors='white',
                    edgecolors='black',
                    linewidths=0.8,
                    alpha=0.7,
                    zorder=3,
                    rasterized=False
                )
        else:
            ax.set_visible(False)
            continue
        
        # Set labels and formatting
        ax.set_xticks([x_positions[True], x_positions[False]])
        ax.set_xticklabels(['Active', 'Passive'])
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(metric_labels[metric])
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylabel(f"{metric_labels[metric]} (Normalized)")
    
    fig.suptitle(f"Session {cherry_picked_session}: Active vs Passive Trajectories (Cherry-Picked)", y=0.98)
    fig.tight_layout()
    
    if SAVE_PLOTS:
        output_dir = RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"spaghetti_metrics_{cherry_picked_session}_cherrypicked.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=300
        )
        plt.close(fig)
    else:
        plt.show()
# %%





















































# %%
# ---------------------------------------------------------------
# SPAGHETTI PLOTS: ACTIVE VS PASSIVE PER METRIC PER SESSION
# ---------------------------------------------------------------
# Create spaghetti plots showing active vs passive for each metric
# One subplot per metric, lines connect active to passive for each image

metric_names = ['rbar', 'PR_emp', 'TR_emp', 'volume_10PC', 'cloud_separation']
metric_labels = {
    'rbar': 'Mean Firing Rate (rbar)',
    'PR_emp': 'Participation Ratio (PR_emp)',
    'TR_emp': 'Trial Ratio (TR_emp)',
    'volume_10PC': 'Volume (10PC)',
    'cloud_separation': 'Cloud Separation'
}
x_positions = {True: 0, False: 1}  # Active first, then passive

# Get unique session IDs
session_ids = sorted(df_metrics_images['session_id'].unique())

for session_id in session_ids:
    df_session = df_metrics_images[df_metrics_images['session_id'] == session_id].copy()
    
    if df_session.empty:
        continue
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 6), sharey=True)
    if len(metric_names) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metric_names):
        df_metric = df_session[df_session[metric].notna()].copy()
        
        if df_metric.empty:
            ax.set_visible(False)
            continue
        
        # Pivot to get active and passive values side by side
        pivot = df_metric.pivot_table(
            index='image_name',
            columns='active',
            values=metric,
            aggfunc='mean'
        )
        
        # Normalize: divide each value by the average of active + passive for that image
        if False in pivot.columns and True in pivot.columns:
            pivot = pivot.dropna(subset=[False, True])
            pivot_mean = (pivot[False] + pivot[True]) / 2
            pivot_normalized = pivot.div(pivot_mean, axis=0)
            
            # Add reference line at y=1 (lower layer)
            ax.axhline(1, color='black', linestyle='--', linewidth=0.8, zorder=0)
            
            # Plot lines connecting active and passive for each image
            for _, row in pivot_normalized.iterrows():
                ax.plot(
                    [x_positions[True], x_positions[False]],
                    [row[True], row[False]],
                    color='black',
                    linewidth=1.0,
                    alpha=0.65,
                    zorder=2
                )
            
            # Plot scatter points for active and passive (in order: active first, then passive)
            for active_state in [True, False]:
                # Get normalized values for this active state
                normalized_values = []
                for img_name in pivot_normalized.index:
                    normalized_values.append(pivot_normalized.loc[img_name, active_state])
                
                ax.scatter(
                    np.full(len(normalized_values), x_positions[active_state]),
                    normalized_values,
                    s=70,
                    facecolors='white',
                    edgecolors='black',
                    linewidths=0.8,
                    alpha=0.7,
                    zorder=3,
                    rasterized=True
                )
        else:
            ax.set_visible(False)
            continue
        
        # Set labels and formatting
        ax.set_xticks([x_positions[True], x_positions[False]])
        ax.set_xticklabels(['Active', 'Passive'])
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(metric_labels[metric])
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylabel(f"{metric_labels[metric]} (Normalized)")
    
    fig.suptitle(f"Session {session_id}: Active vs Passive Trajectories", y=0.98)
    fig.tight_layout()
    
    if SAVE_PLOTS:
        output_dir = RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"spaghetti_metrics_{session_id}.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=300
        )
        plt.close(fig)
    else:
        plt.show()