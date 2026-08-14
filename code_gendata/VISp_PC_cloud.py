# Works correctly
# %%
# --------------------------------------
# COMPUTATION OF ''FIRST-ORDER'' RESULTS
# --------------------------------------
from pathlib import Path
from itertools import combinations
import re

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# %%
# ----------------------------
# NEURAL ACTIVITY POINT CLOUDS
# ----------------------------

# Load Dataset
"""
Session number: Indicates whether a session was the first or second recording day for a given mouse.
Image set: Indicates whether image set G or H was shown during a particular session.
Experience level: Indicates whether the image set shown was familiar to the mouse from training or novel.
"""

ALLEN_DATA_DIR = Path('/data/allen_engagement_data')
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(51)
random.seed(51)

# --- Load Sessions ---
# Day-1 familiar sessions (image sets G and H).
df_sessions = pd.read_pickle(ALLEN_DATA_DIR / 'df_sessions.pk')
df_sessions = df_sessions.query(
    'session_number==1 and experience_level=="Familiar"'
)
print(f"Number of filtered sessions: {len(df_sessions)}")
unique_session_ids = df_sessions.ecephys_session_id.unique()

# ----------------
# DEFINE FUNCTIONS
# ----------------
def stack_groups(df):
    """
    Stack the spikecounts for each group defined by (session_id, image_name, active).

    Parameters:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        pd.DataFrame: DataFrame where each row corresponds to a group with a stacked spikecounts matrix.
    """
    df_rows = []
    for group_name, df_group in df.groupby(['session_id', 'image_name', 'active']):
        session_id, image_name, active = group_name
        stacked_spikecounts = np.vstack(df_group.spikecounts.values)
        df_rows.append({'session_id': session_id, 'image_name': image_name, 'active': active,
                        'stacked_spikecounts': stacked_spikecounts})
    return pd.DataFrame(df_rows)

def subsample_matrix(mat, n_samples=200):
    """
    Subsample rows of a matrix by randomly choosing n_samples entries per column.
    
    Parameters:
        mat (np.array): 2D array with shape (trials, neurons).
        n_samples (int): Number of samples (rows) to select.
    
    Returns:
        np.array: Matrix of shape (n_samples, neurons) with subsampled values.
    
    Raises:
        ValueError: If mat has fewer rows than n_samples.
    """
    M, N = mat.shape
    if M < n_samples:
        raise ValueError(f"Number of trials/rows ({M}) is less than subsample number ({n_samples}).")
    return np.column_stack([np.random.choice(mat[:, j], size=n_samples, replace=False) for j in range(N)])

def apply_pca(data_matrix, use_subset=False, subset_fraction=0.2, n_components=3):
    """
    Apply PCA on the provided data_matrix.

    Optionally, a random subset of neurons (columns) can be selected before applying PCA.

    Parameters:
        data_matrix (np.array): The input matrix (samples x neurons).
        use_subset (bool): Whether to use a subset of neurons.
        subset_fraction (float): Fraction of neurons to select if use_subset is True.
        n_components (int): Number of principal components.

    Returns:
        np.array: The transformed data (samples x n_components).
    """
    total_neurons = data_matrix.shape[1]
    if use_subset:
        subset_size = int(total_neurons * subset_fraction)
        if subset_size < 1:
            raise ValueError("Number of neurons < 1. Increase the total number of neurons.")
        subset_neurons = np.random.choice(total_neurons, size=subset_size, replace=False)
        data_matrix = data_matrix[:, subset_neurons]
        print(f"Using subset of neurons: {subset_size} out of {total_neurons}")
    else:
        print(f"Using all available neurons: {total_neurons}")
    solver_pca = PCA(n_components=n_components)
    return solver_pca.fit_transform(data_matrix)

def print_cloud_info(condition_name, group_labels):
    """Print the number and labels of the clouds in the current analysis."""
    print(f"{condition_name} condition: plotting {len(group_labels)} clouds")
    for label in group_labels:
        print(f"  - {label}")

def plot_3d_pca(pcs_3d, group_data_list, title, colors, filename):
    """Plot the 3D PCA clouds and save the usual rasterized PDF."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    start_idx = 0

    for group, color in zip(group_data_list, colors):
        n_points = group.shape[0]
        pc_slice = pcs_3d[start_idx:start_idx + n_points, :]
        ax.scatter(
            pc_slice[:, 0],
            pc_slice[:, 1],
            pc_slice[:, 2],
            color=color,
            alpha=0.7,
            s=10,
            rasterized=True,
        )
        start_idx += n_points

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.set_title(title)
    fig.savefig(RESULTS_DIR / filename, format='pdf')
    plt.show()
    plt.close(fig)


def save_3d_pca_clouds_svg(pcs_3d, group_data_list, colors, filename):
    """
    Save the same already-computed PCA clouds as a transparent vector SVG.

    The SVG contains only the clouds: no axes, grid, labels, title, frame, or
    background. No PCA or subsampling is repeated here.
    """
    fig = plt.figure(figsize=(12, 10), frameon=False)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((0, 0, 0, 0))

    start_idx = 0
    for group, color in zip(group_data_list, colors):
        n_points = group.shape[0]
        pc_slice = pcs_3d[start_idx:start_idx + n_points, :]
        ax.scatter(
            pc_slice[:, 0],
            pc_slice[:, 1],
            pc_slice[:, 2],
            color=color,
            alpha=0.7,
            s=10,
            edgecolors='none',
            depthshade=True,
            rasterized=False,
        )
        start_idx += n_points

    ax.set_axis_off()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_position([0, 0, 1, 1])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.savefig(
        RESULTS_DIR / filename,
        format='svg',
        transparent=True,
        bbox_inches='tight',
        pad_inches=0,
    )
    plt.close(fig)

# Color palette
# First image uses the requested base colors; the second image uses darker,
# muted variants that remain easy to distinguish without being overly bright.
ACTIVE_COLORS = ["#80B3FF", "#4F86C6"]
PASSIVE_COLORS = ["#FF8080", "#C95F6A"]
ACTIVE_BASELINE_COLORS = ["#C9DDF7", "#A8C4E8"]
PASSIVE_BASELINE_COLORS = ["#F4C6C6", "#E7B0B5"]


def _image_index(image, selected_images):
    """Return 0 or 1 according to the image's position in the current pair."""
    return selected_images.index(image)


def _safe_filename_part(value):
    """Convert an image label into a filesystem-safe filename component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "image"


def combine_groups_stimulus_only(df_list, selected_images):
    """
    Combine stimulus-only groups for one unordered image pair.

    Active clouds use blue shades and passive clouds use red shades. The first
    image receives the requested base colors, while the second receives a
    contrasting, moderately darker version of each color.
    """
    df_all = pd.concat(df_list, ignore_index=True)
    groups = (
        df_all.groupby(['image_name', 'active'], sort=False)
        ['subsampled_stacked_spikecounts']
        .apply(list)
        .reset_index()
    )

    # Preserve the order of selected_images, then active status, so colors and
    # labels are deterministic across runs.
    groups['image_order'] = groups['image_name'].map(
        {image: idx for idx, image in enumerate(selected_images)}
    )
    groups = groups.sort_values(['image_order', 'active']).reset_index(drop=True)

    combined_data = []
    labels = []
    colors = []
    for _, row in groups.iterrows():
        image = row['image_name']
        active = bool(row['active'])
        image_idx = _image_index(image, selected_images)
        combined_data.append(
            np.concatenate(row['subsampled_stacked_spikecounts'], axis=0)
        )
        labels.append(f"{image}_active_{active}")
        colors.append(
            ACTIVE_COLORS[image_idx] if active else PASSIVE_COLORS[image_idx]
        )

    return combined_data, labels, colors


def combine_groups_with_condition(df_list, selected_images, condition_label):
    """
    Combine stimulus and baseline groups for either the active or passive case.

    Stimulus clouds use the main active/passive colors. Baselines use pastel,
    muted versions of the matching condition color.
    """
    nonempty_frames = [df for df in df_list if not df.empty]
    if not nonempty_frames:
        return [], [], []

    df_all = pd.concat(nonempty_frames, ignore_index=True)
    groups = (
        df_all.groupby(['image_name', 'cond'], sort=False)
        ['subsampled_stacked_spikecounts']
        .apply(list)
        .reset_index()
    )
    groups['image_order'] = groups['image_name'].map(
        {image: idx for idx, image in enumerate(selected_images)}
    )
    groups['cond_order'] = groups['cond'].map({'stimulus': 0, 'baseline': 1})
    groups = groups.sort_values(['image_order', 'cond_order']).reset_index(drop=True)

    combined_data = []
    labels = []
    colors = []
    for _, row in groups.iterrows():
        image = row['image_name']
        cond = row['cond']
        image_idx = _image_index(image, selected_images)
        combined_data.append(
            np.concatenate(row['subsampled_stacked_spikecounts'], axis=0)
        )
        labels.append(f"{image}_{condition_label}_{cond}")

        if condition_label == "active":
            palette = ACTIVE_COLORS if cond == "stimulus" else ACTIVE_BASELINE_COLORS
        elif condition_label == "passive":
            palette = PASSIVE_COLORS if cond == "stimulus" else PASSIVE_BASELINE_COLORS
        else:
            raise ValueError("condition_label must be either 'active' or 'passive'.")
        colors.append(palette[image_idx])

    return combined_data, labels, colors


def _subsample_group_dataframe(df, n_samples, group_description):
    """Subsample each stacked matrix, with a useful error message on failure."""
    if df.empty:
        return df
    df = df.copy()
    try:
        df['subsampled_stacked_spikecounts'] = df['stacked_spikecounts'].apply(
            lambda x: subsample_matrix(x, n_samples=n_samples)
        )
    except ValueError as exc:
        raise ValueError(f"{group_description}: {exc}") from exc
    return df


def _run_and_plot_case(
    combined,
    labels,
    colors,
    title,
    pdf_filename,
    subset_flag=False,
    svg_flag=False,
):
    """Compute PCA once, then optionally save both PDF and SVG renderings."""
    if not combined:
        print(f"Skipping {title}: no clouds are available.")
        return

    print(f"Clouds: {labels}")
    combined_data = np.concatenate(combined, axis=0)
    pcs_3d = apply_pca(
        combined_data,
        use_subset=subset_flag,
        subset_fraction=0.2,
        n_components=3,
    )

    # Both outputs use these exact PCA coordinates and cloud boundaries.
    plot_3d_pca(pcs_3d, combined, title, colors, pdf_filename)

    if svg_flag:
        svg_filename = str(Path(pdf_filename).with_suffix('.svg'))
        save_3d_pca_clouds_svg(
            pcs_3d,
            combined,
            colors,
            svg_filename,
        )
        print(f"Saved SVG clouds: {RESULTS_DIR / svg_filename}")


def process_pca_two(session_id, svg_flag=False):
    """
    Generate neural-cloud PCA plots for every unique unordered pair of images.

    For N available images, this processes N * (N - 1) / 2 pairs. Therefore,
    (img_A, img_B) and (img_B, img_A) are considered the same pair.

    When svg_flag is True, each case is also saved as a transparent SVG that
    contains only the clouds. The SVG reuses the exact PCA coordinates used
    for the corresponding PDF; PCA and subsampling are not recomputed.
    """
    session_str = str(session_id)
    print(f"Processing (filtered) session: {session_str}")

    base_dir = ALLEN_DATA_DIR / 'day1'
    activity_file = base_dir / session_str / 'df_activity.pk'
    units_file = base_dir / session_str / 'df_units.pk'

    try:
        df_activity = pd.read_pickle(activity_file)
        df_units = pd.read_pickle(units_file)
    except FileNotFoundError as exc:
        print(f"Session {session_str}: {exc}")
        return

    df_activity['session_id'] = session_str
    df_units['session_id'] = session_str

    df_stimulus = df_activity.loc[
        (df_activity.is_viewing == True)
        & (df_activity.time_from_stimulus >= 0.1)
        & (df_activity.is_image_novel == False)
        & (df_activity.is_change == False)
    ].copy()
    df_baseline = df_activity.loc[
        (df_activity.is_viewing == False)
        & (df_activity.time_from_stimulus > 0)
        & (df_activity.time_from_stimulus < 0.45)
        & (df_activity.is_image_novel == False)
        & (df_activity.is_change == False)
        & (df_activity.is_next_image_same == True)
        & (df_activity.is_next_block_same == True)
    ].copy()

    vis_areas = ['VISp']
    df_units_vis = df_units[
        (df_units.structure_acronym.isin(vis_areas))
        & (df_units.structure_acronym != 'VIS')
    ]
    if df_units_vis.shape[0] < 100:
        print(f"Not enough neurons for session {session_str}. Skipping.")
        return

    idx_vis = df_units_vis.index.values
    df_stimulus['spikecounts'] = df_stimulus['spikecounts'].apply(lambda x: x[idx_vis])
    df_baseline['spikecounts'] = df_baseline['spikecounts'].apply(lambda x: x[idx_vis])

    df_stimulus_stacked = stack_groups(df_stimulus)
    df_baseline_stacked = stack_groups(df_baseline)

    available_images = sorted(
        df_stimulus_stacked.loc[
            df_stimulus_stacked['session_id'] == session_str, 'image_name'
        ].unique(),
        key=str,
    )
    if len(available_images) < 2:
        print(f"Session {session_str} has fewer than 2 images. Skipping analysis.")
        return

    image_pairs = list(combinations(available_images, 2))
    print(
        f"Session {session_str}: processing {len(image_pairs)} unique image pairs "
        f"from {len(available_images)} images."
    )

    n_samples = 1000
    subset_flag = False

    for pair_number, pair in enumerate(image_pairs, start=1):
        selected_images = list(pair)
        image_1, image_2 = selected_images
        print("\n" + "=" * 80)
        print(
            f"Session {session_str} | Pair {pair_number}/{len(image_pairs)}: "
            f"({image_1}, {image_2})"
        )

        df_stim_1 = df_stimulus_stacked.loc[
            (df_stimulus_stacked['session_id'] == session_str)
            & (df_stimulus_stacked['image_name'] == image_1)
        ].copy()
        df_base_1 = df_baseline_stacked.loc[
            (df_baseline_stacked['session_id'] == session_str)
            & (df_baseline_stacked['image_name'] == image_1)
        ].copy()
        df_stim_2 = df_stimulus_stacked.loc[
            (df_stimulus_stacked['session_id'] == session_str)
            & (df_stimulus_stacked['image_name'] == image_2)
        ].copy()
        df_base_2 = df_baseline_stacked.loc[
            (df_baseline_stacked['session_id'] == session_str)
            & (df_baseline_stacked['image_name'] == image_2)
        ].copy()

        try:
            df_stim_1 = _subsample_group_dataframe(df_stim_1, n_samples, f"stimulus {image_1}")
            df_base_1 = _subsample_group_dataframe(df_base_1, n_samples, f"baseline {image_1}")
            df_stim_2 = _subsample_group_dataframe(df_stim_2, n_samples, f"stimulus {image_2}")
            df_base_2 = _subsample_group_dataframe(df_base_2, n_samples, f"baseline {image_2}")
        except ValueError as exc:
            print(f"Skipping pair ({image_1}, {image_2}): {exc}")
            continue

        stim_combined, stim_labels, stim_colors = combine_groups_stimulus_only(
            [df_stim_1, df_stim_2], selected_images
        )

        active_frames = []
        passive_frames = []
        for stim_df, base_df in [(df_stim_1, df_base_1), (df_stim_2, df_base_2)]:
            stim_active = stim_df[stim_df.active == True].copy()
            stim_active['cond'] = 'stimulus'
            base_active = base_df[base_df.active == True].copy()
            base_active['cond'] = 'baseline'
            active_frames.extend([stim_active, base_active])

            stim_passive = stim_df[stim_df.active == False].copy()
            stim_passive['cond'] = 'stimulus'
            base_passive = base_df[base_df.active == False].copy()
            base_passive['cond'] = 'baseline'
            passive_frames.extend([stim_passive, base_passive])

        active_combined, active_labels, active_colors = combine_groups_with_condition(
            active_frames, selected_images, condition_label='active'
        )
        passive_combined, passive_labels, passive_colors = combine_groups_with_condition(
            passive_frames, selected_images, condition_label='passive'
        )

        print_cloud_info("Stimulus Only", stim_labels)
        print_cloud_info("Active Only", active_labels)
        print_cloud_info("Passive Only", passive_labels)

        pair_text = f"{image_1} vs {image_2}"
        pair_file = f"{_safe_filename_part(image_1)}_vs_{_safe_filename_part(image_2)}"

        _run_and_plot_case(
            stim_combined,
            stim_labels,
            stim_colors,
            f"3D PCA (Stimulus Only, Session {session_str}, {pair_text})",
            f"pca_3d_stimulus_only_session_{session_str}_{pair_file}.pdf",
            subset_flag,
            svg_flag,
        )
        _run_and_plot_case(
            active_combined,
            active_labels,
            active_colors,
            f"3D PCA (Active Only, Session {session_str}, {pair_text})",
            f"pca_3d_active_only_session_{session_str}_{pair_file}.pdf",
            subset_flag,
            svg_flag,
        )

        _run_and_plot_case(
            passive_combined,
            passive_labels,
            passive_colors,
            f"3D PCA (Passive Only, Session {session_str}, {pair_text})",
            f"pca_3d_passive_only_session_{session_str}_{pair_file}.pdf",
            subset_flag,
            svg_flag,
        )

# %%
# ----------------------
# GENERATE NEURAL CLOUDS
# ----------------------
# for sess in unique_session_ids:
#     process_pca_two(sess)
# %%
# 1139846596
# 1086200042
process_pca_two(1086200042, svg_flag=False)
# %%