# Works correctly
# %%
# --------------------------------------
# COMPUTATION OF ''FIRST-ORDER'' RESULTS
# --------------------------------------
"""
PCA neural-activity clouds for circuit areas (no baselines).

Same cohort and stimulus-window filters as VISp_PC_cloud.py. Units are grouped
into circuit labels (not always Allen acronyms):

  LGd   = {LGd}
  VISpl = {VISp, VISrl, VISl, VISal}   # primary + lateral
  VISm  = {VISpm, VISam}               # medial
  SCMRN = {MRN, SCig, SCiw}

For each session, area, and image, one 3D PCA plot shows the stimulus-evoked
active and passive clouds. Figures are written to
clean_scripts/results/circuit_clouds/<area>/.
"""
from pathlib import Path
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
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'circuit_clouds'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(51)
random.seed(51)

# Label, then the structure acronyms that belong to that label.
GROUP_DEFS = [
    ('LGd', {'LGd'}),
    ('VISpl', {'VISp', 'VISrl', 'VISl', 'VISal'}),
    ('VISm', {'VISpm', 'VISam'}),
    ('SCMRN', {'MRN', 'SCig', 'SCiw'}),
]
MIN_UNITS_PER_AREA = 30

ACTIVE_COLOR = "#2A7FFF"
PASSIVE_COLOR = "#AA0000"

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


def _area_results_dir(area_name):
    """Return (and create) the nested results folder for one circuit area."""
    area_dir = RESULTS_DIR / _safe_filename_part(area_name)
    area_dir.mkdir(parents=True, exist_ok=True)
    return area_dir


def plot_3d_pca(pcs_3d, group_data_list, title, colors, filepath):
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
    fig.savefig(filepath, format='pdf')
    plt.close(fig)


def save_3d_pca_clouds_svg(pcs_3d, group_data_list, colors, filepath):
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
        filepath,
        format='svg',
        transparent=True,
        bbox_inches='tight',
        pad_inches=0,
    )
    plt.close(fig)


def _safe_filename_part(value):
    """Convert a label into a filesystem-safe filename component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "image"


def combine_groups_single_image(df_image):
    """
    Combine active and passive stimulus groups for one image.

    Active is plotted first (blue), then passive (red).
    """
    groups = (
        df_image.groupby(['active'], sort=False)
        ['subsampled_stacked_spikecounts']
        .apply(list)
        .reset_index()
    )
    groups['active_order'] = groups['active'].map({True: 0, False: 1})
    groups = groups.sort_values('active_order').reset_index(drop=True)

    combined_data = []
    labels = []
    colors = []
    for _, row in groups.iterrows():
        active = bool(row['active'])
        combined_data.append(
            np.concatenate(row['subsampled_stacked_spikecounts'], axis=0)
        )
        labels.append(f"active_{active}")
        colors.append(ACTIVE_COLOR if active else PASSIVE_COLOR)

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
    pdf_path,
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
    plot_3d_pca(pcs_3d, combined, title, colors, pdf_path)

    if svg_flag:
        svg_path = pdf_path.with_suffix('.svg')
        save_3d_pca_clouds_svg(
            pcs_3d,
            combined,
            colors,
            svg_path,
        )
        print(f"Saved SVG clouds: {svg_path}")


def process_pca_area(
    session_id,
    df_stimulus,
    df_units,
    area_name,
    acronyms,
    svg_flag=False,
):
    """
    Generate active/passive PCA clouds for one circuit area and every image.
    """
    session_str = str(session_id)
    df_units_area = df_units[df_units.structure_acronym.isin(acronyms)]
    n_units = df_units_area.shape[0]
    if n_units < MIN_UNITS_PER_AREA:
        print(
            f"Not enough neurons in {area_name} for session {session_str} "
            f"({n_units} < {MIN_UNITS_PER_AREA}). Skipping."
        )
        return

    print(f"\n{'#' * 80}")
    print(f"Area {area_name}: {n_units} neurons (session {session_str})")
    print(f"{'#' * 80}")

    idx_area = df_units_area.index.values
    df_stimulus_area = df_stimulus.copy()
    df_stimulus_area['spikecounts'] = df_stimulus_area['spikecounts'].apply(
        lambda x: x[idx_area]
    )
    df_stimulus_stacked = stack_groups(df_stimulus_area)

    available_images = sorted(
        df_stimulus_stacked.loc[
            df_stimulus_stacked['session_id'] == session_str, 'image_name'
        ].unique(),
        key=str,
    )
    if len(available_images) < 1:
        print(f"Session {session_str} ({area_name}) has no images. Skipping.")
        return

    print(
        f"Session {session_str} ({area_name}): processing "
        f"{len(available_images)} images."
    )

    n_samples = 1000
    subset_flag = False
    area_dir = _area_results_dir(area_name)

    for image_number, image_name in enumerate(available_images, start=1):
        print("\n" + "=" * 80)
        print(
            f"Session {session_str} | {area_name} | Image "
            f"{image_number}/{len(available_images)}: {image_name}"
        )

        df_stim = df_stimulus_stacked.loc[
            (df_stimulus_stacked['session_id'] == session_str)
            & (df_stimulus_stacked['image_name'] == image_name)
        ].copy()

        try:
            df_stim = _subsample_group_dataframe(
                df_stim, n_samples, f"{area_name} stimulus {image_name}"
            )
        except ValueError as exc:
            print(f"Skipping image {image_name} in {area_name}: {exc}")
            continue

        stim_combined, stim_labels, stim_colors = combine_groups_single_image(
            df_stim
        )

        print_cloud_info(f"{area_name} {image_name}", stim_labels)

        image_file = _safe_filename_part(image_name)
        pdf_path = area_dir / (
            f"pca_3d_stimulus_session_{session_str}_{image_file}.pdf"
        )

        _run_and_plot_case(
            stim_combined,
            stim_labels,
            stim_colors,
            (
                f"3D PCA ({area_name}, Session {session_str}, {image_name}: "
                f"active vs passive)"
            ),
            pdf_path,
            subset_flag,
            svg_flag,
        )


def process_pca_circuit(session_id, svg_flag=False):
    """
    Generate active/passive neural-cloud PCA plots for every circuit area.

    One plot per image: the stimulus-evoked active cloud and the matching
    passive cloud, projected onto a joint 3-PC space.

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

    for area_name, acronyms in GROUP_DEFS:
        process_pca_area(
            session_id,
            df_stimulus,
            df_units,
            area_name,
            acronyms,
            svg_flag=svg_flag,
        )

# %%
# ----------------------
# GENERATE NEURAL CLOUDS
# ----------------------
for sess in unique_session_ids:
    process_pca_circuit(sess)
# %%
