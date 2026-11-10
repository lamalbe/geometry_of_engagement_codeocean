# %%
import numpy as np
import pandas as pd
import scipy.linalg as la
import itertools
import multiprocessing

import sys
from pathlib import Path

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

import config

ALLEN_DATA_DIR = config.ALLEN_PREPROCESSED_DIR
# %%
# --------------------------------------------------------------------
# DEFINE FUNCTIONS TO COMPUTE GEOMETRY METRICS (DIMENSIONALITY, VOLUME, EXTENSION)
# --------------------------------------------------------------------

def statsvalues2PR(mii, mij, sii, sij, N=10**4):
    """
    Calculate the participation ratio (PR) using covariance statistics.

    Parameters:
        mii (float): Mean of the diagonal of the covariance matrix.
        mij (float): Mean of the off-diagonal elements.
        sii (float): Standard deviation of the diagonal elements.
        sij (float): Standard deviation of the off-diagonal elements.
        N (int): A scaling parameter (default: 10**4).

    Returns:
        float: The computed participation ratio (PR).
    """
    PR = 1 / (1/N + 1/N * (sii / mii)**2 + (N-1) / N * (sij/mii)**2 + (N-1) / N * (mij / mii)**2)
    return PR

def geometry_stats(spkcounts):
    """
    Compute geometry statistics including dimensionality, volume, and extension from spike count data.

    Parameters:
        spkcounts (np.array): A 2D array where each row corresponds to a neuron's spike counts.

    Returns:
        dict: A dictionary containing:
            - 'PR_emp': empirical participation ratio from the covariance matrix
            - 'PR_fit': fitted participation ratio based on statsvalues2PR
            - 'dimensionality': participation ratio as a measure of effective dimensionality
            - 'volume': volume of the activity cloud (sqrt of determinant of covariance)
            - 'extension': largest eigenvalue (spread along principal direction)
            - 'dim_70', 'dim_80', 'dim_90', 'dim_95': number of dimensions explaining variance thresholds
            - 'eigenvalues_70', 'eigenvalues_80', 'eigenvalues_90', 'eigenvalues_95': lists of eigenvalues
            - 'volume_70', 'volume_80', 'volume_90', 'volume_95': volumes for each threshold
            - 'mii': mean diagonal element
            - 'mij': mean off-diagonal element
            - 'sii': standard deviation of the diagonal
            - 'sij': standard deviation of the off-diagonal
    """
    C = np.cov(spkcounts)
    if np.sum(C) == 0 or np.sum(spkcounts) == 0:
        return {}

    ofdiag = np.triu(np.ones_like(C, dtype=bool), k=1)
    ondiag = np.eye(C.shape[0], dtype=bool)

    N = C.shape[0]
    mii = C[ondiag].mean()
    mij = C[ofdiag].mean()
    sii = np.sqrt(1 / (N - 1) * np.sum((C[ondiag] - mii) ** 2))
    sij = np.sqrt(1 / (N * (N - 1) - 2) * np.sum((C[ofdiag] - mij) ** 2))

    # Compute PR_emp (participation ratio)
    PR_emp = (np.trace(C)) ** 2 / (np.trace(C @ C))
    PR_fit = statsvalues2PR(mii, mij, sii, sij)

    # Compute eigenvalues for volume and extension
    eigenvalues = la.eigh(C)[0]
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
    eigenvalues = eigenvalues[eigenvalues > 0]  # Only positive eigenvalues

    if len(eigenvalues) == 0:
        return {}

    # PCA spectrum: compute explained variance and cumulative explained variance
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues / total_variance
    cum_explained_variance = np.cumsum(explained_variance)

    # Find dimensions explaining 70%, 80%, 90%, 95% variance
    dim_70 = np.where(cum_explained_variance > 0.7)[0]
    dim_80 = np.where(cum_explained_variance > 0.8)[0]
    dim_90 = np.where(cum_explained_variance > 0.9)[0]
    dim_95 = np.where(cum_explained_variance > 0.95)[0]

    dim_70_idx = dim_70[0] if len(dim_70) > 0 else len(eigenvalues) - 1
    dim_80_idx = dim_80[0] if len(dim_80) > 0 else len(eigenvalues) - 1
    dim_90_idx = dim_90[0] if len(dim_90) > 0 else len(eigenvalues) - 1
    dim_95_idx = dim_95[0] if len(dim_95) > 0 else len(eigenvalues) - 1

    # Get eigenvalues explaining up to each threshold (inclusive)
    eig_70 = eigenvalues[:dim_70_idx + 1]
    eig_80 = eigenvalues[:dim_80_idx + 1]
    eig_90 = eigenvalues[:dim_90_idx + 1]
    eig_95 = eigenvalues[:dim_95_idx + 1]

    # Compute volumes for each threshold (sqrt of product of eigenvalues)
    def compute_volume_from_eigs(eigs):
        if len(eigs) == 0:
            return 0.0
        log_sum = np.sum(np.log(eigs))
        return np.exp(0.5 * log_sum) if log_sum > -np.inf else 0.0

    volume_70 = compute_volume_from_eigs(eig_70)
    volume_80 = compute_volume_from_eigs(eig_80)
    volume_90 = compute_volume_from_eigs(eig_90)
    volume_95 = compute_volume_from_eigs(eig_95)

    # Overall volume: For an n-dimensional ellipsoid, volume is proportional to sqrt(det(C))
    # The determinant is the product of eigenvalues, so volume = sqrt(prod(eigenvalues))
    # Using log for numerical stability with large matrices
    log_det = np.sum(np.log(eigenvalues))
    volume = np.exp(0.5 * log_det) if log_det > -np.inf else 0.0

    # Extension: Largest eigenvalue (spread along the principal direction)
    # This represents the "stretching" of the cloud along its main axis
    extension = eigenvalues[0]

    # Dimensionality: Use participation ratio as measure of effective dimensionality
    dimensionality = PR_emp

    dict_stats = {
        'mii': mii,
        'mij': mij,
        'sii': sii,
        'sij': sij,
        'PR_emp': PR_emp,
        'PR_fit': PR_fit,
        'dimensionality': dimensionality,
        'volume': volume,
        'extension': extension,
        'dim_70': dim_70_idx + 1,  # +1 because index is 0-based, dim is 1-based
        'dim_80': dim_80_idx + 1,
        'dim_90': dim_90_idx + 1,
        'dim_95': dim_95_idx + 1,
        'eigenvalues_70': [eig_70],  # Wrap in list for DataFrame storage
        'eigenvalues_80': [eig_80],
        'eigenvalues_90': [eig_90],
        'eigenvalues_95': [eig_95],
        'volume_70': volume_70,
        'volume_80': volume_80,
        'volume_90': volume_90,
        'volume_95': volume_95
    }
    return dict_stats

def compute_cloud_separation(means_dict, covs_dict):
    """
    Compute average cloud separation metric for each image.

    For each pair of images (i, j), computes:
    - Distance between means: d = ||μⱼ - μᵢ||
    - Direction vector: v = (μⱼ - μᵢ) / d
    - Variance along v for each cloud: σᵢ² = v^T Cᵢ v, σⱼ² = v^T Cⱼ v
    - Separation metric: d / sqrt(σᵢ² + σⱼ²)

    Returns the average separation metric per image over all pairs.

    Parameters:
        means_dict (dict): Dictionary mapping image_name to mean activity vector
        covs_dict (dict): Dictionary mapping image_name to covariance matrix

    Returns:
        dict: Dictionary mapping image_name to average separation metric
    """
    image_names = list(means_dict.keys())
    if len(image_names) < 2:
        # Need at least 2 images to compute separation
        return {name: np.nan for name in image_names}

    separation_scores = {name: [] for name in image_names}

    # Compute pairwise separations
    for img_i, img_j in itertools.combinations(image_names, 2):
        mean_i = means_dict[img_i]
        mean_j = means_dict[img_j]
        cov_i = covs_dict[img_i]
        cov_j = covs_dict[img_j]

        # Compute distance and direction
        diff = mean_j - mean_i
        distance = np.linalg.norm(diff)

        if distance == 0:
            # Images have identical means, separation is 0
            separation = 0.0
        else:
            # Unit vector in direction from i to j
            direction = diff / distance

            # Compute variance along this direction for each cloud
            var_i = direction.T @ cov_i @ direction
            var_j = direction.T @ cov_j @ direction

            # Separation metric: normalized distance
            # Higher values indicate better separation
            combined_var = var_i + var_j
            if combined_var > 0:
                separation = distance / np.sqrt(combined_var)
            else:
                separation = np.inf if distance > 0 else 0.0

        separation_scores[img_i].append(separation)
        separation_scores[img_j].append(separation)

    # Average separation per image
    avg_separation = {
        name: np.mean(scores) if len(scores) > 0 else np.nan
        for name, scores in separation_scores.items()
    }

    return avg_separation


def compute_subsampled_stats(spikecounts_all, group_metadata, geometry_stats_func, area, group_size=200, random_state=2):
    """
    Compute geometry statistics on subsampled groups of trials.

    Parameters:
        spikecounts_all (np.array): Array of shape (N_trials, N_neurons)
        group_metadata (dict): Metadata to attach to each result
        geometry_stats_func (callable): Function to compute geometry stats
        area (str): Brain area label
        group_size (int): Size of each subsample group (default: 200)
        random_state (int): Random seed (not used for ordered subsampling)

    Returns:
        list: List of dictionaries with geometry statistics for each subsample
    """
    N_rows = spikecounts_all.shape[0]
    N_neurons = spikecounts_all.shape[1]

    is_change_flag = bool(group_metadata.get('is_change', False))

    if is_change_flag:
        if N_rows == 0:
            return []
        chunks = [np.arange(N_rows)]
    else:
        n_groups = N_rows // group_size
        if n_groups == 0:
            return []
        chunks = [np.arange(i * group_size, (i + 1) * group_size) for i in range(n_groups)]

    stats_list = []
    for chunk_idx, idx_chunk in enumerate(chunks):
        submatrix = spikecounts_all[idx_chunk, :]
        dict_stats = geometry_stats_func(submatrix.T)
        if not dict_stats:
            continue
        dict_stats.update(group_metadata)
        dict_stats.update({
            'area': area,
            'N': N_neurons,
            'subsample_id': chunk_idx,
        })
        stats_list.append(dict_stats)

    return stats_list


def process_session_general(session_id, session_number, experience_level, image_set):
    """
    Process data for a single session to compute VISp geometry statistics
    (dimensionality, volume, extension, cloud separation).

    Parameters:
        session_id: Session identifier
        session_number: Session number (first or second recording day)
        experience_level: Experience level (Familiar or Novel)
        image_set: Image set identifier

    Returns:
        pd.DataFrame: Geometry statistics for this session (empty if VISp has too few units)
    """
    session_str = str(session_id)
    base_dir = ALLEN_DATA_DIR / 'day1'
    activity_file = base_dir / session_str / 'df_activity.pk'
    units_file = base_dir / session_str / 'df_units.pk'

    df_activity_ses = pd.read_pickle(activity_file)
    df_units_ses = pd.read_pickle(units_file)

    df_activity_ses['session_id'] = session_str
    df_units_ses['session_id'] = session_str
    print(f"Processing session: {int(session_str)} ")
    print(f"N: {len(df_units_ses)} and spikes: {len(df_activity_ses.iloc[0]['spikecounts'])}")

    area = 'VISp'
    df_units_area_ses = df_units_ses[df_units_ses.structure_acronym == area]
    if df_units_area_ses.shape[0] < 30:
        return pd.DataFrame()

    df_activity_ses = df_activity_ses[df_activity_ses.is_viewing == True]
    df_activity_ses = df_activity_ses[df_activity_ses.time_from_stimulus >= 0.1]
    df_activity_ses = df_activity_ses[df_activity_ses.is_change == False]
    df_activity_ses['image_name'] = pd.Categorical(df_activity_ses['image_name'], ordered=True)

    idx_units_area = df_units_area_ses.index.values
    df_activity_local = df_activity_ses.copy()
    df_activity_local['spikecounts_area'] = df_activity_local['spikecounts'].apply(
        lambda x: x[idx_units_area]
    )

    stats_list = []
    for outer_group_name, df_outer_group in df_activity_local.groupby(
        ['session_id', 'active', 'is_image_novel']
    ):
        session_id, active, is_image_novel = outer_group_name
        all_stats_for_separation = {}  # Key: (image_name, subsample_id)

        # observed=True: image_name is Categorical, so the default observed=False
        # would emit empty groups for images that never appear in this outer slice.
        for image_name, df_activity_group in df_outer_group.groupby(
            'image_name', observed=True
        ):
            if df_activity_group.empty:
                continue

            group_keys = {
                'session_id': session_id,
                'active': active,
                'image_name': image_name,
                'is_change': False,
                'is_image_novel': is_image_novel,
                'session_number': session_number,
                'experience_level': experience_level,
                'image_set': image_set,
            }

            spikecounts_all = np.stack(df_activity_group.spikecounts_area.values, axis=0)
            N_rows = spikecounts_all.shape[0]
            group_size = 200
            n_groups = N_rows // group_size
            if n_groups == 0:
                continue
            chunks = [np.arange(i * group_size, (i + 1) * group_size) for i in range(n_groups)]

            for chunk_idx, idx_chunk in enumerate(chunks):
                submatrix = spikecounts_all[idx_chunk, :]
                mean_vec = np.mean(submatrix, axis=0)
                cov_matrix = np.cov(submatrix, rowvar=False)

                dict_stats = geometry_stats(submatrix.T)
                if not dict_stats:
                    continue

                dict_stats.update(group_keys)
                dict_stats.update({
                    'area': area,
                    'N': submatrix.shape[1],
                    'subsample_id': chunk_idx,
                })
                all_stats_for_separation[(image_name, chunk_idx)] = (
                    dict_stats, mean_vec, cov_matrix
                )

        subsample_ids = set(key[1] for key in all_stats_for_separation.keys())
        for subsample_id in subsample_ids:
            means_dict = {}
            covs_dict = {}
            image_stats_dict = {}
            for (img_name, sub_id), (stats_dict, mean, cov) in all_stats_for_separation.items():
                if sub_id == subsample_id:
                    means_dict[img_name] = mean
                    covs_dict[img_name] = cov
                    image_stats_dict[img_name] = stats_dict

            if len(means_dict) >= 2:
                separations = compute_cloud_separation(means_dict, covs_dict)
                for img_name, stats_dict in image_stats_dict.items():
                    stats_dict['cloud_separation'] = separations[img_name]
                    stats_list.append(stats_dict)
            else:
                for img_name, stats_dict in image_stats_dict.items():
                    stats_dict['cloud_separation'] = np.nan
                    stats_list.append(stats_dict)

    return pd.DataFrame(stats_list)

# %%
# ------------------------------------
# MAIN: COMPUTE GEOMETRY STATISTICS
# ------------------------------------
# session_number: indicating whether it was the first or second recording day for the mouse
# experience level: tells you whether the image set used for a particular recording was the same
# as the training image set (Familiar), or different from the training image set (Novel).
if __name__ == '__main__':
    data_dir = config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    df_sessions = pd.read_pickle(ALLEN_DATA_DIR / 'df_sessions.pk')
    df_sessions = df_sessions.query(
        'session_number==1 and experience_level=="Familiar"'
    )

    session_meta = df_sessions[
        ['ecephys_session_id', 'session_number', 'experience_level', 'image_set']
    ].values.tolist()

    with multiprocessing.Pool(processes=6) as pool:
        results = pool.starmap(process_session_general, session_meta)

    df_stats = pd.concat(results, ignore_index=True)
    df_stats.to_pickle(data_dir / 'visp_cloud_shape.pk')
    print(f"Saved {data_dir / 'visp_cloud_shape.pk'}")