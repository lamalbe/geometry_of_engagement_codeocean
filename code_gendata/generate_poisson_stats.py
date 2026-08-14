# %%
"""
Area-level participation ratio, mean firing rates, and per-neuron count variances.

For each ecephys session, we load binned spike counts from df_activity (50 ms bins)
and unit metadata from df_units. Activity is filtered to familiar, non-change,
non-novel image presentations during viewing, with time_from_stimulus >= 0.1 s
(first 100 ms after stimulus onset discarded). Day-1 Familiar sessions from both
image sets G and H are included.

Units are grouped into coarse areas (VISp, VISpl, VISm, LGd, SCMRN). VISp is
V1 (structure_acronym VISp) alone; VISpl pools VISp with VISrl, VISl, VISal.
Within each area, for every (session_id, active, image_name) group:

1. Build a matrix of spike counts with shape (n_repetitions, n_neurons), where each
   row is one filtered bin (one repetition of spikecounts_area for that condition).

2. From all repetitions (full T), compute:
   - PR_emp: empirical participation ratio (trace(C)^2 / trace(C @ C)) using
     np.cov(spikecounts.T).
   - mean_firing_rates: per-neuron mean of spikecounts / binsize (Hz) across repetitions.
   - spikecount_variances: diagonal of C (variance of each neuron's counts across
     repetitions), for Poissonianity checks (e.g. Fano = variance / mean count).

3. Additionally, split repetitions in time order into consecutive chunks of 200 and
   compute the same statistics per subsample_id (for repeated variance–mean pairs).

Areas with fewer than 30 units are skipped.

Outputs (clean_scripts/data):
  - df_poisson_stats.pk: one row per (area, session_id, active, image_name)
  - df_poisson_stats_subsampled.pk: one row per (..., subsample_id)
"""

import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

ALLEN_DATA_DIR = Path('/data/allen_engagement_data')

BINSIZE = 0.05
MIN_UNITS_PER_AREA = 30
SUBSAMPLE_CHUNK_SIZE = 200
TIME_FROM_STIMULUS_MIN = 0.1  # discard first 100 ms after stimulus onset

# %%
# --------------------------------------------------------------------
# COVARIANCE / GROUP STATS
# --------------------------------------------------------------------


def compute_group_stats(spikecounts_all, binsize=BINSIZE):
    """
    Compute PR_emp, mean firing rates, and per-neuron spike-count variances.

    Parameters
    ----------
    spikecounts_all : ndarray, shape (n_repetitions, n_neurons)
        Spike counts per repetition (bin-row) and neuron.
    binsize : float
        Bin width in seconds (default 50 ms).

    Returns
    -------
    dict or None
        Keys: PR_emp, mean_firing_rates, spikecount_variances, n_repetitions, N_neurons.
        None if the group has no activity or zero covariance.
    """
    if spikecounts_all.size == 0:
        return None

    spkcounts = spikecounts_all.T  # (n_neurons, n_repetitions)
    if np.sum(spkcounts) == 0:
        return None

    C = np.cov(spkcounts)
    if np.sum(C) == 0:
        return None

    PR_emp = (np.trace(C) ** 2) / np.trace(C @ C)

    return {
        "PR_emp": PR_emp,
        "mean_firing_rates": np.mean(spikecounts_all / binsize, axis=0),
        "spikecount_variances": np.diag(C),
        "n_repetitions": spikecounts_all.shape[0],
        "N_neurons": spikecounts_all.shape[1],
    }


def compute_subsampled_group_stats(
    spikecounts_all, group_size=SUBSAMPLE_CHUNK_SIZE, binsize=BINSIZE
):
    """
    Split repetitions in time order into consecutive chunks and compute group stats
    per chunk.

    Remainder rows that do not fill a full chunk are dropped.

    Returns
    -------
    list of dict
        Each dict is compute_group_stats output plus subsample_id. Empty if T < group_size.
    """
    n_repetitions = spikecounts_all.shape[0]
    n_groups = n_repetitions // group_size
    if n_groups == 0:
        return []

    stats_list = []
    for subsample_id in range(n_groups):
        start = subsample_id * group_size
        chunk = spikecounts_all[start : start + group_size, :]
        chunk_stats = compute_group_stats(chunk, binsize=binsize)
        if chunk_stats is None:
            continue
        chunk_stats["subsample_id"] = subsample_id
        stats_list.append(chunk_stats)

    return stats_list

# %%
# --------------------------------------------------------------------
# SESSION PROCESSING
# --------------------------------------------------------------------

def process_session_general(session_id, session_number, experience_level, image_set):
    """
    Compute full-T and subsampled Poisson/covariance stats for one session.

    Parameters
    ----------
    session_id : int or str
        ecephys session id.
    session_number : int
        Recording day (1 or 2).
    experience_level : str
        Familiar or Novel.
    image_set : str
        Image set shown in this session (G or H). Stored in the output, not used as a filter.
    """
    session_str = str(session_id)
    base_dir = ALLEN_DATA_DIR / 'day1'
    activity_file = base_dir / session_str / "df_activity.pk"
    units_file = base_dir / session_str / "df_units.pk"

    df_activity_ses = pd.read_pickle(activity_file)
    df_units_ses = pd.read_pickle(units_file)

    df_activity_ses["session_id"] = session_str
    df_units_ses["session_id"] = session_str

    print(f"Processing session {session_str}")

    df_activity_ses = df_activity_ses[df_activity_ses["is_viewing"] == True].copy()
    df_activity_ses = df_activity_ses[
        df_activity_ses["time_from_stimulus"] >= TIME_FROM_STIMULUS_MIN
    ].copy()
    df_activity_ses = df_activity_ses[df_activity_ses["image_name"] != "omitted"].copy()
    df_activity_ses = df_activity_ses[df_activity_ses["is_image_novel"] == False].copy()
    df_activity_ses = df_activity_ses[df_activity_ses["is_change"] == False].copy()

    # VISpl / VISm / SCMRN are grouped labels, not Allen acronyms.
    network_defs = {
        "VISp": ["VISp"],
        "VISpl": ["VISp", "VISrl", "VISl", "VISal"],
        "VISm": ["VISpm", "VISam"],
        "LGd": ["LGd"],
        "SCMRN": ["MRN", "SCig", "SCiw"],
    }

    stats_list = []
    subsample_stats_list = []

    for area, acronyms in network_defs.items():
        df_units_area = df_units_ses[df_units_ses["structure_acronym"].isin(acronyms)].copy()
        if df_units_area.shape[0] < MIN_UNITS_PER_AREA:
            continue

        idx_units_area = df_units_area.index.values

        df_activity_local = df_activity_ses.copy()
        df_activity_local["spikecounts_area"] = df_activity_local["spikecounts"].apply(
            lambda x: x[idx_units_area]
        )

        for group_name, df_group in df_activity_local.groupby(
            ["session_id", "active", "image_name"],
            observed=True,
        ):
            if df_group.empty:
                continue

            group_session_id, active, image_name = group_name

            spikecounts_all = np.stack(df_group["spikecounts_area"].values, axis=0)

            group_metadata = {
                "area": area,
                "session_id": group_session_id,
                "active": active,
                "image_name": image_name,
                "image_set": image_set,
                "session_number": session_number,
                "experience_level": experience_level,
            }

            group_stats = compute_group_stats(spikecounts_all)
            if group_stats is not None:
                stats_list.append({**group_metadata, **group_stats})

            for chunk_stats in compute_subsampled_group_stats(spikecounts_all):
                subsample_stats_list.append({**group_metadata, **chunk_stats})

    return pd.DataFrame(stats_list), pd.DataFrame(subsample_stats_list)


# %%
# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
# session_number: first or second recording day for the mouse.
# experience_level: image set same as training (Familiar) or different (Novel).
# image_set: G or H; both are kept for day-1 Familiar sessions.

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    df_sessions = pd.read_pickle(ALLEN_DATA_DIR / 'df_sessions.pk')
    df_sessions = df_sessions.query(
        'session_number == 1 and experience_level == "Familiar"'
    ).copy()

    session_meta = df_sessions[
        ["ecephys_session_id", "session_number", "experience_level", "image_set"]
    ].values.tolist()

    with multiprocessing.Pool(processes=4) as pool:
        session_results = pool.starmap(process_session_general, session_meta)

    df_poisson_stats, df_poisson_stats_subsampled = (
        pd.concat(part, ignore_index=True) for part in zip(*session_results)
    )

    full_path = data_dir / "df_poisson_stats.pk"
    subsampled_path = data_dir / "df_poisson_stats_subsampled.pk"

    df_poisson_stats.to_pickle(full_path)
    df_poisson_stats_subsampled.to_pickle(subsampled_path)

    print(f"Saved {len(df_poisson_stats)} rows to {full_path}.")
    print(f"Saved {len(df_poisson_stats_subsampled)} rows to {subsampled_path}.")
# %%
