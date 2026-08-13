# %%
"""Preprocess Allen Visual Behavior Neuropixels sessions into pickled dataframes.

Raw Allen cache: /data/AllenNPXVisualBehavior
Output:          /data/allen_engagement_data/{day1,day2}/<session_id>/
"""

import gc
import concurrent.futures
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)

import warnings
warnings.filterwarnings('ignore')

# %%
# ------------------------
# PATHS / HYPERPARAMETERS
# ------------------------
ALLEN_RAW_DIR = Path('/data/AllenNPXVisualBehavior')
# AllenSDK expects cache_dir/visual-behavior-neuropixels/{manifests,...}
CACHE_PARENT_DIR = Path(__file__).resolve().parent.parent / 'data'
CACHE_LINK = CACHE_PARENT_DIR / 'visual-behavior-neuropixels'

OUTPUT_DIR = Path('/data/allen_engagement_data')
DAY_SUBDIRS = {1: 'day1', 2: 'day2'}

BINSIZE = 0.05  # 50 ms
MAX_WORKERS = 4


def ensure_allensdk_cache_layout(
    allen_raw_dir: Path = ALLEN_RAW_DIR,
    cache_parent_dir: Path = CACHE_PARENT_DIR,
) -> Path:
    """Ensure AllenSDK local-cache layout via symlink to the server dataset."""
    cache_parent_dir.mkdir(parents=True, exist_ok=True)
    link = cache_parent_dir / 'visual-behavior-neuropixels'
    if link.is_symlink() or link.exists():
        if link.resolve() != allen_raw_dir.resolve():
            raise RuntimeError(
                f'Cache link {link} points to {link.resolve()}, '
                f'expected {allen_raw_dir}'
            )
    else:
        link.symlink_to(allen_raw_dir)

    manifests = link / 'manifests'
    if not manifests.is_dir():
        raise FileNotFoundError(f'Manifests not found at {manifests}')
    return cache_parent_dir


def load_cache(cache_dir: Path) -> VisualBehaviorNeuropixelsProjectCache:
    return VisualBehaviorNeuropixelsProjectCache.from_local_cache(
        cache_dir=str(cache_dir),
        use_static_cache=True,
    )


# %%
# -------------------
# COMPUTE OPTOTAGGING
# -------------------
def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    bins = np.arange(0, windowDur + binSize, binSize)
    counts = np.zeros(bins.size - 1)
    for start in startTimes:
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start + windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd] - start, bins)[0]

    counts = counts / startTimes.size
    return counts / binSize, bins


def compute_optotagging(session, units, channels):
    """Identify light-responsive cortical units and annotate the units table."""
    unit_channels = units.merge(channels, left_on='peak_channel_id', right_index=True)
    unit_channels = unit_channels.sort_values('probe_vertical_position', ascending=False)

    good_unit_filter = (
        (unit_channels['snr'] > 1)
        & (unit_channels['isi_violations'] < 1)
        & (unit_channels['firing_rate'] > 0.1)
    )
    good_units = unit_channels.loc[good_unit_filter]
    spike_times = session.spike_times
    opto_table = session.optotagging_table

    duration = opto_table.duration.min()  # short pulses
    level = opto_table.level.max()        # high power

    cortical_units = good_units[good_units['structure_acronym'].str.contains('VIS')]
    opto_times = opto_table.loc[
        (opto_table['duration'] == duration) & (opto_table['level'] == level)
    ]['start_time'].values

    time_before = 0.01
    psth_duration = 0.03
    binSize = 0.001
    opto_response = []
    for iu, _unit in cortical_units.iterrows():
        unit_response, _bins = makePSTH(
            spike_times[iu],
            opto_times - time_before,
            psth_duration,
            binSize=binSize,
        )
        opto_response.append(unit_response)

    opto_response = np.array(opto_response)
    baseline_window = slice(0, 9)
    response_window = slice(11, 18)

    mean_opto_active = np.mean(opto_response[:, response_window], axis=1)
    mean_opto_passive = np.mean(opto_response[:, baseline_window], axis=1)

    units = units.copy()
    units['mean_opto_active'] = np.nan
    units['mean_opto_passive'] = np.nan
    idxs = units.index.isin(cortical_units.index)
    units.loc[idxs, 'mean_opto_active'] = mean_opto_active
    units.loc[idxs, 'mean_opto_passive'] = mean_opto_passive
    return units


# %%
# -----------------------
# SESSION DATAFRAME LOAD
# -----------------------
def generate_dataframes_session(session_id, cache, ecephys_sessions_table):
    session = cache.get_ecephys_session(ecephys_session_id=session_id)
    units = session.get_units()
    channels = session.get_channels()

    genotype = ecephys_sessions_table.loc[session_id, 'genotype']
    if str(genotype)[:2] != 'wt':
        print(f'        computing optotagging for {session_id}')
        units = compute_optotagging(session, units, channels)

    units_channels = units.merge(channels, left_on='peak_channel_id', right_index=True)
    units_channels = units_channels.sort_values('probe_vertical_position', ascending=False)

    good_unit_filter = (
        (units_channels['snr'] > 1)
        & (units_channels['isi_violations'] < 1)
        & (units_channels['firing_rate'] > 0.1)
    )
    good_units = units_channels.loc[good_unit_filter]
    spike_times = session.spike_times

    df_trials = session.trials
    df_presentations = session.stimulus_presentations

    df_spikes_all = pd.DataFrame(
        {'unit': x, 'spike_times': spike_times[x]} for x in spike_times.keys()
    )
    df_units = good_units.reset_index()
    df_spikes = df_spikes_all[df_spikes_all.unit.isin(df_units.id.values)]

    behavior_sessions_table = cache.get_behavior_session_table()
    valid_sessions = behavior_sessions_table.ecephys_session_id.dropna()
    if any(valid_sessions == int(session_id)):
        behavior_id = behavior_sessions_table[
            behavior_sessions_table.ecephys_session_id == int(session_id)
        ].index.to_list()
        behavior_session = cache.get_behavior_session(behavior_id[0])
        df_speed = behavior_session.running_speed
        df_licks = behavior_session.licks
        df_rewards = behavior_session.rewards
        df_pupils = behavior_session.eye_tracking
    else:
        df_speed = pd.DataFrame(columns=['timestamps', 'speed'])
        df_licks = pd.DataFrame(columns=['timestamps', 'frame'])
        df_rewards = pd.DataFrame(columns=['timestamps', 'volume', 'auto_rewarded'])
        df_pupils = pd.DataFrame(columns=['timestamps', 'pupil_area'])

    return (
        df_units,
        df_spikes,
        df_trials,
        df_presentations,
        df_speed,
        df_licks,
        df_rewards,
        df_pupils,
    )


# %%
# -------------------------------------
# BINNING SPIKES AND BASELINE FUNCTIONS
# -------------------------------------
def spike_binning(df_spikes, df_presentations_set, binsize, vars_to_keep):
    df_presentations_set = df_presentations_set.copy()
    df_presentations_set['next_image'] = df_presentations_set['image_name'].shift(-1)
    df_presentations_set['next_stimulus_block'] = df_presentations_set['stimulus_block'].shift(-1)
    df_presentations_set['next_start_time'] = df_presentations_set['start_time'].shift(-1)
    df_presentations_set['is_next_image_same'] = (
        df_presentations_set['image_name'] == df_presentations_set['next_image']
    )
    df_presentations_set['is_next_block_same'] = (
        df_presentations_set['stimulus_block'] == df_presentations_set['next_stimulus_block']
    )

    df_presentations_set = df_presentations_set[df_presentations_set['duration'] > binsize]
    df_presentations_set['bins_presentations'] = df_presentations_set.apply(
        lambda x: np.arange(x.start_time, x.end_time, binsize), axis=1
    )

    df_presentations_set['bin_start_times'] = df_presentations_set['bins_presentations'].apply(
        lambda x: x[:-1]
    )
    df_presentations_set['bin_end_times'] = df_presentations_set['bins_presentations'].apply(
        lambda x: x[1:]
    )
    df_presentations_set['time_from_stimulus'] = df_presentations_set['bins_presentations'].apply(
        lambda x: np.round(x[:-1] - x.min(), 3)
    )

    bin_start_times = np.concatenate(df_presentations_set['bin_start_times'].values)
    bin_end_times = np.concatenate(df_presentations_set['bin_end_times'].values)
    times_from_stimulus = np.concatenate(df_presentations_set['time_from_stimulus'].values)

    df_presentations_set['bin_to_discard'] = df_presentations_set['bins_presentations'].apply(
        lambda x: len(x)
    )
    bins_to_discard = df_presentations_set['bin_to_discard'].cumsum().values[:-1]
    all_bins = np.concatenate(df_presentations_set['bins_presentations'].values)
    binned_spikes = np.stack(
        df_spikes[:].spike_times.apply(
            lambda x: stats.binned_statistic(x, x, bins=all_bins, statistic='count').statistic
        ).values,
        axis=0,
    )
    binned_spikes = np.delete(binned_spikes, bins_to_discard, axis=1)

    df_activity = pd.DataFrame({
        'spikecounts': [binned_spikes[:, i] for i in range(binned_spikes.shape[1])],
        'time_from_stimulus': times_from_stimulus,
        'bin_start_times': bin_start_times,
        'bin_end_times': bin_end_times,
    })
    df_activity['binsize'] = binsize

    for var in vars_to_keep:
        df_activity[var] = np.concatenate(
            df_presentations_set.apply(
                lambda x: np.repeat(x[var], len(x['bins_presentations']) - 1), axis=1
            ).values
        )

    for flag in ['is_next_image_same', 'is_next_block_same']:
        df_activity[flag] = np.concatenate(
            df_presentations_set.apply(
                lambda x: np.repeat(x[flag], len(x['bins_presentations']) - 1), axis=1
            ).values
        )

    return df_activity, df_presentations_set


def baseline_binning(df_spikes, df_presentations_set, binsize, vars_to_keep):
    df_presentations_set = df_presentations_set.copy()
    df_presentations_set['next_image'] = df_presentations_set['image_name'].shift(-1)
    df_presentations_set['next_stimulus_block'] = df_presentations_set['stimulus_block'].shift(-1)
    df_presentations_set['next_start_time'] = df_presentations_set['start_time'].shift(-1)
    df_presentations_set['is_next_image_same'] = (
        df_presentations_set['image_name'] == df_presentations_set['next_image']
    )
    df_presentations_set['is_next_block_same'] = (
        df_presentations_set['stimulus_block'] == df_presentations_set['next_stimulus_block']
    )

    df_presentations_set['baseline_duration'] = (
        df_presentations_set['next_start_time'] - df_presentations_set['end_time']
    )
    df_presentations_set['baseline_end_time'] = df_presentations_set['next_start_time']

    df_presentations_set.drop(
        columns=['next_image', 'next_stimulus_block', 'next_start_time'],
        inplace=True,
    )
    df_presentations_set = df_presentations_set.iloc[:-1]

    df_presentations_set = df_presentations_set[df_presentations_set['duration'] > binsize]
    df_presentations_set['bins_presentations'] = df_presentations_set.apply(
        lambda x: np.arange(x.end_time, x.baseline_end_time, binsize), axis=1
    )

    df_presentations_set['bin_start_times'] = df_presentations_set['bins_presentations'].apply(
        lambda x: x[:-1]
    )
    df_presentations_set['bin_end_times'] = df_presentations_set['bins_presentations'].apply(
        lambda x: x[1:]
    )
    df_presentations_set['time_from_stimulus'] = df_presentations_set['bins_presentations'].apply(
        lambda x: np.round(x[:-1] - x.min(), 3)
    )

    bin_start_times = np.concatenate(df_presentations_set['bin_start_times'].values)
    bin_end_times = np.concatenate(df_presentations_set['bin_end_times'].values)
    times_from_stimulus = np.concatenate(df_presentations_set['time_from_stimulus'].values)

    df_presentations_set['bin_to_discard'] = df_presentations_set['bins_presentations'].apply(
        lambda x: len(x)
    )
    bins_to_discard = df_presentations_set['bin_to_discard'].cumsum().values[:-1]
    all_bins = np.concatenate(df_presentations_set['bins_presentations'].values)
    binned_spikes = np.stack(
        df_spikes[:].spike_times.apply(
            lambda x: stats.binned_statistic(x, x, bins=all_bins, statistic='count').statistic
        ).values,
        axis=0,
    )
    binned_spikes = np.delete(binned_spikes, bins_to_discard, axis=1)

    df_activity = pd.DataFrame({
        'spikecounts': [binned_spikes[:, i] for i in range(binned_spikes.shape[1])],
        'time_from_stimulus': times_from_stimulus,
        'bin_start_times': bin_start_times,
        'bin_end_times': bin_end_times,
    })
    df_activity['binsize'] = binsize

    for var in vars_to_keep:
        df_activity[var] = np.concatenate(
            df_presentations_set.apply(
                lambda x: np.repeat(x[var], len(x['bins_presentations']) - 1), axis=1
            ).values
        )

    for flag in ['is_next_image_same', 'is_next_block_same']:
        df_activity[flag] = np.concatenate(
            df_presentations_set.apply(
                lambda x: np.repeat(x[flag], len(x['bins_presentations']) - 1), axis=1
            ).values
        )

    return df_activity, df_presentations_set


def _bin_aligned_series(timestamps, values, bin_starts, bin_ends):
    """Map continuous/event series into per-bin {timestamp: value} dicts."""
    if len(timestamps) == 0:
        return [{} for _ in range(len(bin_starts))]

    start_idxs = np.searchsorted(timestamps, bin_starts, side='left')
    end_idxs = np.searchsorted(timestamps, bin_ends, side='right')
    return [
        dict(zip(timestamps[start:end], values[start:end]))
        for start, end in zip(start_idxs, end_idxs)
    ]


# %%
# -----------------------
# SINGLE PROCESS FUNCTION
# -----------------------
def process_session(
    session_id,
    cache_dir,
    output_dir,
    session_number,
    binsize=BINSIZE,
):
    """Load one session, build activity/behavior tables, save under day1/ or day2/."""
    try:
        cache = load_cache(Path(cache_dir))
        ecephys_sessions_table = cache.get_ecephys_session_table()

        (
            df_units,
            df_spikes,
            df_trials,
            df_presentations,
            df_speed,
            df_licks,
            df_rewards,
            df_pupils,
        ) = generate_dataframes_session(session_id, cache, ecephys_sessions_table)

        vars_to_keep = [
            'active',
            'stimulus_block',
            'is_change',
            'image_name',
            'stimulus_presentations_id',
            'is_image_novel',
        ]
        blocks_to_keep = [0, 5]

        df_presentations_set = (
            df_presentations[df_presentations['stimulus_block'].isin(blocks_to_keep)]
            .reset_index()
        )
        df_presentations_stim = df_presentations_set.copy()
        df_presentations_base = df_presentations_set.copy()

        df_activity_stimulus, _ = spike_binning(
            df_spikes, df_presentations_stim, binsize, vars_to_keep
        )
        df_activity_stimulus['is_viewing'] = True

        df_activity_baseline, _ = baseline_binning(
            df_spikes, df_presentations_base, binsize, vars_to_keep
        )
        df_activity_baseline['is_viewing'] = False

        df_activity = pd.concat(
            [df_activity_stimulus, df_activity_baseline],
            ignore_index=True,
        )
        df_activity['session_number'] = int(session_number)

        bin_starts = df_activity['bin_start_times'].values
        bin_ends = df_activity['bin_end_times'].values

        df_activity['speed'] = _bin_aligned_series(
            df_speed['timestamps'].values,
            df_speed['speed'].values,
            bin_starts,
            bin_ends,
        )
        df_activity['pupil_area'] = _bin_aligned_series(
            df_pupils['timestamps'].values,
            df_pupils['pupil_area'].values,
            bin_starts,
            bin_ends,
        )
        df_activity['licks'] = _bin_aligned_series(
            df_licks['timestamps'].values,
            df_licks['frame'].values,
            bin_starts,
            bin_ends,
        )
        df_activity['rewards'] = _bin_aligned_series(
            df_rewards['timestamps'].values,
            df_rewards['volume'].values,
            bin_starts,
            bin_ends,
        )
        df_activity['auto_rew'] = _bin_aligned_series(
            df_rewards['timestamps'].values,
            df_rewards['auto_rewarded'].values,
            bin_starts,
            bin_ends,
        )

        day_name = DAY_SUBDIRS[int(session_number)]
        session_dir = Path(output_dir) / day_name / str(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        df_units.to_pickle(session_dir / 'df_units.pk')
        df_spikes.to_pickle(session_dir / 'df_spikes.pk')
        df_trials.to_pickle(session_dir / 'df_trials.pk')
        df_presentations.to_pickle(session_dir / 'df_presentations.pk')
        df_activity.to_pickle(session_dir / 'df_activity.pk')
        df_speed.to_pickle(session_dir / 'df_speed.pk')
        df_licks.to_pickle(session_dir / 'df_licks.pk')
        df_rewards.to_pickle(session_dir / 'df_rewards.pk')
        df_pupils.to_pickle(session_dir / 'df_pupils.pk')

        print(f'Saved session {session_id} -> {session_dir}')

    except Exception as e:
        print(f'Error processing session {session_id}: {e}')

    finally:
        for var_name in [
            'df_units',
            'df_spikes',
            'df_trials',
            'df_presentations',
            'df_activity',
            'df_speed',
            'df_licks',
            'df_rewards',
            'df_pupils',
            'cache',
        ]:
            if var_name in locals():
                del locals()[var_name]
        gc.collect()


# %%
# ----------------------------------------------------------
# MAIN: COMPUTE MAIN NEURAL ACTIVITY AND BEHAVIOR DATAFRAMES
# ----------------------------------------------------------
if __name__ == '__main__':
    test_run = False
    maxidx = 1 if test_run else None  # None = all sessions

    cache_dir = ensure_allensdk_cache_layout()
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for day in DAY_SUBDIRS.values():
        (output_dir / day).mkdir(parents=True, exist_ok=True)

    print(f'Allen cache parent: {cache_dir}')
    print(f'Output directory:   {output_dir}')

    cache = load_cache(cache_dir)
    ecephys_sessions_table = cache.get_ecephys_session_table()
    df_sessions = ecephys_sessions_table.reset_index()
    df_sessions.to_pickle(output_dir / 'df_sessions.pk')

    if 'session_number' not in df_sessions.columns:
        raise KeyError('ecephys session table is missing session_number')

    session_rows = df_sessions[['ecephys_session_id', 'session_number']].dropna()
    if maxidx is not None:
        session_rows = session_rows.iloc[:maxidx]

    print(f'Processing {len(session_rows)} sessions '
          f'(day1={int((session_rows.session_number == 1).sum())}, '
          f'day2={int((session_rows.session_number == 2).sum())})')

    # Parallel processing; each worker rebuilds its own cache (pickle-safe).
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                process_session,
                int(row.ecephys_session_id),
                str(cache_dir),
                str(output_dir),
                int(row.session_number),
                BINSIZE,
            )
            for row in session_rows.itertuples(index=False)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'Session processing generated an exception: {exc}')
