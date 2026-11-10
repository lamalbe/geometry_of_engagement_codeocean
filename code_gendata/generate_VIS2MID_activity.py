# %%
"""
Build a pooled activity table for cortex-to-midbrain grouped areas.

Day-1 Familiar sessions (image sets G and H). Viewing bins with
time_from_stimulus >= 0.1 s; omitted, novel, and change presentations dropped.

Grouped labels (not Allen acronyms):
  VISpl = {VISp, VISrl, VISl, VISal}
  VISm  = {VISpm, VISam}
  SCMRN = {MRN, SCig, SCiw}

Output (data/):
  df_cortex2mid.pk
"""
import multiprocessing
import sys
from pathlib import Path

import pandas as pd

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

import config

ALLEN_DATA_DIR = config.ALLEN_PREPROCESSED_DIR
TIME_FROM_STIMULUS_MIN = 0.1  # discard first 100 ms after stimulus onset

# Label, then the structure acronyms that belong to that label.
GROUP_DEFS = [
    ('VISpl', {'VISp', 'VISrl', 'VISl', 'VISal'}),
    ('VISm', {'VISpm', 'VISam'}),
    ('SCMRN', {'MRN', 'SCig', 'SCiw'}),
]

# %%
# --------------------------------------------------------------------
# SESSION PROCESSING
# --------------------------------------------------------------------


def process_session_general(session_id, session_number, experience_level, image_set):
    session_str = str(session_id)
    base_dir = ALLEN_DATA_DIR / 'day1'
    activity_file = base_dir / session_str / 'df_activity.pk'
    units_file = base_dir / session_str / 'df_units.pk'

    df_activity_ses = pd.read_pickle(activity_file)
    df_units_ses = pd.read_pickle(units_file)

    df_activity_ses['session_id'] = session_str
    df_units_ses['session_id'] = session_str

    print(f"Processing session: {session_str}")

    df_activity_ses = df_activity_ses[df_activity_ses.is_viewing == True].copy()
    df_activity_ses = df_activity_ses[
        df_activity_ses.time_from_stimulus >= TIME_FROM_STIMULUS_MIN
    ].copy()
    df_activity_ses['image_name'] = pd.Categorical(df_activity_ses['image_name'], ordered=True)

    df_activity_ses = df_activity_ses[
        (df_activity_ses['image_name'] != 'omitted') &
        (df_activity_ses['is_image_novel'] == False) &
        (df_activity_ses['is_change'] == False)
    ].copy()

    output_dfs = []

    for area_name, areas_of_interest in GROUP_DEFS:
        df_units_area_ses = df_units_ses[
            df_units_ses.structure_acronym.isin(areas_of_interest)
        ].copy()
        if df_units_area_ses.empty:
            continue

        idx_units_area = df_units_area_ses.index.values
        n_units_area = len(idx_units_area)

        df_activity_local = df_activity_ses.copy()
        df_activity_local['spikecounts_area'] = df_activity_local['spikecounts'].apply(
            lambda x: x[idx_units_area]
        )
        df_activity_local['area'] = area_name
        df_activity_local['N'] = n_units_area
        df_activity_local['session_number'] = session_number
        df_activity_local['experience_level'] = experience_level
        df_activity_local['image_set'] = image_set

        keep_cols = [
            'session_id',
            'active',
            'image_name',
            'bin_start_times',
            'time_from_stimulus',
            'is_change',
            'is_image_novel',
            'session_number',
            'experience_level',
            'image_set',
            'area',
            'N',
            'spikecounts_area',
        ]
        keep_cols = [c for c in keep_cols if c in df_activity_local.columns]
        output_dfs.append(df_activity_local[keep_cols].copy())

    if len(output_dfs) == 0:
        return pd.DataFrame()

    return pd.concat(output_dfs, ignore_index=True)


# %%
# ------------------------------------
# MAIN
# ------------------------------------
# session_number: first or second recording day.
# experience_level: Familiar (same as training) or Novel.
# image_set: G or H; both are kept for day-1 Familiar sessions.

if __name__ == '__main__':
    data_dir = config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    df_sessions = pd.read_pickle(ALLEN_DATA_DIR / 'df_sessions.pk')
    df_sessions = df_sessions.query(
        'session_number == 1 and experience_level == "Familiar"'
    ).copy()

    session_meta = df_sessions[
        ['ecephys_session_id', 'session_number', 'experience_level', 'image_set']
    ].values.tolist()

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(process_session_general, session_meta)

    results = [df for df in results if df is not None and not df.empty]

    if len(results) == 0:
        df_cortex2mid = pd.DataFrame()
    else:
        df_cortex2mid = pd.concat(results, ignore_index=True)

    out_file = data_dir / 'df_cortex2mid.pk'
    df_cortex2mid.to_pickle(out_file)

    print(f"Saved {len(df_cortex2mid)} rows to {out_file}")
# %%
