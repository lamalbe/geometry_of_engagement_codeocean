# %%
import numpy as np
import pandas as pd
import multiprocessing

from pathlib import Path

ALLEN_DATA_DIR = Path('/data/allen_engagement_data')

# Label, then the structure acronyms that belong to that label.
# VISpl / VISm / SCMRN are grouped labels, not Allen acronyms.
AREAS_OF_INTEREST = [
    ('VISp', ['VISp']),
    ('VISrl', ['VISrl']),
    ('VISl', ['VISl']),
    ('VISal', ['VISal']),
    ('VISpm', ['VISpm']),
    ('VISam', ['VISam']),
    ('VISpl', ['VISp', 'VISrl', 'VISl', 'VISal']),
    ('VISm', ['VISpm', 'VISam']),
    ('LGd', ['LGd']),
    ('SCMRN', ['MRN', 'SCig', 'SCiw']),
]
# %%
# --------------------------------------------------------------------
# DEFINE FUNCTIONS TO GET VIS, TH, HIPPO, AND MID BRAIN STATISTICS
# --------------------------------------------------------------------

def statsvalues2PR(mii, mij, sii, sij, N=10**4):
    PR = 1 / (1/N + 1/N * (sii / mii)**2 + (N-1) / N * (sij/mii)**2 + (N-1) / N * (mij / mii)**2)
    return PR

def cov_stats(spkcounts):
    C = np.cov(spkcounts)  # spkcounts can be an array of neuron activities, np.cov() estimates the covariance
    if np.sum(C) == 0 or np.sum(spkcounts) == 0:
        return {}
    ofdiag = np.triu(np.ones_like(C, dtype=bool), k=1)
    ondiag = np.eye(C.shape[0], dtype=bool)

    N = C.shape[0]
    mii = C[ondiag].mean()
    mij = C[ofdiag].mean()
    sii = np.sqrt(1 / (N - 1) * np.sum((C[ondiag] - mii) ** 2))
    sij = np.sqrt(1 / (N * (N - 1) - 2) * np.sum((C[ofdiag] - mij) ** 2))

    TV_emp = np.trace(C)
    PR_emp = (np.trace(C)) ** 2 / (np.trace(C @ C))
    diag_elements = C[ondiag]
    PR_fit = statsvalues2PR(mii, mij, sii, sij)

    dict_stats = {
        'mii': mii, 'mij': mij, 'sii': sii, 'sij': sij,
        'TV_emp': TV_emp,
        'PR_emp': PR_emp, 'PR_fit': PR_fit,
        'variances': [diag_elements],
    }
    return dict_stats


def process_session_general(session_id, session_number, experience_level, image_set):

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

    df_activity_ses = df_activity_ses[df_activity_ses.is_viewing == True]
    df_activity_ses = df_activity_ses[df_activity_ses.time_from_stimulus >= 0.05]
    df_activity_ses = df_activity_ses[df_activity_ses.is_change == False]
    df_activity_ses['image_name'] = pd.Categorical(df_activity_ses['image_name'], ordered=True)

    df_activity_ses['mean_speed'] = df_activity_ses['speed'].apply(
        lambda d: np.mean(list(d.values())) if isinstance(d, dict) and len(d) > 0 else np.nan
    )
    df_activity_ses['mean_pupil_area'] = df_activity_ses['pupil_area'].apply(
        lambda d: np.mean(list(d.values())) if isinstance(d, dict) and len(d) > 0 else np.nan
    )

    stats_list = []
    for area, structures in AREAS_OF_INTEREST:
        df_units_area_ses = df_units_ses[df_units_ses.structure_acronym.isin(structures)]
        if df_units_area_ses.shape[0] < 10:
            continue

        idx_units_area = df_units_area_ses.index.values
        df_activity_local = df_activity_ses.copy()
        df_activity_local['spikecounts_area'] = df_activity_local['spikecounts'].apply(
            lambda x: x[idx_units_area]
        )

        for group_name, df_activity_group in df_activity_local.groupby(
            ['session_id', 'active', 'image_name', 'is_image_novel'],
            observed=True,
        ):
            if df_activity_group.empty:
                continue

            session_id, active, image_name, is_image_novel = group_name
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
            speed_all = df_activity_group['mean_speed'].to_numpy(dtype=float)
            pupil_all = df_activity_group['mean_pupil_area'].to_numpy(dtype=float)

            avgspikecounts = np.mean(spikecounts_all, axis=0)
            dict_stats = cov_stats(spikecounts_all.T)
            dict_stats.update(group_keys)
            dict_stats.update({
                'area': area,
                'N': spikecounts_all.shape[1],
                'avgspikecounts': [avgspikecounts],
                'speed_values': [speed_all.astype(float)],
                'pupil_values': [pupil_all.astype(float)],
            })
            stats_list.append(dict_stats)

    return pd.DataFrame(stats_list)

# %%
# ------------------------------------
# MAIN: COMPUTE BRAIN AREAS STATISTICS
# ------------------------------------
# session_number: indicating whether it was the first or second recording day for the mouse
# experience level: tells you whether the image set used for a particular recording was the same
# as the training image set (Familiar), or different from the training image set (Novel).
if __name__ == '__main__':
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    df_sessions = pd.read_pickle(ALLEN_DATA_DIR / 'df_sessions.pk')
    df_sessions = df_sessions.query(
        'session_number==1 and experience_level=="Familiar" and image_set=="G"'
    )

    session_meta = df_sessions[
        ['ecephys_session_id', 'session_number', 'experience_level', 'image_set']
    ].values.tolist()

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(process_session_general, session_meta)

    df_stats = pd.concat(results, ignore_index=True)
    df_stats.to_pickle(data_dir / 'df_stats_area.pk')
    print(f"Saved {data_dir / 'df_stats_area.pk'}")
# %%
# Dummy test plot
import seaborn as sns
data_dir = Path(__file__).resolve().parent.parent / 'data'
load_df = pd.read_pickle(data_dir / 'df_stats_area.pk')
sns.catplot(data=load_df, x='area', y='PR_emp', hue='active', kind='box')
# %%