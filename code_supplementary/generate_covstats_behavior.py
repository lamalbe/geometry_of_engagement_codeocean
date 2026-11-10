# %%
"""Covariance stats split by unengagement, speed terciles, and pupil terciles.

Day-1 Familiar sessions. Viewing bins use time_from_stimulus >= 0.1 s;
omitted, novel, and change presentations dropped.

Needs data/df_behavior.pk (from suppl_behavior_session.py).
If df_unengagement_summary.pk and df_view_terciles.pk are missing, they are
built first. Covariance tables are written to data/.
"""
import json
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

import config

ALLEN_DATA_DIR = config.ALLEN_PREPROCESSED_DIR
TIME_FROM_STIMULUS_MIN = 0.1
MAX_TIME = 60 * 60 + 20
LICK_BIN_SEC = 1
SMOOTH_SEC = 60
REL_THRESH = 0.20
MIN_DISENGAGED_SEC = 90
GROUP_SIZE = 200
UNENG_TIME = 12 * 60  # s; min overall unengaged time to label unengaged=True
N_WORKERS = 4
FORCE_REBUILD_BEHAVIOR_TABLES = False

DATA_DIR = config.DATA_DIR
BEHAVIOR_FILE = DATA_DIR / "df_behavior.pk"
UNENG_SUMMARY_FILE = DATA_DIR / "df_unengagement_summary.pk"
VIEW_TERCILES_FILE = DATA_DIR / "df_view_terciles.pk"
STATS_UNENG_FILE = DATA_DIR / "suppl_stats_unengagement.pk"
STATS_SPEED_FILE = DATA_DIR / "suppl_stats_speed_terciles.pk"
STATS_PUPIL_FILE = DATA_DIR / "suppl_stats_pupil_terciles.pk"

AREAS_OF_INTEREST = [
    ("VISp", ["VISp"]),
    ("VISrl", ["VISrl"]),
    ("VISl", ["VISl"]),
    ("VISal", ["VISal"]),
    ("VISpm", ["VISpm"]),
    ("VISam", ["VISam"]),
    ("VISpl", ["VISp", "VISrl", "VISl", "VISal"]),
    ("VISm", ["VISpm", "VISam"]),
    ("LGd", ["LGd"]),
    ("SCMRN", ["MRN", "SCig", "SCiw"]),
]

UNENG_BY_SESSION = {}
UNENG_TOTAL_BY_SESSION = {}
VIEW_TERCILES = {}


# %%
# --------------------------------------------------------------------
# BEHAVIOR-TABLE HELPERS (first part)
# --------------------------------------------------------------------

def merge_dicts(series):
    merged = {}
    for item in series.dropna():
        if isinstance(item, dict):
            merged.update(item)
    return merged


def filter_dict_by_time_active_block(d, active, max_time=MAX_TIME):
    """Keep only the first MAX_TIME for active=True; keep everything for passive."""
    if not isinstance(d, dict):
        return {}

    filtered = {}
    for k, v in d.items():
        try:
            t = float(k)
            if not np.isfinite(t):
                continue
            if active is True and t > max_time:
                continue
            filtered[k] = v
        except (TypeError, ValueError):
            continue
    return filtered


def clip_time_by_active(x, active, max_time=MAX_TIME):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(x):
        return np.nan
    if active is True:
        return min(x, max_time)
    return x


def dict_to_arrays(d, drop_below=None):
    times, values = [], []
    if not isinstance(d, dict):
        return np.array([]), np.array([])

    for t, v in d.items():
        try:
            t = float(t)
            v = float(v)
            if np.isfinite(t) and np.isfinite(v):
                if drop_below is None or v >= drop_below:
                    times.append(t)
                    values.append(v)
        except (TypeError, ValueError):
            continue

    if not times:
        return np.array([]), np.array([])
    idx = np.argsort(times)
    return np.array(times)[idx], np.array(values)[idx]


def dict_to_event_times(d, positive_only=True):
    times = []
    if not isinstance(d, dict):
        return np.array([])

    for t, v in d.items():
        try:
            t = float(t)
            v = float(v)
            if not (np.isfinite(t) and np.isfinite(v)):
                continue
            if (not positive_only) or (v > 0):
                times.append(t)
        except (TypeError, ValueError):
            continue

    if not times:
        return np.array([])
    return np.sort(np.array(times, dtype=float))


def compute_smoothed_event_rate(event_times, t_start, t_end, bin_sec=1, smooth_sec=60):
    if t_end <= t_start:
        return np.array([]), np.array([]), np.array([])

    bin_edges = np.arange(t_start, t_end + bin_sec, bin_sec)
    if len(bin_edges) < 2:
        return np.array([]), np.array([]), np.array([])

    counts, _ = np.histogram(event_times, bins=bin_edges)
    event_rate = counts / bin_sec
    win = max(1, int(round(smooth_sec / bin_sec)))
    kernel = np.ones(win) / win
    event_rate_smooth = np.convolve(event_rate, kernel, mode="same")
    t_centers = bin_edges[:-1] + bin_sec / 2
    return t_centers, event_rate, event_rate_smooth


def extract_disengagement_intervals(
    lick_times,
    t_start,
    t_end,
    bin_sec=1,
    smooth_sec=60,
    rel_thresh=0.20,
    min_interval_sec=90,
):
    t_centers, _lick_rate, lick_rate_smooth = compute_smoothed_event_rate(
        lick_times, t_start, t_end, bin_sec=bin_sec, smooth_sec=smooth_sec
    )
    if len(t_centers) == 0:
        return [], np.nan

    peak = np.max(lick_rate_smooth) if len(lick_rate_smooth) else 0.0
    if peak <= 0:
        return [(t_start, t_end)], 0.0

    threshold = rel_thresh * peak
    disengaged = lick_rate_smooth < threshold

    intervals = []
    in_interval = False
    start = None
    for i, flag in enumerate(disengaged):
        t = t_centers[i]
        if flag and not in_interval:
            start = t
            in_interval = True
        elif not flag and in_interval:
            end = t
            if end - start >= min_interval_sec:
                intervals.append((start, end))
            in_interval = False
            start = None
    if in_interval:
        end = t_centers[-1]
        if end - start >= min_interval_sec:
            intervals.append((start, end))
    return intervals, threshold


def intervals_to_dict(intervals, decimals=3):
    return {round(float(a), decimals): round(float(b), decimals) for a, b in intervals}


def total_interval_duration(intervals):
    if not intervals:
        return 0.0
    return float(sum(b - a for a, b in intervals))


def clean_positive(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return values[values > 0]


def terciles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan, np.nan
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    return float(q1), float(q2)


def behavior_tables_available():
    return UNENG_SUMMARY_FILE.exists() and VIEW_TERCILES_FILE.exists()


def build_behavior_summary_tables():
    """From df_behavior.pk: unengagement intervals and viewing speed/pupil terciles."""
    if not BEHAVIOR_FILE.exists():
        raise FileNotFoundError(
            f"Missing {BEHAVIOR_FILE}. Run suppl_behavior_session.py first."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_behavior_all = pd.read_pickle(BEHAVIOR_FILE)
    df_behavior_all = df_behavior_all[
        (df_behavior_all["experience_level"] == "Familiar")
        & (df_behavior_all["session_number"] == 1)
    ].copy()

    cols_to_filter = [
        "speed_values",
        "pupil_values",
        "licks_values",
        "rewards_values",
        "auto_rew_values",
    ]

    mask_uneng = (
        (df_behavior_all["view_flag"] == False)
        & (df_behavior_all["active"] == True)
    )
    df_uneng = df_behavior_all[mask_uneng].copy()
    for col in cols_to_filter:
        if col in df_uneng.columns:
            df_uneng[col] = df_uneng.apply(
                lambda row: filter_dict_by_time_active_block(
                    row[col], row["active"], max_time=MAX_TIME
                ),
                axis=1,
            )
    df_uneng["start_time"] = df_uneng.apply(
        lambda row: clip_time_by_active(row["start_time"], row["active"], max_time=MAX_TIME),
        axis=1,
    )
    df_uneng["end_time"] = df_uneng.apply(
        lambda row: clip_time_by_active(row["end_time"], row["active"], max_time=MAX_TIME),
        axis=1,
    )

    uneng_rows = []
    for session_id, df_ses in df_uneng.groupby("session_id", dropna=False):
        licks_dict = merge_dicts(df_ses["licks_values"])
        lick_t = dict_to_event_times(licks_dict, positive_only=True)
        t_start = pd.to_numeric(df_ses["start_time"], errors="coerce").min()
        t_end = pd.to_numeric(df_ses["end_time"], errors="coerce").max()
        if pd.isna(t_start) or pd.isna(t_end):
            intervals, threshold = [], np.nan
        else:
            intervals, threshold = extract_disengagement_intervals(
                lick_t,
                t_start=float(t_start),
                t_end=float(t_end),
                bin_sec=LICK_BIN_SEC,
                smooth_sec=SMOOTH_SEC,
                rel_thresh=REL_THRESH,
                min_interval_sec=MIN_DISENGAGED_SEC,
            )
        uneng_rows.append(
            {
                "session_id": session_id,
                "active": True,
                "view_flag": False,
                "unengagement_intervals": intervals_to_dict(intervals),
                "overall_unengagement_time": total_interval_duration(intervals),
                "n_unengagement_intervals": len(intervals),
                "lick_threshold": threshold,
            }
        )
    df_unengagement_summary = pd.DataFrame(uneng_rows)

    mask_view = df_behavior_all["view_flag"] == True
    df_view = df_behavior_all[mask_view].copy()
    for col in cols_to_filter:
        if col in df_view.columns:
            df_view[col] = df_view.apply(
                lambda row: filter_dict_by_time_active_block(
                    row[col], row["active"], max_time=MAX_TIME
                ),
                axis=1,
            )

    tercile_rows = []
    for (session_id, active), df_ses in df_view.groupby(["session_id", "active"], dropna=False):
        if active not in (True, False):
            continue
        _, speed_v = dict_to_arrays(merge_dicts(df_ses["speed_values"]), drop_below=0)
        _, pupil_v = dict_to_arrays(merge_dicts(df_ses["pupil_values"]))
        speed_v = clean_positive(speed_v)
        pupil_v = clean_positive(pupil_v)
        speed_t1, speed_t2 = terciles(speed_v)
        pupil_t1, pupil_t2 = terciles(pupil_v)
        tercile_rows.append(
            {
                "session_id": session_id,
                "active": active,
                "view_flag": True,
                "speed_tercile_1": speed_t1,
                "speed_tercile_2": speed_t2,
                "pupil_tercile_1": pupil_t1,
                "pupil_tercile_2": pupil_t2,
                "n_speed_samples": int(len(speed_v)),
                "n_pupil_samples": int(len(pupil_v)),
            }
        )
    df_view_terciles = pd.DataFrame(tercile_rows)

    df_unengagement_summary.to_pickle(UNENG_SUMMARY_FILE)
    df_view_terciles.to_pickle(VIEW_TERCILES_FILE)
    print(f"Saved {UNENG_SUMMARY_FILE}")
    print(f"Saved {VIEW_TERCILES_FILE}")
    return df_unengagement_summary, df_view_terciles


def load_behavior_lookups():
    global UNENG_BY_SESSION, UNENG_TOTAL_BY_SESSION, VIEW_TERCILES

    df_unengagement_summary = pd.read_pickle(UNENG_SUMMARY_FILE)
    df_view_terciles = pd.read_pickle(VIEW_TERCILES_FILE)

    uneng_by_session = {}
    uneng_total_by_session = {}
    for _, row in df_unengagement_summary.iterrows():
        session_str = str(row["session_id"])
        interval_obj = (
            row["unengagement_intervals"]
            if "unengagement_intervals" in row
            else row.get("unengagement_intervals_json", {})
        )
        uneng_by_session[session_str] = parse_intervals(interval_obj)
        uneng_total_by_session[session_str] = float(
            row.get("overall_unengagement_time", 0.0)
        )

    view_terciles = {}
    for _, row in df_view_terciles.iterrows():
        session_str = str(row["session_id"])
        active = bool(row["active"])
        view_terciles[(session_str, active)] = {
            "speed_t1": row["speed_tercile_1"],
            "speed_t2": row["speed_tercile_2"],
            "pupil_t1": row["pupil_tercile_1"],
            "pupil_t2": row["pupil_tercile_2"],
        }

    UNENG_BY_SESSION = uneng_by_session
    UNENG_TOTAL_BY_SESSION = uneng_total_by_session
    VIEW_TERCILES = view_terciles


def ensure_behavior_tables():
    if FORCE_REBUILD_BEHAVIOR_TABLES or not behavior_tables_available():
        print("Building unengagement / tercile tables from df_behavior.pk")
        build_behavior_summary_tables()
    else:
        print(f"Using existing tables:\n  {UNENG_SUMMARY_FILE}\n  {VIEW_TERCILES_FILE}")
    load_behavior_lookups()


# %%
# --------------------------------------------------------------------
# COVARIANCE STATISTICS
# --------------------------------------------------------------------

def statsvalues2PR(mii, mij, sii, sij, N=10**4):
    return 1 / (
        1 / N
        + 1 / N * (sii / mii) ** 2
        + (N - 1) / N * (sij / mii) ** 2
        + (N - 1) / N * (mij / mii) ** 2
    )


def cov_stats(spkcounts):
    """spkcounts: array of shape (n_neurons, n_samples)."""
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
    TV_emp = np.trace(C)
    PR_emp = (np.trace(C) ** 2) / np.trace(C @ C)
    PR_fit = statsvalues2PR(mii, mij, sii, sij)
    return {
        "mii": mii,
        "mij": mij,
        "sii": sii,
        "sij": sij,
        "TV_emp": TV_emp,
        "PR_emp": PR_emp,
        "PR_fit": PR_fit,
    }


def parse_intervals(interval_obj):
    if isinstance(interval_obj, dict):
        raw = interval_obj
    elif isinstance(interval_obj, str):
        try:
            raw = json.loads(interval_obj)
        except Exception:
            return []
    else:
        return []

    intervals = []
    for k, v in raw.items():
        try:
            a = float(k)
            b = float(v)
            if np.isfinite(a) and np.isfinite(b) and b >= a:
                intervals.append((a, b))
        except (TypeError, ValueError):
            continue
    intervals.sort(key=lambda x: x[0])
    return intervals


def point_in_intervals(t, intervals):
    if not np.isfinite(t):
        return False
    for a, b in intervals:
        if a <= t <= b:
            return True
    return False


def midpoint_from_start_end(row):
    try:
        a = float(row["bin_start_times"])
        b = float(row["bin_end_times"])
        if np.isfinite(a) and np.isfinite(b):
            return 0.5 * (a + b)
    except Exception:
        pass
    return np.nan


def mean_dict_value(d, positive_only=False):
    if not isinstance(d, dict) or len(d) == 0:
        return np.nan
    vals = []
    for v in d.values():
        try:
            x = float(v)
            if np.isfinite(x) and ((not positive_only) or (x > 0)):
                vals.append(x)
        except (TypeError, ValueError):
            continue
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def assign_tercile(value, t1, t2):
    if not np.isfinite(value) or not np.isfinite(t1) or not np.isfinite(t2):
        return np.nan
    if value <= t1:
        return "low"
    if value <= t2:
        return "mid"
    return "high"


def compute_subsampled_stats(
    spikecounts_all,
    group_metadata,
    cov_stats_func,
    area,
    group_size=GROUP_SIZE,
):
    """spikecounts_all: array of shape (n_samples, n_neurons)."""
    if spikecounts_all.ndim != 2:
        return []
    n_rows, n_neurons = spikecounts_all.shape
    n_groups = n_rows // group_size
    if n_groups == 0:
        return []

    stats_list = []
    for chunk_idx in range(n_groups):
        idx_chunk = np.arange(chunk_idx * group_size, (chunk_idx + 1) * group_size)
        submatrix = spikecounts_all[idx_chunk, :]
        dict_stats = cov_stats_func(submatrix.T)
        if len(dict_stats) == 0:
            continue
        out = dict(dict_stats)
        out.update(group_metadata)
        out.update(
            {
                "area": area,
                "N": n_neurons,
                "subsample_id": chunk_idx,
                "avgspikecounts": [np.mean(submatrix, axis=0)],
                "n_bins_used": len(idx_chunk),
            }
        )
        stats_list.append(out)
    return stats_list


def session_paths(session_id):
    session_str = str(session_id)
    base = ALLEN_DATA_DIR / "day1" / session_str
    return session_str, base / "df_activity.pk", base / "df_units.pk"


def viewing_activity(df_activity_ses, active_only=False):
    mask = (
        (df_activity_ses["is_viewing"] == True)
        & (df_activity_ses["time_from_stimulus"] >= TIME_FROM_STIMULUS_MIN)
        & (df_activity_ses["image_name"] != "omitted")
        & (df_activity_ses["is_image_novel"] == False)
        & (df_activity_ses["is_change"] == False)
    )
    if active_only:
        mask = mask & (df_activity_ses["active"] == True)
    return df_activity_ses.loc[mask].copy()


# %%
# --------------------------------------------------------------------
# SESSION PROCESSORS
# --------------------------------------------------------------------

def process_session_unengagement(session_id, session_number, experience_level):
    """Cov stats during viewing, labeled by lick-rate unengagement intervals."""
    session_str, activity_file, units_file = session_paths(session_id)
    if not activity_file.exists() or not units_file.exists():
        print(f"[unengagement] Missing files for {session_str}")
        return pd.DataFrame()

    df_activity_ses = pd.read_pickle(activity_file)
    df_units_ses = pd.read_pickle(units_file)
    df_activity_ses["session_id"] = session_str
    df_units_ses["session_id"] = session_str
    print(f"[unengagement] Processing session {session_str}")

    intervals = UNENG_BY_SESSION.get(session_str, [])
    overall_uneng_time = UNENG_TOTAL_BY_SESSION.get(session_str, 0.0)
    session_has_unengaged_state = overall_uneng_time >= UNENG_TIME

    df_activity_ses = viewing_activity(df_activity_ses, active_only=True)
    if df_activity_ses.empty:
        return pd.DataFrame()

    df_activity_ses["bin_midpoint"] = df_activity_ses.apply(midpoint_from_start_end, axis=1)
    if session_has_unengaged_state:
        df_activity_ses["unengaged"] = df_activity_ses["bin_midpoint"].apply(
            lambda t: point_in_intervals(t, intervals)
        )
    else:
        df_activity_ses["unengaged"] = False

    stats_list = []
    for area_name, acronyms in AREAS_OF_INTEREST:
        df_units_area = df_units_ses[df_units_ses["structure_acronym"].isin(acronyms)]
        if df_units_area.shape[0] < 10:
            continue
        idx_units_area = df_units_area.index.values
        df_local = df_activity_ses.copy()
        df_local["spikecounts_area"] = df_local["spikecounts"].apply(
            lambda x: x[idx_units_area]
        )
        for (unengaged_flag, image_name), df_group in df_local.groupby(
            ["unengaged", "image_name"], dropna=False
        ):
            if unengaged_flag not in (True, False) or len(df_group) < GROUP_SIZE:
                continue
            spikecounts_all = np.stack(df_group["spikecounts_area"].values, axis=0)
            meta = {
                "session_id": session_str,
                "session_number": session_number,
                "experience_level": experience_level,
                "image_name": image_name,
                "active": True,
                "view_flag": True,
                "unengaged": bool(unengaged_flag),
                "overall_unengagement_time": overall_uneng_time,
                "session_has_unengaged_state": bool(session_has_unengaged_state),
                "behavior_metric": "unengagement",
            }
            stats_list.extend(
                compute_subsampled_stats(
                    spikecounts_all, meta, cov_stats, area_name, GROUP_SIZE
                )
            )
    return pd.DataFrame(stats_list)


def process_session_speed_terciles(session_id, session_number, experience_level):
    session_str, activity_file, units_file = session_paths(session_id)
    if not activity_file.exists() or not units_file.exists():
        print(f"[speed terciles] Missing files for {session_str}")
        return pd.DataFrame()

    df_activity_ses = pd.read_pickle(activity_file)
    df_units_ses = pd.read_pickle(units_file)
    df_activity_ses["session_id"] = session_str
    df_units_ses["session_id"] = session_str
    print(f"[speed terciles] Processing session {session_str}")

    df_activity_ses = viewing_activity(df_activity_ses)
    if df_activity_ses.empty:
        return pd.DataFrame()

    df_activity_ses["mean_speed"] = df_activity_ses["speed"].apply(
        lambda d: mean_dict_value(d, positive_only=True)
    )

    stats_list = []
    for active_flag, df_active in df_activity_ses.groupby("active", dropna=False):
        if active_flag not in (True, False):
            continue
        terc = VIEW_TERCILES.get((session_str, bool(active_flag)))
        if terc is None:
            continue
        df_active = df_active.copy()
        df_active["tercile"] = df_active["mean_speed"].apply(
            lambda x: assign_tercile(x, terc["speed_t1"], terc["speed_t2"])
        )
        df_active = df_active[df_active["tercile"].isin(["low", "mid", "high"])].copy()
        if df_active.empty:
            continue

        for area_name, acronyms in AREAS_OF_INTEREST:
            df_units_area = df_units_ses[df_units_ses["structure_acronym"].isin(acronyms)]
            if df_units_area.shape[0] < 10:
                continue
            idx_units_area = df_units_area.index.values
            df_local = df_active.copy()
            df_local["spikecounts_area"] = df_local["spikecounts"].apply(
                lambda x: x[idx_units_area]
            )
            for (tercile_label, image_name), df_group in df_local.groupby(
                ["tercile", "image_name"], dropna=False
            ):
                if tercile_label not in ("low", "mid", "high") or len(df_group) < GROUP_SIZE:
                    continue
                spikecounts_all = np.stack(df_group["spikecounts_area"].values, axis=0)
                meta = {
                    "session_id": session_str,
                    "session_number": session_number,
                    "experience_level": experience_level,
                    "image_name": image_name,
                    "active": bool(active_flag),
                    "view_flag": True,
                    "tercile": tercile_label,
                    "behavior_metric": "speed",
                }
                stats_list.extend(
                    compute_subsampled_stats(
                        spikecounts_all, meta, cov_stats, area_name, GROUP_SIZE
                    )
                )
    return pd.DataFrame(stats_list)


def process_session_pupil_terciles(session_id, session_number, experience_level):
    session_str, activity_file, units_file = session_paths(session_id)
    if not activity_file.exists() or not units_file.exists():
        print(f"[pupil terciles] Missing files for {session_str}")
        return pd.DataFrame()

    df_activity_ses = pd.read_pickle(activity_file)
    df_units_ses = pd.read_pickle(units_file)
    df_activity_ses["session_id"] = session_str
    df_units_ses["session_id"] = session_str
    print(f"[pupil terciles] Processing session {session_str}")

    df_activity_ses = viewing_activity(df_activity_ses)
    if df_activity_ses.empty:
        return pd.DataFrame()

    df_activity_ses["mean_pupil"] = df_activity_ses["pupil_area"].apply(
        lambda d: mean_dict_value(d, positive_only=True)
    )

    stats_list = []
    for active_flag, df_active in df_activity_ses.groupby("active", dropna=False):
        if active_flag not in (True, False):
            continue
        terc = VIEW_TERCILES.get((session_str, bool(active_flag)))
        if terc is None:
            continue
        df_active = df_active.copy()
        df_active["tercile"] = df_active["mean_pupil"].apply(
            lambda x: assign_tercile(x, terc["pupil_t1"], terc["pupil_t2"])
        )
        df_active = df_active[df_active["tercile"].isin(["low", "mid", "high"])].copy()
        if df_active.empty:
            continue

        for area_name, acronyms in AREAS_OF_INTEREST:
            df_units_area = df_units_ses[df_units_ses["structure_acronym"].isin(acronyms)]
            if df_units_area.shape[0] < 10:
                continue
            idx_units_area = df_units_area.index.values
            df_local = df_active.copy()
            df_local["spikecounts_area"] = df_local["spikecounts"].apply(
                lambda x: x[idx_units_area]
            )
            for (tercile_label, image_name), df_group in df_local.groupby(
                ["tercile", "image_name"], dropna=False
            ):
                if tercile_label not in ("low", "mid", "high") or len(df_group) < GROUP_SIZE:
                    continue
                spikecounts_all = np.stack(df_group["spikecounts_area"].values, axis=0)
                meta = {
                    "session_id": session_str,
                    "session_number": session_number,
                    "experience_level": experience_level,
                    "image_name": image_name,
                    "active": bool(active_flag),
                    "view_flag": True,
                    "tercile": tercile_label,
                    "behavior_metric": "pupil",
                }
                stats_list.extend(
                    compute_subsampled_stats(
                        spikecounts_all, meta, cov_stats, area_name, GROUP_SIZE
                    )
                )
    return pd.DataFrame(stats_list)


def _concat_results(results):
    frames = [df for df in results if df is not None and not df.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# %%
# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ensure_behavior_tables()

    df_sessions = pd.read_pickle(ALLEN_DATA_DIR / "df_sessions.pk")
    df_sessions = df_sessions.query(
        'session_number == 1 and experience_level == "Familiar"'
    ).copy()
    session_meta = df_sessions[
        ["ecephys_session_id", "session_number", "experience_level"]
    ].values.tolist()
    print(f"Processing {len(session_meta)} day-1 Familiar sessions")

    skip_unengagement = os.environ.get("SKIP_UNENGAGEMENT", "0") == "1"
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        if skip_unengagement:
            print("Skipping unengagement rebuild (SKIP_UNENGAGEMENT=1)")
            results_uneng = []
        else:
            results_uneng = pool.starmap(process_session_unengagement, session_meta)
        results_speed = pool.starmap(process_session_speed_terciles, session_meta)
        results_pupil = pool.starmap(process_session_pupil_terciles, session_meta)

    df_stats_speed_terciles = _concat_results(results_speed)
    df_stats_pupil_terciles = _concat_results(results_pupil)

    if not skip_unengagement:
        df_stats_unengagement = _concat_results(results_uneng)
        df_stats_unengagement.to_pickle(STATS_UNENG_FILE)
    df_stats_speed_terciles.to_pickle(STATS_SPEED_FILE)
    df_stats_pupil_terciles.to_pickle(STATS_PUPIL_FILE)

    print("Saved:")
    if not skip_unengagement:
        print(f"  {STATS_UNENG_FILE} ({len(df_stats_unengagement)} rows)")
    print(f"  {STATS_SPEED_FILE} ({len(df_stats_speed_terciles)} rows)")
    print(f"  {STATS_PUPIL_FILE} ({len(df_stats_pupil_terciles)} rows)")
# %%
