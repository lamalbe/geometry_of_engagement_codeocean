# %%
"""Supplementary behavior traces and session-level speed/pupil/lick summaries.

Day-1 Familiar sessions (image sets G and H). Viewing bins use
time_from_stimulus >= 0.1 s; omitted, novel, and change presentations dropped.

Reads:  /data/allen_engagement_data/day1/<session_id>/df_activity.pk
Writes: clean_scripts/data/df_behavior.pk
        clean_scripts/results/*.pdf
"""
import multiprocessing
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ALLEN_DATA_DIR = Path("/data/allen_engagement_data")
TIME_FROM_STIMULUS_MIN = 0.1
MAX_TIME = 60 * 60 + 20  # seconds

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

DATA_DIR = _SCRIPT_DIR.parent / "data"
RESULTS_DIR = _SCRIPT_DIR.parent / "results"
SAVE_PLOTS = True


def save_fig(fig, filename):
    if not SAVE_PLOTS:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / filename, format="pdf", bbox_inches="tight")


# %%
# --------------------------------------------------------------------
# SESSION PROCESSING
# --------------------------------------------------------------------

def merge_dicts(series):
    merged = {}

    for item in series.dropna():
        if isinstance(item, dict):
            merged.update(item)

    return merged


def safe_min(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return vals.min() if len(vals) else np.nan


def safe_max(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return vals.max() if len(vals) else np.nan


def process_session_general(session_id, session_number, experience_level, image_set, view_flag=False):
    session_str = str(session_id)
    activity_file = ALLEN_DATA_DIR / "day1" / session_str / "df_activity.pk"
    if not activity_file.exists():
        print(f"Missing activity file: {activity_file}")
        return pd.DataFrame()

    df_activity_ses = pd.read_pickle(activity_file)
    print(f"Processing session {session_str} (view_flag={view_flag})")

    if view_flag:
        df_activity_ses = df_activity_ses[df_activity_ses["is_viewing"] == True].copy()
        df_activity_ses = df_activity_ses[
            df_activity_ses["time_from_stimulus"] >= TIME_FROM_STIMULUS_MIN
        ].copy()
        df_activity_ses = df_activity_ses[df_activity_ses["image_name"] != "omitted"].copy()
        df_activity_ses = df_activity_ses[df_activity_ses["is_image_novel"] == False].copy()
        df_activity_ses = df_activity_ses[df_activity_ses["is_change"] == False].copy()

    cols = [
        "bin_start_times",
        "bin_end_times",
        "active",
        "stimulus_block",
        "speed",
        "pupil_area",
        "licks",
        "rewards",
        "auto_rew",
    ]
    df_activity_ses = df_activity_ses[cols].copy()
    df_activity_ses["session_id"] = session_str

    grouped_rows = []

    for (active, stimulus_block), df_group in df_activity_ses.groupby(
        ["active", "stimulus_block"], dropna=False
    ):
        row = {
            "session_number": session_number,
            "experience_level": experience_level,
            "image_set": image_set,
            "session_id": session_str,
            "active": active,
            "stimulus_block": stimulus_block,
            "view_flag": view_flag,
            "start_time": safe_min(df_group["bin_start_times"]),
            "end_time": safe_max(df_group["bin_end_times"]),
            "speed_values": merge_dicts(df_group["speed"]),
            "pupil_values": merge_dicts(df_group["pupil_area"]),
            "licks_values": merge_dicts(df_group["licks"]),
            "rewards_values": merge_dicts(df_group["rewards"]),
            "auto_rew_values": merge_dicts(df_group["auto_rew"]),
        }
        grouped_rows.append(row)

    return pd.DataFrame(grouped_rows)

# --------------------------------------------------------------------
# HELPERS: BEHAVIOR ACTIVE (LICKS) & AVERAGE METRICS
# --------------------------------------------------------------------


def filter_dict_by_time(d):
    if not isinstance(d, dict):
        return {}

    filtered = {}
    for k, v in d.items():
        try:
            t = float(k)
            if np.isfinite(t) and t <= MAX_TIME:
                filtered[k] = v
        except (TypeError, ValueError):
            continue

    return filtered


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
    """
    Build a 1D event-rate signal over time from event timestamps.

    Returns
    -------
    t_centers : array
        Time points in seconds, one per bin.
    event_rate : array
        Raw event rate in events/s.
    event_rate_smooth : array
        Moving-average-smoothed event rate in events/s.
    """
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
    """
    Disengagement = smoothed lick rate below rel_thresh * peak smoothed lick rate.

    Returns
    -------
    intervals_sec : list of tuples
        List of (start_sec, end_sec) disengagement intervals.
    t_centers : array
        Time points in seconds for the lick-rate signal.
    lick_rate : array
        Raw lick rate in licks/s.
    lick_rate_smooth : array
        Smoothed lick rate in licks/s.
    threshold : float
        Disengagement threshold in licks/s.
    """
    t_centers, lick_rate, lick_rate_smooth = compute_smoothed_event_rate(
        lick_times, t_start, t_end, bin_sec=bin_sec, smooth_sec=smooth_sec
    )

    if len(t_centers) == 0:
        return [], t_centers, lick_rate, lick_rate_smooth, np.nan

    peak = np.max(lick_rate_smooth) if len(lick_rate_smooth) else 0.0
    if peak <= 0:
        return [(t_start, t_end)], t_centers, lick_rate, lick_rate_smooth, 0.0

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

    return intervals, t_centers, lick_rate, lick_rate_smooth, threshold


def mean_positive(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if len(values) == 0:
        return np.nan
    return values.mean()


def compute_session_duration_from_dicts(*dicts):
    all_times = []

    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k in d.keys():
            try:
                t = float(k)
                if np.isfinite(t):
                    all_times.append(t)
            except (TypeError, ValueError):
                continue

    if len(all_times) < 2:
        return np.nan

    return float(np.max(all_times) - np.min(all_times))


def compute_average_event_rate(event_times, duration_sec):
    if not np.isfinite(duration_sec) or duration_sec <= 0:
        return np.nan
    return len(event_times) / duration_sec


def kde_rug_positive(ax, values, title, xlabel, xlim_right=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if len(values) == 0:
        ax.text(0.5, 0.5, "No positive data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_xlim(left=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return

    sns.kdeplot(x=values, ax=ax, fill=False, warn_singular=False)
    sns.rugplot(x=values, ax=ax, height=0.08)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_xlim(left=0)

    if xlim_right is not None and np.isfinite(xlim_right) and xlim_right > 0:
        ax.set_xlim(0, xlim_right)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def kde_with_terciles(ax, values, title, xlabel, *, nonnegative=False, xlim_right=None):
    """KDE only (no rug), with vertical lines at sample tercile boundaries (1/3 and 2/3 quantiles)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if nonnegative:
        values = values[values >= 0]

    if len(values) < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return

    sns.kdeplot(x=values, ax=ax, fill=False, warn_singular=False)
    t1, t2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    ax.axvline(t1, color="0.35", linestyle="--", linewidth=1.2)
    ax.axvline(t2, color="0.35", linestyle="--", linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if nonnegative:
        ax.set_xlim(left=0)
    if xlim_right is not None and np.isfinite(xlim_right) and xlim_right > 0:
        cur_left, _ = ax.get_xlim()
        ax.set_xlim(cur_left, xlim_right)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# %%
# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_sessions = pd.read_pickle(ALLEN_DATA_DIR / "df_sessions.pk")
    df_sessions = df_sessions.query(
        'session_number == 1 and experience_level == "Familiar"'
    ).copy()

    session_meta = df_sessions[
        ["ecephys_session_id", "session_number", "experience_level", "image_set"]
    ].values.tolist()
    print(f"Processing {len(session_meta)} day-1 Familiar sessions")

    with multiprocessing.Pool(processes=4) as pool:
        results_no_stim = pool.starmap(
            process_session_general,
            [(*meta, False) for meta in session_meta],
        )
        results_stim = pool.starmap(
            process_session_general,
            [(*meta, True) for meta in session_meta],
        )

    results = [
        df for df in (results_no_stim + results_stim)
        if df is not None and not df.empty
    ]
    df_behavior = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    keep_cols = [
        "session_number",
        "experience_level",
        "image_set",
        "session_id",
        "active",
        "stimulus_block",
        "view_flag",
        "start_time",
        "end_time",
        "speed_values",
        "pupil_values",
        "licks_values",
        "rewards_values",
        "auto_rew_values",
    ]
    df_behavior = df_behavior[[c for c in keep_cols if c in df_behavior.columns]]

    out_file = DATA_DIR / "df_behavior.pk"
    df_behavior.to_pickle(out_file)
    print(f"Saved {len(df_behavior)} rows to {out_file}")

# %%
# -----------------------------------------------------------------------
# BEHAVIOR ACTIVE ANALYSIS: LICKS DISPLAYING UNENGAGEMENT IN ACTIVE BLOCK
# -----------------------------------------------------------------------

df_behavior_all = pd.read_pickle(DATA_DIR / "df_behavior.pk")

data_filter = (
    (df_behavior_all.view_flag == False)
    & (df_behavior_all.active == True)
    & (df_behavior_all.experience_level == "Familiar")
    & (df_behavior_all.session_number == 1)
)
df_behavior_filtered = df_behavior_all[data_filter]

cols_to_filter = [
    "speed_values",
    "pupil_values",
    "licks_values",
    "rewards_values",
    "auto_rew_values",
]
df_behavior = df_behavior_filtered.copy()
for col in cols_to_filter:
    df_behavior[col] = df_behavior[col].apply(filter_dict_by_time)

df_behavior["start_time"] = df_behavior["start_time"].clip(upper=MAX_TIME)
df_behavior["end_time"] = df_behavior["end_time"].clip(upper=MAX_TIME)

# %%
# ============================================================
# SETTINGS
# ============================================================
PLOT_RANDOM_SESSION = True   # True: one random session, False: all sessions

LICK_BIN_SEC = 1              # time resolution of lick-rate signal
REWARD_BIN_SEC = 1            # time resolution of reward-rate signal
SMOOTH_SEC = 60               # moving-average window in seconds
REL_THRESH = 0.20             # disengaged if smoothed lick rate < 20% of peak
MIN_DISENGAGED_SEC = 90       # minimum duration of disengagement interval

# ============================================================
# SESSION SELECTION
# ============================================================
session_ids = df_behavior["session_id"].unique()
if PLOT_RANDOM_SESSION:
    session_ids = [random.choice(session_ids)]

# ============================================================
# PLOTTING LOOP
# ============================================================
for session_id in session_ids:
    df_ses = df_behavior[df_behavior["session_id"] == session_id]

    speed_dict = merge_dicts(df_ses["speed_values"])
    pupil_dict = merge_dicts(df_ses["pupil_values"])
    licks_dict = merge_dicts(df_ses["licks_values"])
    rewards_dict = merge_dicts(df_ses["rewards_values"])

    speed_t, speed_v = dict_to_arrays(speed_dict, drop_below=0)
    pupil_t, pupil_v = dict_to_arrays(pupil_dict)
    lick_t = dict_to_event_times(licks_dict, positive_only=True)
    reward_t = dict_to_event_times(rewards_dict, positive_only=True)

    t_start = float(df_ses["start_time"].min())
    t_end = float(df_ses["end_time"].max())

    intervals_sec, lick_rate_t, lick_rate_raw, lick_rate_smooth, lick_thresh = extract_disengagement_intervals(
        lick_t,
        t_start=t_start,
        t_end=t_end,
        bin_sec=LICK_BIN_SEC,
        smooth_sec=SMOOTH_SEC,
        rel_thresh=REL_THRESH,
        min_interval_sec=MIN_DISENGAGED_SEC,
    )

    reward_rate_t, reward_rate_raw, reward_rate_smooth = compute_smoothed_event_rate(
        reward_t,
        t_start=t_start,
        t_end=t_end,
        bin_sec=REWARD_BIN_SEC,
        smooth_sec=SMOOTH_SEC,
    )

    print(f"\nSession {session_id}")
    if intervals_sec:
        print("Disengagement intervals (s):")
        for a, b in intervals_sec:
            print(f"  {a:.1f} - {b:.1f}")
    else:
        print("No disengagement intervals detected.")

    fig, axes = plt.subplots(
        7, 1,
        figsize=(22, 12),
        sharex=True,
        gridspec_kw={
            "height_ratios": [0.6, 1.2, 1.2, 0.6, 0.6, 1.2, 1.2],
            "hspace": 0.12,
        },
    )

    (
        ax_rewards,
        ax_rewardrate,
        ax_lickrate,
        ax_licks_disc,
        ax_licks,
        ax_speed,
        ax_pupil,
    ) = axes

    if len(reward_t):
        ax_rewards.vlines(reward_t, 0, 1, color="tab:orange", linewidth=1.2)
    ax_rewards.set_ylim(0, 1)
    ax_rewards.set_yticks([])
    ax_rewards.set_ylabel("rewards", rotation=0, labelpad=35, va="center")

    for a, b in intervals_sec:
        ax_rewardrate.axvspan(a, b, alpha=0.20, color="red")
    if len(reward_rate_t):
        ax_rewardrate.plot(reward_rate_t, reward_rate_smooth, linewidth=1.5)
    ax_rewardrate.set_ylabel("reward rate")

    for a, b in intervals_sec:
        ax_lickrate.axvspan(a, b, alpha=0.20, color="red")
    if len(lick_rate_t):
        ax_lickrate.plot(lick_rate_t, lick_rate_smooth, linewidth=1.5)
        ax_lickrate.axhline(lick_thresh, linestyle="--", linewidth=1.0)
    ax_lickrate.set_ylabel("lick rate")

    for a, b in intervals_sec:
        ax_licks_disc.axvspan(a, b, alpha=0.25, color="red")
    if len(lick_t):
        ax_licks_disc.vlines(lick_t, 0, 1, color="0.25", linewidth=0.8)
    ax_licks_disc.set_ylim(0, 1)
    ax_licks_disc.set_yticks([])
    ax_licks_disc.set_ylabel("licks\n+ dis.", rotation=0, labelpad=35, va="center")

    if len(lick_t):
        ax_licks.vlines(lick_t, 0, 1, color="0.25", linewidth=0.8)
    ax_licks.set_ylim(0, 1)
    ax_licks.set_yticks([])
    ax_licks.set_ylabel("licks", rotation=0, labelpad=35, va="center")

    if len(speed_t):
        ax_speed.plot(speed_t, speed_v, linewidth=1.1)
    ax_speed.set_ylabel("speed")

    if len(pupil_t):
        ax_pupil.plot(pupil_t, pupil_v, linewidth=1.1)
    ax_pupil.set_ylabel("pupil")

    for ax in axes:
        ax.set_xlim(t_start, t_end)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_pupil.set_xlabel("time (s)")

    plt.suptitle(
        f"Session {session_id} | disengagement from smoothed lick rate "
        f"(window={SMOOTH_SEC}s, threshold={REL_THRESH:.2f}*peak, min_len={MIN_DISENGAGED_SEC}s)",
        y=0.98,
    )
    save_fig(fig, f"session_{session_id}_lick_rate.pdf")
    plt.show()

# %%
# --------------------------------------------------------------------
# VIEWING (view_flag=True): SPEED & PUPIL KDE WITH TERCILES PER SESSION × ACTIVE
# --------------------------------------------------------------------
sns.set_theme(style="white")

data_filter_view = (
    (df_behavior_all.view_flag == True)
    & (df_behavior_all.experience_level == "Familiar")
    & (df_behavior_all.session_number == 1)
)
df_view = df_behavior_all[data_filter_view].copy()

for (session_id, active), df_ses in df_view.groupby(["session_id", "active"], dropna=False):
    if active not in (True, False):
        continue

    speed_dict = merge_dicts(df_ses["speed_values"])
    pupil_dict = merge_dicts(df_ses["pupil_values"])
    _, speed_v = dict_to_arrays(speed_dict, drop_below=0)
    _, pupil_v = dict_to_arrays(pupil_dict)

    act_label = "active" if active else "passive"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    kde_with_terciles(
        axes[0],
        speed_v,
        title="Speed",
        xlabel="speed",
        nonnegative=True,
    )
    kde_with_terciles(
        axes[1],
        pupil_v,
        title="Pupil",
        xlabel="pupil",
    )

    plt.suptitle(
        f"session {session_id} | {act_label}\n"
        f"view_flag=True | terciles: dashed lines at 1/3 and 2/3 sample quantiles",
        y=1.06,
        fontsize=11,
    )
    plt.tight_layout()
    save_fig(fig, f"session_{session_id}_{act_label}_terciles.pdf")
    plt.show()

# %%
# --------------------------------------------------------------------
# AVERAGE METRICS
# --------------------------------------------------------------------

sns.set_theme(style="white")

data_filter = (
    (df_behavior_all.view_flag == False)
    & (df_behavior_all.experience_level == "Familiar")
    & (df_behavior_all.session_number == 1)
)
df_behavior = df_behavior_all[data_filter].copy()

summary_rows = []

for (session_id, active), df_ses in df_behavior.groupby(["session_id", "active"], dropna=False):
    speed_dict = merge_dicts(df_ses["speed_values"]) if "speed_values" in df_ses else {}
    pupil_dict = merge_dicts(df_ses["pupil_values"]) if "pupil_values" in df_ses else {}
    licks_dict = merge_dicts(df_ses["licks_values"]) if "licks_values" in df_ses else {}
    rewards_dict = merge_dicts(df_ses["rewards_values"]) if "rewards_values" in df_ses else {}

    _, speed_v = dict_to_arrays(speed_dict, drop_below=0)
    _, pupil_v = dict_to_arrays(pupil_dict)
    lick_t = dict_to_event_times(licks_dict, positive_only=True)
    reward_t = dict_to_event_times(rewards_dict, positive_only=True)

    duration_sec = compute_session_duration_from_dicts(
        speed_dict, pupil_dict, licks_dict, rewards_dict
    )

    summary_rows.append(
        {
            "session_id": session_id,
            "active": active,
            "avg_speed": mean_positive(speed_v),
            "avg_pupil": mean_positive(pupil_v),
            "avg_lick_rate": compute_average_event_rate(lick_t, duration_sec),
            "avg_reward_rate": compute_average_event_rate(reward_t, duration_sec),
            "duration_sec": duration_sec,
        }
    )

df_summary = pd.DataFrame(summary_rows)
df_summary = df_summary[df_summary["active"].isin([True, False])].copy()

print(df_summary.head())

df_active = df_summary[df_summary["active"] == True].copy()
df_passive = df_summary[df_summary["active"] == False].copy()

speed_vals_combined = pd.concat(
    [df_active["avg_speed"], df_passive["avg_speed"]], ignore_index=True
).to_numpy(dtype=float)
speed_vals_combined = speed_vals_combined[np.isfinite(speed_vals_combined)]
speed_vals_combined = speed_vals_combined[speed_vals_combined > 0]
speed_xmax = np.nanmax(speed_vals_combined) if len(speed_vals_combined) else None

pupil_vals_combined = pd.concat(
    [df_active["avg_pupil"], df_passive["avg_pupil"]], ignore_index=True
).to_numpy(dtype=float)
pupil_vals_combined = pupil_vals_combined[np.isfinite(pupil_vals_combined)]
pupil_vals_combined = pupil_vals_combined[pupil_vals_combined > 0]
pupil_xmax = np.nanmax(pupil_vals_combined) if len(pupil_vals_combined) else None

reward_vals_active = df_active["avg_reward_rate"].to_numpy(dtype=float)
reward_vals_active = reward_vals_active[np.isfinite(reward_vals_active)]
reward_vals_active = reward_vals_active[reward_vals_active > 0]
reward_xmax = np.nanmax(reward_vals_active) if len(reward_vals_active) else None

lick_vals_active = df_active["avg_lick_rate"].to_numpy(dtype=float)
lick_vals_active = lick_vals_active[np.isfinite(lick_vals_active)]
lick_vals_active = lick_vals_active[lick_vals_active > 0]
lick_xmax = np.nanmax(lick_vals_active) if len(lick_vals_active) else None

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()

kde_rug_positive(
    axes[0],
    df_active["avg_speed"].values,
    title="Active=True | Average speed",
    xlabel="average speed",
    xlim_right=speed_xmax,
)

kde_rug_positive(
    axes[1],
    df_active["avg_pupil"].values,
    title="Active=True | Average pupil",
    xlabel="average pupil",
    xlim_right=pupil_xmax,
)

kde_rug_positive(
    axes[2],
    df_active["avg_reward_rate"].values,
    title="Active=True | Average reward rate",
    xlabel="average reward rate (events/s)",
    xlim_right=reward_xmax,
)

kde_rug_positive(
    axes[3],
    df_active["avg_lick_rate"].values,
    title="Active=True | Average lick rate",
    xlabel="average lick rate (events/s)",
    xlim_right=lick_xmax,
)

kde_rug_positive(
    axes[4],
    df_passive["avg_speed"].values,
    title="Active=False | Average speed",
    xlabel="average speed",
    xlim_right=speed_xmax,
)

kde_rug_positive(
    axes[5],
    df_passive["avg_pupil"].values,
    title="Active=False | Average pupil",
    xlabel="average pupil",
    xlim_right=pupil_xmax,
)

plt.tight_layout()
save_fig(fig, "behavior_kde_summary.pdf")
plt.show()
# %%
