# %%
"""Change-aligned lick/reward rasters for 20 random day-1 Familiar sessions.

Each page is one session. Rows are hit change trials (epochs). Window is
±2 s around image-change onset. Gray bars are non-change flashes; the
change flash is blue. Black ticks are licks; red diamonds are rewards.

Reads:  /data/allen_engagement_data/day1/<session_id>/{df_presentations,df_licks,df_rewards}.pk
Writes: clean_scripts/results/change_aligned_licks_rewards_day1.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ALLEN_DATA_DIR = Path("/data/allen_engagement_data")
N_SESSIONS = 20
TIME_BEFORE = 1.0
TIME_AFTER = 4.0
REWARD_WINDOW = (0.0, 1.0)  # s after change; keep hit trials only
RNG_SEED = 0
FLASH_GRAY = "#D0D0D0"
CHANGE_BLUE = "#BFD7EA"
LICK_COLOR = "0.15"
REWARD_COLOR = "#C0392B"

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

RESULTS_DIR = _SCRIPT_DIR.parent / "results"
SAVE_PLOTS = True


def session_dir(session_id):
    return ALLEN_DATA_DIR / "day1" / str(session_id)


def load_session_tables(session_id):
    base = session_dir(session_id)
    paths = {
        "presentations": base / "df_presentations.pk",
        "licks": base / "df_licks.pk",
        "rewards": base / "df_rewards.pk",
    }
    if not all(p.exists() for p in paths.values()):
        missing = [p.name for p in paths.values() if not p.exists()]
        return None, missing
    return {key: pd.read_pickle(path) for key, path in paths.items()}, []


def _times(df, column="timestamps"):
    if df is None or df.empty or column not in df.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def hit_change_epochs(tables):
    """Active-block change presentations that received a reward shortly after onset."""
    presentations = tables["presentations"].copy()
    if "is_change" not in presentations.columns or "start_time" not in presentations.columns:
        return []

    active_col = presentations["active"] if "active" in presentations.columns else True
    omitted = (
        presentations["omitted"] if "omitted" in presentations.columns
        else presentations.get("image_name", pd.Series("", index=presentations.index)) == "omitted"
    )
    changes = presentations.loc[
        (presentations["is_change"] == True)
        & (active_col == True)
        & (omitted == False)
    ].sort_values("start_time")

    lick_t = _times(tables["licks"])
    reward_t = _times(tables["rewards"])
    epochs = []
    for _, change in changes.iterrows():
        t0 = float(change["start_time"])
        t1 = float(change["end_time"]) if "end_time" in change and pd.notna(change["end_time"]) else t0 + 0.25
        rewards = reward_t[(reward_t >= t0 + REWARD_WINDOW[0]) & (reward_t <= t0 + REWARD_WINDOW[1])]
        if rewards.size == 0:
            continue

        nearby = presentations.loc[
            (presentations["end_time"] >= t0 - TIME_BEFORE)
            & (presentations["start_time"] <= t0 + TIME_AFTER)
        ]
        flashes = []
        for _, stim in nearby.iterrows():
            flashes.append(
                {
                    "start": float(stim["start_time"]) - t0,
                    "end": float(stim["end_time"]) - t0,
                    "is_change": bool(stim["is_change"]),
                }
            )
        epochs.append(
            {
                "t0": t0,
                "change_start": 0.0,
                "change_end": t1 - t0,
                "licks": lick_t[(lick_t >= t0 - TIME_BEFORE) & (lick_t <= t0 + TIME_AFTER)] - t0,
                "rewards": rewards - t0,
                "flashes": flashes,
            }
        )
    return epochs


def plot_session_raster(ax, epochs, session_id):
    n = len(epochs)
    if n and epochs[0]["flashes"]:
        for flash in epochs[0]["flashes"]:
            color = CHANGE_BLUE if flash["is_change"] else FLASH_GRAY
            ax.axvspan(flash["start"], flash["end"], color=color, lw=0, zorder=0)

    for i, epoch in enumerate(epochs):
        y = n - 1 - i
        if epoch["licks"].size:
            ax.vlines(
                epoch["licks"],
                y - 0.38,
                y + 0.38,
                color=LICK_COLOR,
                lw=0.7,
                zorder=2,
            )
        if epoch["rewards"].size:
            ax.scatter(
                epoch["rewards"],
                np.full(epoch["rewards"].size, y),
                marker="D",
                s=18,
                color=REWARD_COLOR,
                zorder=3,
                linewidths=0,
            )

    ax.set_xlim(-TIME_BEFORE, TIME_AFTER)
    ax.set_ylim(-0.6, max(n - 0.4, 0.6))
    ax.set_xlabel("time from change (s)")
    ax.set_ylabel("epochs")
    ax.set_yticks([])
    ax.set_title(f"session {session_id}  |  {n} hit changes", loc="left", fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.annotate(
        "?",
        xy=(0.0, n - 0.2 if n else 0.5),
        xytext=(0.0, n + 0.25 if n else 1.0),
        ha="center",
        va="bottom",
        color="#E67E22",
        fontsize=11,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "#E67E22", "lw": 1.0},
        annotation_clip=False,
    )

    legend = [
        Line2D([0], [0], color=LICK_COLOR, lw=1.2, label="lick"),
        Line2D(
            [0],
            [0],
            color=REWARD_COLOR,
            marker="D",
            lw=0,
            markersize=6,
            label="reward",
        ),
        Patch(facecolor=FLASH_GRAY, edgecolor="none", label="image"),
        Patch(facecolor=CHANGE_BLUE, edgecolor="none", label="change"),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=8)


def sample_day1_sessions(df_sessions, n=N_SESSIONS, seed=RNG_SEED):
    cohort = df_sessions.query(
        'session_number == 1 and experience_level == "Familiar"'
    ).copy()
    ids = cohort["ecephys_session_id"].astype(int).tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return ids, len(ids)


# %%
if __name__ == "__main__":
    df_sessions = pd.read_pickle(ALLEN_DATA_DIR / "df_sessions.pk")
    shuffled_ids, n_available = sample_day1_sessions(df_sessions)
    print(f"Day-1 Familiar sessions available: {n_available}")

    selected = []
    for session_id in shuffled_ids:
        tables, missing = load_session_tables(session_id)
        if tables is None:
            print(f"Skip {session_id}: missing {missing}")
            continue
        epochs = hit_change_epochs(tables)
        if not epochs:
            print(f"Skip {session_id}: no hit change trials")
            continue
        selected.append((session_id, epochs))
        print(f"Keep {session_id}: {len(epochs)} hit changes")
        if len(selected) >= N_SESSIONS:
            break

    if len(selected) < N_SESSIONS:
        print(f"Only found {len(selected)} usable sessions (wanted {N_SESSIONS})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "change_aligned_licks_rewards_day1.pdf"

    if SAVE_PLOTS:
        with PdfPages(out_path) as pdf:
            for session_id, epochs in selected:
                height = max(3.8, min(11.0, 0.14 * len(epochs) + 1.6))
                fig, ax = plt.subplots(figsize=(6.8, height))
                plot_session_raster(ax, epochs, session_id)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        print(f"Saved {len(selected)} pages -> {out_path}")
    else:
        session_id, epochs = selected[0]
        fig, ax = plt.subplots(figsize=(6.8, max(3.8, 0.14 * len(epochs) + 1.6)))
        plot_session_raster(ax, epochs, session_id)
        fig.tight_layout()
        plt.show()
# %%
