# %%
"""
Total recorded units by area, pooled across sessions.

One histogram per recording day:
  Familiar = session_number 1 (day1)
  Novel    = session_number 2 (day2)

SCm is the sum of SCiw and SCig (no Allen acronym SCm in the unit table).
Figures go to clean_scripts/results.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ALLEN_DATA_DIR = Path("/data/allen_engagement_data")
DAY_SUBDIRS = {1: "day1", 2: "day2"}

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

RESULTS_DIR = _SCRIPT_DIR.parent / "results"
SAVE_PLOTS = True

SCALE_UNITS = 5000

# Display label, then Allen structure_acronyms that count toward it.
AREA_GROUPS = [
    ("Thal.", ["LGd", "LGv", "LP", "MG", "SGN"], "#E89B8C"),
    ("VIS ctx", ["VISal", "VISam", "VISl", "VISp", "VISpm", "VISrl"], "#3AA8A4"),
    ("Hipp.", ["CA1", "CA3", "DG", "ProS", "SUB"], "#7CB342"),
    ("Midbrain", ["APN", "MB", "MRN", "SCm"], "#D4A0C4"),
]

AREA_SOURCES = {
    "SCm": ["SCiw", "SCig"],
    "MG": ["MG", "MGd", "MGm", "MGv"],
}

COHORTS = [
    ("Familiar", 1),
    ("Novel", 2),
]
GROUP_GAP = 0.7

# %%
# -------------
#  UNIT COUNTS
# -------------


def area_source_acronyms(area_label):
    return AREA_SOURCES.get(area_label, [area_label])


def count_units_for_cohort(df_sessions, session_number):
    """Count units pooled over all sessions of one recording day."""
    day_name = DAY_SUBDIRS[session_number]
    session_ids = df_sessions.loc[
        df_sessions["session_number"] == session_number, "ecephys_session_id"
    ]
    counts = {area: 0 for _, areas, _ in AREA_GROUPS for area in areas}
    n_loaded = 0
    n_missing = 0

    for session_id in session_ids:
        units_file = ALLEN_DATA_DIR / day_name / str(session_id) / "df_units.pk"
        if not units_file.exists():
            n_missing += 1
            print(f"Missing units file: {units_file}")
            continue
        df_units = pd.read_pickle(units_file)
        acronyms = df_units["structure_acronym"].astype(str)
        n_loaded += 1
        for area, sources in ((a, area_source_acronyms(a)) for a in counts):
            counts[area] += int(acronyms.isin(sources).sum())

    print(
        f"session_number={session_number} ({day_name}): "
        f"loaded {n_loaded} sessions, missing {n_missing}"
    )
    return counts


def plot_unit_histogram(counts, title, filename):
    positions = []
    labels = []
    colors = []
    heights = []
    group_spans = []
    x = 0.0

    for group_name, areas, color in AREA_GROUPS:
        start = x
        for area in areas:
            positions.append(x)
            labels.append(area)
            colors.append(color)
            heights.append(counts.get(area, 0))
            x += 1.0
        group_spans.append((group_name, color, start, x - 1.0))
        x += GROUP_GAP

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(positions, heights, width=0.92, color=colors, align="center", linewidth=0)

    ymax = max(heights) if heights else SCALE_UNITS
    ymax = max(ymax, SCALE_UNITS) * 1.18

    x_left = min(positions) - 0.6
    ax.plot(
        [x_left, x_left],
        [0, SCALE_UNITS],
        color="black",
        lw=1.6,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.text(
        x_left - 0.25,
        SCALE_UNITS / 2,
        f"{SCALE_UNITS} units",
        rotation=90,
        va="center",
        ha="right",
        fontsize=10,
    )

    for group_name, color, x0, x1 in group_spans:
        ax.text(
            0.5 * (x0 + x1),
            ymax * 0.92,
            group_name,
            ha="center",
            va="bottom",
            color=color,
            fontsize=12,
            fontweight="medium",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=9)
    ax.set_xlim(min(positions) - 1.4, max(positions) + 0.6)
    ax.set_ylim(0, ymax)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_title(title, loc="left", fontsize=11, pad=8)
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    fig.tight_layout()

    if SAVE_PLOTS:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(RESULTS_DIR / filename, format="pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# %%
if __name__ == "__main__":
    df_sessions = pd.read_pickle(ALLEN_DATA_DIR / "df_sessions.pk")

    for experience_level, session_number in COHORTS:
        n_sessions = int((df_sessions["session_number"] == session_number).sum())
        print(
            f"{experience_level}: {n_sessions} sessions "
            f"(session_number={session_number})"
        )

    for experience_level, session_number in COHORTS:
        counts = count_units_for_cohort(df_sessions, session_number)
        df_counts = pd.DataFrame(
            {"area": list(counts.keys()), "n_units": list(counts.values())}
        )
        print(f"\n{experience_level} unit counts:\n{df_counts.to_string(index=False)}")
        plot_unit_histogram(
            counts,
            title=f"{experience_level} (day {session_number})",
            filename=f"unit_counts_by_area_{experience_level.lower()}_day{session_number}.pdf",
        )
# %%
