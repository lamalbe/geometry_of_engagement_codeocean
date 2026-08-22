# %%
"""Check whether df_units row order matches df_spikes / df_activity neuron order.

For each session under /data/allen_engagement_data/{day1,day2}/<session_id>/,
compare df_units['id'] to df_spikes['unit']. Equality means generate_VIS2MID_activity.py
can index spikecounts with df_units.index; mismatch means area slices are misaligned.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ALLEN_DATA_DIR = Path("/data/allen_engagement_data")
DAY_SUBDIRS = {1: "day1", 2: "day2"}


def session_dirs(df_sessions: pd.DataFrame):
    rows = df_sessions[["ecephys_session_id", "session_number"]].dropna()
    for row in rows.itertuples(index=False):
        session_id = int(row.ecephys_session_id)
        session_number = int(row.session_number)
        day_name = DAY_SUBDIRS[session_number]
        yield session_id, session_number, day_name, ALLEN_DATA_DIR / day_name / str(session_id)


def check_session(session_dir: Path) -> dict:
    units_file = session_dir / "df_units.pk"
    spikes_file = session_dir / "df_spikes.pk"

    if not units_file.exists() or not spikes_file.exists():
        missing = []
        if not units_file.exists():
            missing.append("df_units.pk")
        if not spikes_file.exists():
            missing.append("df_spikes.pk")
        return {
            "status": "missing",
            "aligned": False,
            "n_units": np.nan,
            "n_spikes": np.nan,
            "n_same_position": np.nan,
            "frac_same_position": np.nan,
            "same_id_set": np.nan,
            "note": "missing " + ", ".join(missing),
        }

    df_units = pd.read_pickle(units_file)
    df_spikes = pd.read_pickle(spikes_file)

    if "id" not in df_units.columns:
        return {
            "status": "error",
            "aligned": False,
            "n_units": len(df_units),
            "n_spikes": len(df_spikes),
            "n_same_position": np.nan,
            "frac_same_position": np.nan,
            "same_id_set": np.nan,
            "note": "df_units has no 'id' column",
        }
    if "unit" not in df_spikes.columns:
        return {
            "status": "error",
            "aligned": False,
            "n_units": len(df_units),
            "n_spikes": len(df_spikes),
            "n_same_position": np.nan,
            "frac_same_position": np.nan,
            "same_id_set": np.nan,
            "note": "df_spikes has no 'unit' column",
        }

    unit_ids = df_units["id"].to_numpy()
    spike_ids = df_spikes["unit"].to_numpy()
    n_units = unit_ids.size
    n_spikes = spike_ids.size
    n_compare = min(n_units, n_spikes)
    n_same = int(np.sum(unit_ids[:n_compare] == spike_ids[:n_compare])) if n_compare else 0
    aligned = bool(n_units == n_spikes and n_same == n_units)
    same_id_set = set(unit_ids.tolist()) == set(spike_ids.tolist())

    note = "aligned" if aligned else "order mismatch"
    if n_units != n_spikes:
        note = f"length mismatch (units={n_units}, spikes={n_spikes})"
    elif not same_id_set:
        note = "different id sets"

    return {
        "status": "ok",
        "aligned": aligned,
        "n_units": n_units,
        "n_spikes": n_spikes,
        "n_same_position": n_same,
        "frac_same_position": (n_same / n_compare) if n_compare else np.nan,
        "same_id_set": same_id_set,
        "note": note,
    }


# %%
if __name__ == "__main__":
    sessions_file = ALLEN_DATA_DIR / "df_sessions.pk"
    if not sessions_file.exists():
        raise FileNotFoundError(f"Session table not found: {sessions_file}")

    df_sessions = pd.read_pickle(sessions_file)
    records = []

    for session_id, session_number, day_name, session_dir in session_dirs(df_sessions):
        result = check_session(session_dir)
        records.append(
            {
                "session_id": session_id,
                "session_number": session_number,
                "day": day_name,
                **result,
            }
        )
        flag = "OK" if result["aligned"] else "MISMATCH"
        print(
            f"{flag}  {day_name}/{session_id}  "
            f"n_units={result['n_units']} n_spikes={result['n_spikes']}  "
            f"same_pos={result['n_same_position']}  "
            f"frac={result['frac_same_position']}  "
            f"{result['note']}"
        )

    df = pd.DataFrame(records)
    n_total = len(df)
    n_ok = int(df["aligned"].sum())
    n_mismatch = int(((df["status"] == "ok") & ~df["aligned"]).sum())
    n_missing = int((df["status"] == "missing").sum())
    n_error = int((df["status"] == "error").sum())

    print("\n--- summary ---")
    print(f"sessions listed: {n_total}")
    print(f"aligned:         {n_ok}")
    print(f"mismatch:        {n_mismatch}")
    print(f"missing files:   {n_missing}")
    print(f"errors:          {n_error}")

    mismatches = df[(df["status"] == "ok") & ~df["aligned"]]
    if not mismatches.empty:
        print("\nmismatched sessions:")
        print(
            mismatches[
                [
                    "day",
                    "session_id",
                    "n_units",
                    "n_spikes",
                    "n_same_position",
                    "frac_same_position",
                    "same_id_set",
                    "note",
                ]
            ].to_string(index=False)
        )
# %%
