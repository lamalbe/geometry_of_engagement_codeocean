# %%
"""
Modes alignment: mean-centered covariance PCA/SVD and subspace overlap analyses.

Channels: VISm -> SCMRN, VISpl -> SCMRN.
No train/test splits, z-scoring, correlation matrices, or SVCA reliability metrics.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import svd as scipy_svd
from scipy.optimize import linear_sum_assignment

# %%
# -------------
#  CONFIG
# -------------
TOP_K = 10
MIN_NEURONS_CHANNEL = 30
VAR_EPS = 1e-12

CHANNELS: Tuple[Tuple[str, str], ...] = (
    ("VISm", "SCMRN"),
    ("VISpl", "SCMRN"),
)
TARGET_AREAS: Tuple[str, ...] = ("VISpl", "VISm", "SCMRN")

SORT_KEYS = ["bin_start_times", "time_from_stimulus"]
GROUP_KEYS = ("session_id", "image_name", "active")
TIME_FROM_STIMULUS_MIN = 0.1  # discard first 100 ms after stimulus onset

data_dir = Path(__file__).resolve().parent.parent / "data"
activity_file = data_dir / "df_cortex2mid.pk"

data_filter_columns_dropped = [
    "session_number",
    "experience_level",
    "is_image_novel",
    "is_change",
    "image_set",
]

# %%
# ---------------------------
#  DATA LOADING
# ---------------------------


def load_filtered_activity(activity_path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(activity_file) if activity_path is None else Path(activity_path)
    df_activity = pd.read_pickle(path)
    data_filter = (
        (df_activity.session_number == 1)
        & (df_activity.experience_level == "Familiar")
        & (df_activity.image_name != "omitted")
        & (df_activity.is_image_novel == False)
        & (df_activity.is_change == False)
        & (df_activity.time_from_stimulus >= TIME_FROM_STIMULUS_MIN)
        & (df_activity.N >= MIN_NEURONS_CHANNEL)
        & (df_activity.area.isin(TARGET_AREAS))
    )
    return df_activity[data_filter].drop(data_filter_columns_dropped, axis=1).copy()


def channel_label(source: str, target: str) -> str:
    return f"{source}_to_{target}"


def neuron_count_session(df: pd.DataFrame, session_id: Any, area: str) -> int:
    sub = df.loc[df["session_id"] == session_id, "area"] == area
    if not sub.any():
        return 0
    block = df.loc[df["session_id"] == session_id].loc[df["area"] == area]
    row = block.iloc[0]
    return int(np.asarray(row["spikecounts_area"], dtype=np.float64).ravel().size)


def eligible_sessions_for_channel(
    df: pd.DataFrame,
    source: str,
    target: str,
    min_neurons: int = MIN_NEURONS_CHANNEL,
) -> set:
    ok: set = set()
    for sid in df["session_id"].unique():
        n_src = neuron_count_session(df, sid, source)
        n_tgt = neuron_count_session(df, sid, target)
        if n_src >= min_neurons and n_tgt >= min_neurons:
            ok.add(sid)
    return ok


def print_channel_session_counts(df: pd.DataFrame) -> Dict[str, set]:
    """Print surviving session counts per channel; return eligible session sets."""
    out: Dict[str, set] = {}
    for source, target in CHANNELS:
        ch = channel_label(source, target)
        ok = eligible_sessions_for_channel(df, source, target)
        out[ch] = ok
        print(f"Channel {ch}: {len(ok)} sessions with >={MIN_NEURONS_CHANNEL} neurons in both areas")
    return out


# %%
# ---------------------------
#  ACTIVITY MATRICES
# ---------------------------


def _stack_consistent_vectors(series: pd.Series) -> np.ndarray:
    vecs = [np.asarray(v, dtype=np.float64).ravel() for v in series]
    if not vecs:
        return np.empty((0, 0), dtype=np.float64)
    lengths = pd.Series([len(v) for v in vecs])
    common_len = int(lengths.mode().iloc[0])
    vecs = [v for v in vecs if len(v) == common_len]
    if not vecs:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(vecs)


def build_activity_matrix(group_df: pd.DataFrame, area: str) -> np.ndarray:
    """
    Samples x neurons for one (session_id, image_name, active, area).
    Rows sorted by SORT_KEYS.
    """
    sub = group_df[group_df["area"] == area].copy()
    if sub.empty:
        return np.empty((0, 0), dtype=np.float64)
    for key in SORT_KEYS:
        if key not in sub.columns:
            raise KeyError(f"Missing alignment/sort column: {key}")
    sub = sub.sort_values(SORT_KEYS, kind="mergesort")
    sub = sub.drop_duplicates(subset=SORT_KEYS, keep="first")
    return _stack_consistent_vectors(sub["spikecounts_area"])


def mean_center_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract per-neuron mean over samples. Returns (X_centered, mu)."""
    if X.size == 0:
        return X, np.empty(0, dtype=np.float64)
    mu = X.mean(axis=0)
    return X - mu, mu


def local_covariance_and_pcs(X: np.ndarray, top_k: int = TOP_K) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean-centered local covariance C = X.T @ X / (T-1); top_k eigenpairs.
    Returns C, eigvals (top_k,), P (n_units, top_k) unit columns.
    """
    Xc, _ = mean_center_columns(X)
    T, N = Xc.shape
    if T < 2 or N < 1:
        raise ValueError(f"Need T>=2 and N>=1 for PCA, got T={T}, N={N}")
    C = (Xc.T @ Xc) / (T - 1)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    k = min(top_k, evecs.shape[1])
    P = evecs[:, :k]
    P = _normalize_columns(P)
    return C, evals[:k], P


def cross_covariance_and_svd(
    X_source: np.ndarray,
    X_target: np.ndarray,
    top_k: int = TOP_K,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    C_cross = Xs.T @ Xt / (T-1) with mean-centered areas.
    Returns C_cross, singular values (top_k,), U (n_src, top_k), V (n_tgt, top_k).
    """
    Xs, _ = mean_center_columns(X_source)
    Xt, _ = mean_center_columns(X_target)
    if Xs.shape[0] != Xt.shape[0]:
        raise ValueError(f"Trial count mismatch: source {Xs.shape[0]} vs target {Xt.shape[0]}")
    T = Xs.shape[0]
    if T < 2:
        raise ValueError(f"Need T>=2 for cross-covariance, got T={T}")
    C = (Xs.T @ Xt) / (T - 1)
    U, s, Vt = scipy_svd(C, full_matrices=False)
    k = min(top_k, len(s))
    U_k = _normalize_columns(U[:, :k])
    V_k = _normalize_columns(Vt[:k, :].T)
    return C, s[:k], U_k, V_k


def _normalize_columns(B: np.ndarray, eps: float = VAR_EPS) -> np.ndarray:
    out = np.asarray(B, dtype=np.float64).copy()
    if out.ndim != 2:
        return out
    for j in range(out.shape[1]):
        col = out[:, j]
        n = float(np.linalg.norm(col))
        if n > eps:
            out[:, j] = col / n
    return out


def _abs_cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A, B (N, K_a) and (N, K_b) unit columns -> (K_a, K_b) abs cosine."""
    Ka, Kb = A.shape[1], B.shape[1]
    M = np.empty((Ka, Kb), dtype=np.float64)
    for i in range(Ka):
        for j in range(Kb):
            M[i, j] = float(abs(np.dot(A[:, i], B[:, j])))
    return M


def hungarian_match_abs_cosine(
    A: np.ndarray,
    B: np.ndarray,
    rank: int = TOP_K,
) -> List[Tuple[int, int, float]]:
    """
    One-to-one matching maximizing sum |cos| via Hungarian algorithm.
    Returns list of (active_rank_1based, matched_rank_1based, abs_cosine).
    """
    k_eff = min(rank, A.shape[1], B.shape[1])
    if k_eff < 1:
        return []
    A_use = _normalize_columns(A[:, :k_eff])
    B_use = _normalize_columns(B[:, :k_eff])
    cos_m = _abs_cosine_matrix(A_use, B_use)
    row_ind, col_ind = linear_sum_assignment(-cos_m)
    pairs: List[Tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind):
        pairs.append((int(r) + 1, int(c) + 1, float(cos_m[r, c])))
    pairs.sort(key=lambda x: x[0])
    return pairs


def nuclear_norm_trace_analog(M: np.ndarray) -> float:
    """Nuclear norm = sum of singular values."""
    if M.size == 0:
        return float("nan")
    s = scipy_svd(M, compute_uv=False)
    return float(np.sum(s))


# %%
# ---------------------------
#  FIT CACHE (per group)
# ---------------------------

FitKey = Tuple[Any, Any, Any, str, str]  # session, image, active, channel, area or "_cross"


def _pad_modes(P: np.ndarray, k: int = TOP_K) -> np.ndarray:
    if P.shape[1] >= k:
        return P[:, :k]
    out = np.full((P.shape[0], k), np.nan, dtype=np.float64)
    out[:, : P.shape[1]] = P
    return out


def _pad_vals(v: np.ndarray, k: int = TOP_K) -> np.ndarray:
    out = np.full(k, np.nan, dtype=np.float64)
    out[: len(v)] = v
    return out


class GroupFits:
    """Caches PCA/SVD results for downstream tables."""

    def __init__(self) -> None:
        self.local_pc: Dict[Tuple[Any, Any, Any, str, str], Dict[str, Any]] = {}
        self.cross_svd: Dict[Tuple[Any, Any, Any, str], Dict[str, Any]] = {}

    def store_local(
        self,
        session_id: Any,
        image_name: Any,
        active: Any,
        channel: str,
        area: str,
        C: np.ndarray,
        evals: np.ndarray,
        P: np.ndarray,
    ) -> None:
        self.local_pc[(session_id, image_name, active, channel, area)] = {
            "C": C,
            "evals": evals,
            "P": _pad_modes(P),
        }

    def store_cross(
        self,
        session_id: Any,
        image_name: Any,
        active: Any,
        channel: str,
        source: str,
        C_cross: np.ndarray,
        s: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
    ) -> None:
        self.cross_svd[(session_id, image_name, active, channel)] = {
            "source": source,
            "C_cross": C_cross,
            "s": _pad_vals(s),
            "U": _pad_modes(U),
            "V": _pad_modes(V),
        }


def fit_all_groups(
    df: pd.DataFrame,
    eligible_by_channel: Dict[str, set],
) -> GroupFits:
    fits = GroupFits()
    for source, target in CHANNELS:
        ch = channel_label(source, target)
        allow = eligible_by_channel[ch]
        areas_in_ch = (source, target)
        for (session_id, image_name, active), g in df.groupby(list(GROUP_KEYS), sort=False):
            if session_id not in allow:
                continue
            g_ch = g[g["area"].isin(areas_in_ch)]
            if g_ch.empty:
                continue
            try:
                X_src = build_activity_matrix(g_ch, source)
                X_tgt = build_activity_matrix(g_ch, target)
            except KeyError:
                continue
            if X_src.shape[0] != X_tgt.shape[0]:
                warnings.warn(
                    f"Trial count mismatch session={session_id} image={image_name} "
                    f"active={active} channel={ch}: {source} T={X_src.shape[0]} "
                    f"vs {target} T={X_tgt.shape[0]} — skipping group.",
                    stacklevel=2,
                )
                continue
            if X_src.shape[0] < 2:
                continue
            for area, X in ((source, X_src), (target, X_tgt)):
                C, evals, P = local_covariance_and_pcs(X)
                fits.store_local(session_id, image_name, active, ch, area, C, evals, P)
            C_cr, s, U, V = cross_covariance_and_svd(X_src, X_tgt)
            fits.store_cross(session_id, image_name, active, ch, source, C_cr, s, U, V)
    return fits


# %%
# ---------------------------
#  OUTPUT DATAFRAMES
# ---------------------------


def build_df_basis_vectors(fits: GroupFits) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (sid, img, act, ch, area), d in fits.local_pc.items():
        P = d["P"]
        ev = d["evals"]
        for k in range(TOP_K):
            vec = P[:, k] if k < P.shape[1] else np.full(P.shape[0], np.nan)
            rows.append(
                {
                    "session_id": sid,
                    "image_name": img,
                    "active": act,
                    "channel": ch,
                    "area": area,
                    "basis_type": "PC",
                    "component_rank": k + 1,
                    "eigenvector": vec.copy(),
                    "eigenvalue": float(ev[k]) if k < len(ev) else np.nan,
                    "singular_value": np.nan,
                }
            )
    for (sid, img, act, ch), d in fits.cross_svd.items():
        source = d["source"]
        U, V, s = d["U"], d["V"], d["s"]
        for k in range(TOP_K):
            u = U[:, k] if k < U.shape[1] else np.full(U.shape[0], np.nan)
            v = V[:, k] if k < V.shape[1] else np.full(V.shape[0], np.nan)
            sv = float(s[k]) if k < len(s) and np.isfinite(s[k]) else np.nan
            rows.append(
                {
                    "session_id": sid,
                    "image_name": img,
                    "active": act,
                    "channel": ch,
                    "area": source,
                    "basis_type": "SV_source",
                    "component_rank": k + 1,
                    "eigenvector": u.copy(),
                    "eigenvalue": np.nan,
                    "singular_value": sv,
                }
            )
            rows.append(
                {
                    "session_id": sid,
                    "image_name": img,
                    "active": act,
                    "channel": ch,
                    "area": "SCMRN",
                    "basis_type": "SV_target",
                    "component_rank": k + 1,
                    "eigenvector": v.copy(),
                    "eigenvalue": np.nan,
                    "singular_value": sv,
                }
            )
    return pd.DataFrame(rows)


def _get_pc(fits: GroupFits, sid, img, act, ch, area) -> Optional[np.ndarray]:
    d = fits.local_pc.get((sid, img, act, ch, area))
    return None if d is None else d["P"]


def _get_sv(fits: GroupFits, sid, img, act, ch, area: str) -> Optional[np.ndarray]:
    d = fits.cross_svd.get((sid, img, act, ch))
    if d is None:
        return None
    if area == d["source"]:
        return d["U"]
    if area == "SCMRN":
        return d["V"]
    return None


def _append_hungarian_rows(
    rows: List[Dict[str, Any]],
    *,
    session_id: Any,
    image_name: Any,
    channel: str,
    area: str,
    comparison: str,
    pairs: List[Tuple[int, int, float]],
    active: Optional[Any] = None,
) -> None:
    for active_rank, matched_rank, acos in pairs:
        row = {
            "session_id": session_id,
            "image_name": image_name,
            "channel": channel,
            "area": area,
            "comparison": comparison,
            "active_rank": active_rank,
            "matched_rank": matched_rank,
            "abs_cosine": acos,
        }
        if active is not None:
            row["active"] = active
        rows.append(row)


def build_df_max_cosine_active_passive(fits: GroupFits) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    index_keys: set = set()
    for (sid, img, act, ch, area) in fits.local_pc:
        index_keys.add((sid, img, ch, area))

    for sid, img, ch, area in index_keys:
        P_act = _get_pc(fits, sid, img, True, ch, area)
        P_pas = _get_pc(fits, sid, img, False, ch, area)
        if P_act is not None and P_pas is not None:
            pairs = hungarian_match_abs_cosine(P_act, P_pas)
            _append_hungarian_rows(
                rows,
                session_id=sid,
                image_name=img,
                channel=ch,
                area=area,
                comparison="PC_PC",
                pairs=pairs,
            )
        SV_act = _get_sv(fits, sid, img, True, ch, area)
        SV_pas = _get_sv(fits, sid, img, False, ch, area)
        if SV_act is not None and SV_pas is not None:
            pairs = hungarian_match_abs_cosine(SV_act, SV_pas)
            _append_hungarian_rows(
                rows,
                session_id=sid,
                image_name=img,
                channel=ch,
                area=area,
                comparison="SV_SV",
                pairs=pairs,
            )
        if P_act is not None and SV_pas is not None:
            pairs = hungarian_match_abs_cosine(P_act, SV_pas)
            _append_hungarian_rows(
                rows,
                session_id=sid,
                image_name=img,
                channel=ch,
                area=area,
                comparison="PC_SV",
                pairs=pairs,
            )
        if SV_act is not None and P_pas is not None:
            pairs = hungarian_match_abs_cosine(SV_act, P_pas)
            _append_hungarian_rows(
                rows,
                session_id=sid,
                image_name=img,
                channel=ch,
                area=area,
                comparison="SV_PC",
                pairs=pairs,
            )
    return pd.DataFrame(rows)


def build_df_max_cosine_within_condition(fits: GroupFits) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (sid, img, act, ch, area) in fits.local_pc:
        P = _get_pc(fits, sid, img, act, ch, area)
        SV = _get_sv(fits, sid, img, act, ch, area)
        if P is None or SV is None:
            continue
        pairs = hungarian_match_abs_cosine(P, SV)
        _append_hungarian_rows(
            rows,
            session_id=sid,
            image_name=img,
            channel=ch,
            area=area,
            comparison="PC_SV",
            pairs=pairs,
            active=act,
        )
    return pd.DataFrame(rows)


def build_df_subspace_overlap_pc(fits: GroupFits) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    index_keys: set = set()
    for (sid, img, act, ch, area) in fits.local_pc:
        index_keys.add((sid, img, ch, area))

    for sid, img, ch, area in index_keys:
        d_act = fits.local_pc.get((sid, img, True, ch, area))
        d_pas = fits.local_pc.get((sid, img, False, ch, area))
        if d_act is None or d_pas is None:
            continue
        D_act = d_act["P"]
        D_pas = d_pas["P"]
        C_act = d_act["C"]
        C_pas = d_pas["C"]
        lam_act = d_act["evals"]
        lam_pas = d_pas["evals"]
        k_act = min(TOP_K, D_act.shape[1], len(lam_act))
        k_pas = min(TOP_K, D_pas.shape[1], len(lam_pas))
        if k_act < 1 or k_pas < 1:
            continue
        D_act_k = D_act[:, :k_act]
        D_pas_k = D_pas[:, :k_pas]
        denom_pas = float(np.sum(lam_pas[:k_pas]))
        denom_act = float(np.sum(lam_act[:k_act]))
        if denom_pas > VAR_EPS:
            num = float(np.trace(D_act_k.T @ C_pas @ D_act_k))
            rows.append(
                {
                    "session_id": sid,
                    "image_name": img,
                    "channel": ch,
                    "area": area,
                    "direction": "passive_captured_by_active",
                    "overlap": num / denom_pas,
                }
            )
        if denom_act > VAR_EPS:
            num = float(np.trace(D_pas_k.T @ C_act @ D_pas_k))
            rows.append(
                {
                    "session_id": sid,
                    "image_name": img,
                    "channel": ch,
                    "area": area,
                    "direction": "active_captured_by_passive",
                    "overlap": num / denom_act,
                }
            )
    return pd.DataFrame(rows)


def build_df_subspace_overlap_svd(fits: GroupFits) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    channels_seen: set = set()
    for (sid, img, act, ch) in fits.cross_svd:
        channels_seen.add((sid, img, ch))

    for sid, img, ch in channels_seen:
        d_act = fits.cross_svd.get((sid, img, True, ch))
        d_pas = fits.cross_svd.get((sid, img, False, ch))
        if d_act is None or d_pas is None:
            continue
        U_act, V_act = d_act["U"], d_act["V"]
        U_pas, V_pas = d_pas["U"], d_pas["V"]
        C_act = d_act["C_cross"]
        C_pas = d_pas["C_cross"]
        s_act, s_pas = d_act["s"], d_pas["s"]
        k_act = min(TOP_K, U_act.shape[1], V_act.shape[1], len(s_act))
        k_pas = min(TOP_K, U_pas.shape[1], V_pas.shape[1], len(s_pas))
        if k_act < 1 or k_pas < 1:
            continue
        U_act_k, V_act_k = U_act[:, :k_act], V_act[:, :k_act]
        U_pas_k, V_pas_k = U_pas[:, :k_pas], V_pas[:, :k_pas]
        denom_pas = float(np.nansum(s_pas[:k_pas]))
        denom_act = float(np.nansum(s_act[:k_act]))
        if denom_pas > VAR_EPS:
            M = U_act_k.T @ C_pas @ V_act_k
            rows.append(
                {
                    "session_id": sid,
                    "image_name": img,
                    "channel": ch,
                    "direction": "passive_captured_by_active",
                    "overlap": nuclear_norm_trace_analog(M) / denom_pas,
                }
            )
        if denom_act > VAR_EPS:
            M = U_pas_k.T @ C_act @ V_pas_k
            rows.append(
                {
                    "session_id": sid,
                    "image_name": img,
                    "channel": ch,
                    "direction": "active_captured_by_passive",
                    "overlap": nuclear_norm_trace_analog(M) / denom_act,
                }
            )
    return pd.DataFrame(rows)


def run_modes_alignment_pipeline(
    activity_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_filtered_activity(activity_path)
    print(f"Loaded {len(df)} rows after cohort filter.")
    eligible = print_channel_session_counts(df)
    fits = fit_all_groups(df, eligible)
    df_basis = build_df_basis_vectors(fits)
    df_cos_ap = build_df_max_cosine_active_passive(fits)
    df_cos_within = build_df_max_cosine_within_condition(fits)
    df_ov_pc = build_df_subspace_overlap_pc(fits)
    df_ov_svd = build_df_subspace_overlap_svd(fits)
    return df_basis, df_cos_ap, df_cos_within, df_ov_pc, df_ov_svd


def save_outputs(
    df_basis: pd.DataFrame,
    df_cos_ap: pd.DataFrame,
    df_cos_within: pd.DataFrame,
    df_ov_pc: pd.DataFrame,
    df_ov_svd: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> None:
    out = data_dir if out_dir is None else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "df_basis_vectors.pk": df_basis,
        "df_max_cosine_active_passive.pk": df_cos_ap,
        "df_max_cosine_within_condition.pk": df_cos_within,
        "df_subspace_overlap_pc.pk": df_ov_pc,
        "df_subspace_overlap_svd.pk": df_ov_svd,
    }
    for name, frame in paths.items():
        p = out / name
        frame.to_pickle(p)
        print(f"Saved {name} ({len(frame)} rows) -> {p.resolve()}")


# %%
if __name__ == "__main__":
    basis, cos_ap, cos_within, ov_pc, ov_svd = run_modes_alignment_pipeline()
    print(
        f"\nShapes: basis={basis.shape} cos_ap={cos_ap.shape} "
        f"cos_within={cos_within.shape} ov_pc={ov_pc.shape} ov_svd={ov_svd.shape}"
    )
    save_outputs(basis, cos_ap, cos_within, ov_pc, ov_svd)

# %%
