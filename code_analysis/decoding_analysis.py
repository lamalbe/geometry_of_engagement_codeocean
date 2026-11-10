# %%
"""
Lick decoding relative to VIS–SCMRN communication subspaces.

Reads data/df_decoding.pk (from generate_decoding.py).
Cohort: day-1 Familiar, image sets G and H, viewing bins with
time_from_stimulus >= 0.1 s, non-change, non-novel, non-omitted, N >= 30.
Labels are licks on non-change images, not change reports.

Channels: VISm → SCMRN (main) and VISpl → SCMRN (supplement).
Per session, decode *active* lick vs no-lick from:

  1. Full space.
  2. Active communication k-D (U_k in cortex, V_k in SCMRN).
  3. Orthogonal complement of that subspace.
  4. Random k-D (dimensionality control).
  5. Local top-k PCs (active).
  6. Communication SVD fit on active no-lick trials only (supplement).
  7. Active activity projected onto passive communication / passive PCs.

Also records alpha_k = ||B_k^T w||^2 for the full-space decoder w vs
active / passive / random communication subspaces (supplement: orientation
of w, not a test that those modes carry lick information).

Main figures (VISm → SCMRN): readout boxes and communication vs PCA scatter.
Supplementary: passive/no-lick readouts, other scatters, alpha_k, and the
same set for VISpl → SCMRN (_vispl_scmrn).

Figures: results/decoding_analysis/decoding_*.pdf
Tables:  data/df_decoding_{repeat,session,alpha_session}.pk
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr, svd as scipy_svd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# %%
# -------------
#  CONFIG
# -------------

TOP_K = 5
VAR_EPS = 1e-12
ALIGN_KEYS = ["bin_start_times", "time_from_stimulus", "image_name"]

CHANNELS = [
    ("VISm", "SCMRN"),
    ("VISpl", "SCMRN"),
]
ALL_AREAS = sorted({a for pair in CHANNELS for a in pair})

LOGISTIC_C = 0.1
N_UNDERSAMPLE_REPEATS = 10
N_RANDOM_DECODE = 5
N_RANDOM_ALPHA = 50
N_CV_FOLDS = 5
MIN_LICK_TRIALS = 30
MIN_NOLICK_SVD = 50
MIN_UNITS = 30
TIME_FROM_STIMULUS_MIN = 0.1
RANDOM_SEED = 0
SAVE_PLOTS = True
SHOW_PLOTS = False
FORCE_REBUILD = False

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

import config

DATA_DIR = config.DATA_DIR
RESULTS_DIR = config.results_dir("decoding_analysis")
activity_file = DATA_DIR / "df_decoding.pk"
REPEAT_FILE = DATA_DIR / "df_decoding_repeat.pk"
SESSION_FILE = DATA_DIR / "df_decoding_session.pk"
ALPHA_SESSION_FILE = DATA_DIR / "df_decoding_alpha_session.pk"

AREA_FILL = {
    "VISpl": "#0077B6",
    "VISm": "#34A0A4",
    "SCMRN": "#C36F09",
}
AREA_PALETTE = {
    "VISpl": {True: "#0077B6", False: "#90E0EF"},
    "VISm": {True: "#34A0A4", False: "#48CAE4"},
    "SCMRN": {True: "#C36F09", False: "#EEBA0B"},
}
RANDOM_FILL = "#888888"

MAIN_READOUTS = [
    "full",
    "communication",
    "pca",
    "random",
    "orthogonal",
]
PASSIVE_READOUTS = [
    "communication",
    "communication_passive",
    "pca",
    "pca_passive",
    "communication_nolick",
]
READOUT_ORDER = [
    "full",
    "communication",
    "pca",
    "random",
    "orthogonal",
    "communication_nolick",
    "communication_passive",
    "pca_passive",
]
READOUT_LABELS = {
    "full": "full",
    "communication": "comm. k-D",
    "pca": "PCA k-D",
    "random": "random k-D",
    "orthogonal": "orth. complement",
    "communication_nolick": "comm. (no-lick SVD)",
    "communication_passive": "comm. (passive)",
    "pca_passive": "PCA (passive)",
}
READOUT_COLORS = {
    "full": "#333333",
    "communication": "#C36F09",
    "pca": "#34A0A4",
    "random": "#888888",
    "orthogonal": "#6C757D",
    "communication_nolick": "#7B2CBF",
    "communication_passive": "#D4A373",
    "pca_passive": "#90E0EF",
}


def _style_filled_boxes(bp, box_colors):
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)
        patch.set_alpha(0.7)
    for i in range(len(box_colors)):
        bp["whiskers"][i * 2].set_color("black")
        bp["whiskers"][i * 2 + 1].set_color("black")
        bp["whiskers"][i * 2].set_linewidth(1.5)
        bp["whiskers"][i * 2 + 1].set_linewidth(1.5)
        bp["caps"][i * 2].set_color("black")
        bp["caps"][i * 2 + 1].set_color("black")
        bp["caps"][i * 2].set_linewidth(1.5)
        bp["caps"][i * 2 + 1].set_linewidth(1.5)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    for flier in bp.get("fliers", []):
        flier.set_markerfacecolor("white")
        flier.set_markeredgecolor("black")
        flier.set_markersize(5)
        flier.set_alpha(0.8)
        flier.set_marker("o")


def _save_fig(fig, filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    if SAVE_PLOTS:
        fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"  Saved figure: {path}")
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()


# %%
# ---------------------------
#  LINEAR ALGEBRA / DECODE
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


def mean_center_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X.size == 0:
        return X, np.empty(0, dtype=np.float64)
    mu = X.mean(axis=0)
    return X - mu, mu


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


def licks_to_bool(licks_series: pd.Series) -> pd.Series:
    return licks_series.map(
        lambda x: False if isinstance(x, dict) and len(x) == 0 else True
    )


def build_area_matrix(df: pd.DataFrame, area: str) -> Tuple[np.ndarray, pd.DataFrame]:
    sub = df[df["area"] == area].copy()
    if sub.empty:
        return np.empty((0, 0), dtype=np.float64), sub
    for key in ALIGN_KEYS:
        if key not in sub.columns:
            raise KeyError(f"Missing alignment column: {key}")
    sub = sub.sort_values(ALIGN_KEYS, kind="mergesort")
    sub = sub.drop_duplicates(subset=ALIGN_KEYS, keep="first")
    X = _stack_consistent_vectors(sub["spikecounts_area"])
    return X, sub


def build_matched_area_pair(
    df: pd.DataFrame, source_area: str, target_area: str
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    keep_cols = ALIGN_KEYS + ["spikecounts_area"]
    extra = ["licks_bool"] if "licks_bool" in df.columns else []
    src = df[df["area"] == source_area][keep_cols + extra].copy()
    tgt = df[df["area"] == target_area][keep_cols].copy()
    merged = src.merge(tgt, on=ALIGN_KEYS, suffixes=("_src", "_tgt"), how="inner")
    if merged.empty:
        return np.empty((0, 0)), np.empty((0, 0)), merged
    X_src = _stack_consistent_vectors(merged["spikecounts_area_src"])
    X_tgt = _stack_consistent_vectors(merged["spikecounts_area_tgt"])
    n = min(X_src.shape[0], X_tgt.shape[0])
    return X_src[:n], X_tgt[:n], merged.iloc[:n]


def cross_covariance_and_svd(
    X_source: np.ndarray,
    X_target: np.ndarray,
    top_k: int = TOP_K,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, _ = mean_center_columns(X_source)
    Xt, _ = mean_center_columns(X_target)
    if Xs.shape[0] != Xt.shape[0]:
        raise ValueError(f"Trial count mismatch: {Xs.shape[0]} vs {Xt.shape[0]}")
    T = Xs.shape[0]
    if T < 2:
        raise ValueError(f"Need T>=2 for cross-covariance, got T={T}")
    C = (Xs.T @ Xt) / (T - 1)
    U, s, Vt = scipy_svd(C, full_matrices=False)
    k = min(top_k, len(s))
    U_k = _normalize_columns(U[:, :k])
    V_k = _normalize_columns(Vt[:k, :].T)
    return s[:k], U_k, V_k


def local_pca_basis(X: np.ndarray, top_k: int = TOP_K) -> np.ndarray:
    Xc, _ = mean_center_columns(X)
    if Xc.shape[0] < 2 or Xc.shape[1] < 2:
        return np.empty((X.shape[1] if X.ndim == 2 else 0, 0))
    _, _, Vt = scipy_svd(Xc, full_matrices=False)
    k = min(top_k, Vt.shape[0], Xc.shape[1])
    return _normalize_columns(Vt[:k, :].T)


def random_orthonormal_subspace(
    n_neurons: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    if k > n_neurons:
        raise ValueError(f"k={k} exceeds n_neurons={n_neurons}")
    A = rng.standard_normal((n_neurons, k))
    Q, _ = qr(A, mode="economic")
    return _normalize_columns(Q[:, :k])


def project_onto(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    return X @ B


def residual_from(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Activity in the orthogonal complement of B (communication U_k/V_k, not PCs)."""
    return X - X @ B @ B.T


def make_logistic(C: float = LOGISTIC_C) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=2000,
    )


def fit_l2_logistic(
    X: np.ndarray, y: np.ndarray, C: float = LOGISTIC_C
) -> Tuple[np.ndarray, float]:
    y = np.asarray(y, dtype=np.int64).ravel()
    if X.shape[0] < 2 or X.shape[1] < 1 or len(np.unique(y)) < 2:
        raise ValueError("Need n>=2, p>=1, and both classes.")
    clf = make_logistic(C=C)
    clf.fit(X, y)
    w = np.asarray(clf.coef_, dtype=np.float64).ravel()
    b = float(np.asarray(clf.intercept_).ravel()[0])
    return w, b


def unit_vector(w: np.ndarray, eps: float = VAR_EPS) -> np.ndarray:
    n = float(np.linalg.norm(w))
    if n < eps:
        return w
    return w / n


def alpha_k(w: np.ndarray, B_k: np.ndarray) -> float:
    w = unit_vector(w)
    if B_k.size == 0:
        return np.nan
    return float(np.sum((B_k.T @ w) ** 2))


def cv_balanced_accuracy_auc(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_CV_FOLDS,
    C: float = LOGISTIC_C,
    random_state: int = 0,
) -> Tuple[float, float]:
    y = np.asarray(y, dtype=np.int64).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    n_folds_eff = min(n_folds, n_pos, n_neg)
    if n_folds_eff < 2:
        raise ValueError(
            f"Need >=2 samples per class for CV, got pos={n_pos}, neg={n_neg}"
        )
    cv = StratifiedKFold(n_splits=n_folds_eff, shuffle=True, random_state=random_state)
    clf = make_logistic(C=C)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    return float(balanced_accuracy_score(y, pred)), float(roc_auc_score(y, proba))


def undersample_balanced_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(~y.astype(bool))
    n_pos = len(pos_idx)
    if n_pos == 0 or len(neg_idx) == 0:
        raise ValueError("Need both classes for undersampling.")
    if len(neg_idx) >= n_pos:
        neg_pick = rng.choice(neg_idx, size=n_pos, replace=False)
    else:
        neg_pick = neg_idx
    return np.sort(np.concatenate([pos_idx, neg_pick]))


def _safe_cv(X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[float, float]:
    if X.size == 0 or X.shape[0] < 4 or X.shape[1] < 1:
        return np.nan, np.nan
    try:
        return cv_balanced_accuracy_auc(X, y, random_state=random_state)
    except ValueError:
        return np.nan, np.nan


# %%
# ---------------------------
#  SESSION LOOP
# ---------------------------


def decode_readouts_for_area(
    X: np.ndarray,
    y: np.ndarray,
    B_comm: np.ndarray,
    B_pca: np.ndarray,
    B_nolick: Optional[np.ndarray],
    B_comm_pas: Optional[np.ndarray],
    B_pca_pas: Optional[np.ndarray],
    rng: np.random.Generator,
    session_id: str,
    area: str,
    channel: str,
) -> List[Dict[str, Any]]:
    n_lick = int(np.sum(y))
    n_nolick = int(np.sum(~y.astype(bool)))
    if n_lick < MIN_LICK_TRIALS:
        print(f"  Skipping {area}: fewer than {MIN_LICK_TRIALS} lick trials.")
        return []
    if B_comm.shape[0] != X.shape[1]:
        print(
            f"  Skipping {area}: neuron mismatch "
            f"(decoder {X.shape[1]} vs comm {B_comm.shape[0]})."
        )
        return []

    Xc, _ = mean_center_columns(X)
    rows: List[Dict[str, Any]] = []
    k_comm = B_comm.shape[1]
    k_pca = B_pca.shape[1] if B_pca.size else 0

    def _row(repeat, readout, acc, auc, n_bal, alphas=None):
        out = {
            "session_id": session_id,
            "channel": channel,
            "area": area,
            "repeat": repeat,
            "readout": readout,
            "k": TOP_K,
            "n_neurons": int(X.shape[1]),
            "n_lick": n_lick,
            "n_nolick": n_nolick,
            "n_trials_balanced": int(n_bal),
            "balanced_accuracy": acc,
            "roc_auc": auc,
            "alpha_active": np.nan,
            "alpha_passive": np.nan,
            "alpha_random_mean": np.nan,
        }
        if alphas is not None:
            out["alpha_active"] = alphas[0]
            out["alpha_passive"] = alphas[1]
            out["alpha_random_mean"] = alphas[2]
        return out

    for repeat in range(N_UNDERSAMPLE_REPEATS):
        try:
            idx = undersample_balanced_indices(y, rng)
        except ValueError as exc:
            print(f"    {area} repeat {repeat}: skipped ({exc})")
            continue
        X_bal = Xc[idx]
        y_bal = y[idx].astype(np.int64)
        rs = RANDOM_SEED + repeat

        readouts: Dict[str, np.ndarray] = {
            "full": X_bal,
            "communication": project_onto(X_bal, B_comm),
            "orthogonal": residual_from(X_bal, B_comm),
        }
        if k_pca > 0 and B_pca.shape[0] == X_bal.shape[1]:
            readouts["pca"] = project_onto(X_bal, B_pca)
        if B_nolick is not None and B_nolick.shape[0] == X_bal.shape[1]:
            readouts["communication_nolick"] = project_onto(X_bal, B_nolick)
        if B_comm_pas is not None and B_comm_pas.shape[0] == X_bal.shape[1]:
            readouts["communication_passive"] = project_onto(X_bal, B_comm_pas)
        if B_pca_pas is not None and B_pca_pas.size and B_pca_pas.shape[0] == X_bal.shape[1]:
            readouts["pca_passive"] = project_onto(X_bal, B_pca_pas)

        rand_acc = []
        rand_auc = []
        for _ in range(N_RANDOM_DECODE):
            try:
                B_rand = random_orthonormal_subspace(X_bal.shape[1], k_comm, rng)
            except ValueError:
                continue
            acc, auc = _safe_cv(project_onto(X_bal, B_rand), y_bal, rs)
            if np.isfinite(acc):
                rand_acc.append(acc)
                rand_auc.append(auc)

        alphas = None
        try:
            w, _ = fit_l2_logistic(X_bal, y_bal, C=LOGISTIC_C)
            w_hat = unit_vector(w)
            a_act = alpha_k(w_hat, B_comm)
            a_pas = (
                alpha_k(w_hat, B_comm_pas)
                if B_comm_pas is not None and B_comm_pas.shape[0] == X_bal.shape[1]
                else np.nan
            )
            rand_alphas = [
                alpha_k(
                    w_hat, random_orthonormal_subspace(X_bal.shape[1], k_comm, rng)
                )
                for _ in range(N_RANDOM_ALPHA)
            ]
            alphas = (a_act, a_pas, float(np.mean(rand_alphas)))
        except ValueError:
            alphas = None

        for readout, X_use in readouts.items():
            acc, auc = _safe_cv(X_use, y_bal, rs)
            rows.append(
                _row(
                    repeat, readout, acc, auc, len(idx),
                    alphas=alphas if readout == "full" else None,
                )
            )
        if rand_acc:
            rows.append(
                _row(
                    repeat, "random",
                    float(np.mean(rand_acc)), float(np.mean(rand_auc)), len(idx),
                )
            )
    return rows


def _fit_comm_svd(df_block, source, target):
    X_src, X_tgt, merged = build_matched_area_pair(df_block, source, target)
    if X_src.shape[0] < 2:
        return None, None, None, None, merged
    try:
        _, U_k, V_k = cross_covariance_and_svd(X_src, X_tgt, top_k=TOP_K)
    except ValueError:
        return None, None, None, None, merged
    return U_k, V_k, X_src, X_tgt, merged


def run_all_sessions(df_filtered: pd.DataFrame) -> pd.DataFrame:
    session_ids = sorted(df_filtered["session_id"].unique())
    rng = np.random.default_rng(RANDOM_SEED)
    rows: List[Dict[str, Any]] = []

    print(
        f"\nDecoding analysis: k={TOP_K}, "
        f"{N_UNDERSAMPLE_REPEATS} undersample repeats, "
        f"{N_RANDOM_DECODE} random decode subspaces, "
        f"{N_RANDOM_ALPHA} random alpha subspaces, {N_CV_FOLDS}-fold CV."
    )
    print("  Channels:", ", ".join(f"{s}→{t}" for s, t in CHANNELS))

    for i_session, session_id in enumerate(session_ids):
        print(f"\nSession {i_session + 1}/{len(session_ids)}: {session_id}")
        df_sess = df_filtered[df_filtered["session_id"] == session_id]
        df_active = df_sess[df_sess["active"] == True]
        df_passive = df_sess[df_sess["active"] == False]

        for source, target in CHANNELS:
            channel = f"{source}_to_{target}"
            decode_areas = (target, source)
            print(f"  Channel {channel}")

            U_comm, V_comm, X_src, X_tgt, merged = _fit_comm_svd(
                df_active, source, target
            )
            if U_comm is None:
                print("    Skipping: insufficient active aligned trials / SVD.")
                continue
            print(
                f"    active aligned={X_src.shape[0]}, "
                f"{source} N={U_comm.shape[0]}, {target} N={V_comm.shape[0]}"
            )

            B_nolick_by_area: Dict[str, Optional[np.ndarray]] = {
                source: None, target: None,
            }
            if merged is not None and "licks_bool" in merged.columns:
                y_match = merged["licks_bool"].to_numpy(dtype=bool)[: X_src.shape[0]]
                n_nl = int(np.sum(~y_match))
                if n_nl >= MIN_NOLICK_SVD:
                    try:
                        _, U_nl, V_nl = cross_covariance_and_svd(
                            X_src[~y_match], X_tgt[~y_match], top_k=TOP_K
                        )
                        B_nolick_by_area[source] = U_nl
                        B_nolick_by_area[target] = V_nl
                        print(f"    no-lick SVD on {n_nl} aligned trials.")
                    except ValueError as exc:
                        print(f"    no-lick SVD skipped ({exc}).")
                else:
                    print(
                        f"    no-lick SVD skipped "
                        f"(n_nolick={n_nl} < {MIN_NOLICK_SVD})."
                    )

            U_pas, V_pas, X_pas_src, _, _ = _fit_comm_svd(df_passive, source, target)
            B_comm_pas_by_area: Dict[str, Optional[np.ndarray]] = {
                source: None, target: None,
            }
            if U_pas is not None:
                B_comm_pas_by_area[source] = U_pas
                B_comm_pas_by_area[target] = V_pas
                print(f"    passive aligned={X_pas_src.shape[0]}")
            else:
                print("    passive communication SVD skipped.")

            B_comm_by_area = {source: U_comm, target: V_comm}

            for area in decode_areas:
                X_dec, meta_dec = build_area_matrix(df_active, area)
                if X_dec.size == 0:
                    print(f"    Skipping {area}: no active matrix.")
                    continue
                y_all = meta_dec["licks_bool"].to_numpy(dtype=bool)
                if len(y_all) != X_dec.shape[0]:
                    n = min(len(y_all), X_dec.shape[0])
                    X_dec, y_all = X_dec[:n], y_all[:n]
                B_pca = local_pca_basis(X_dec, top_k=TOP_K)
                X_pas_area, _ = build_area_matrix(df_passive, area)
                B_pca_pas = (
                    local_pca_basis(X_pas_area, top_k=TOP_K)
                    if X_pas_area.size
                    else None
                )
                area_rows = decode_readouts_for_area(
                    X=X_dec,
                    y=y_all,
                    B_comm=B_comm_by_area[area],
                    B_pca=B_pca,
                    B_nolick=B_nolick_by_area[area],
                    B_comm_pas=B_comm_pas_by_area[area],
                    B_pca_pas=B_pca_pas,
                    rng=rng,
                    session_id=str(session_id),
                    area=area,
                    channel=channel,
                )
                rows.extend(area_rows)

    return pd.DataFrame(rows)


# %%
# ---------------------------
#  SUMMARIES AND PLOTS
# ---------------------------


def session_means(df_repeat: pd.DataFrame) -> pd.DataFrame:
    keys = ["session_id", "channel", "area", "readout"]
    keys = [k for k in keys if k in df_repeat.columns]
    return df_repeat.groupby(keys, as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"),
        roc_auc=("roc_auc", "mean"),
        n_lick=("n_lick", "first"),
        n_neurons=("n_neurons", "first"),
        n_nolick=("n_nolick", "first"),
    )


def session_alpha(df_repeat: pd.DataFrame) -> pd.DataFrame:
    full = df_repeat[df_repeat["readout"] == "full"].copy()
    if full.empty or "alpha_active" not in full.columns:
        return pd.DataFrame()
    return full.groupby(["session_id", "channel", "area"], as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"),
        roc_auc=("roc_auc", "mean"),
        alpha_active=("alpha_active", "mean"),
        alpha_passive=("alpha_passive", "mean"),
        alpha_random_mean=("alpha_random_mean", "mean"),
        n_lick=("n_lick", "first"),
        n_neurons=("n_neurons", "first"),
    )


def pivot_readout(
    df_sess: pd.DataFrame, metric: str = "balanced_accuracy"
) -> pd.DataFrame:
    idx = ["session_id", "area", "n_lick", "n_neurons"]
    if "channel" in df_sess.columns:
        idx = ["session_id", "channel", "area", "n_lick", "n_neurons"]
    wide = df_sess.pivot_table(
        index=idx,
        columns="readout",
        values=metric,
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    return wide


def _channel_tag(source: str, target: str) -> str:
    return f"{source}_to_{target}"


def _file_tag(source: str, target: str) -> str:
    if source == "VISm" and target == "SCMRN":
        return ""
    return f"_{source.lower()}_{target.lower()}"


def _print_diagnostics(
    df_sess: pd.DataFrame, df_repeat: pd.DataFrame, df_alpha: pd.DataFrame
) -> None:
    channels = (
        sorted(df_sess["channel"].unique())
        if "channel" in df_sess.columns
        else [None]
    )
    for channel in channels:
        dfc = df_sess if channel is None else df_sess[df_sess["channel"] == channel]
        dfr = (
            df_repeat if channel is None else df_repeat[df_repeat["channel"] == channel]
        )
        dfa = (
            df_alpha
            if channel is None or df_alpha.empty
            else df_alpha[df_alpha["channel"] == channel]
        )
        areas = [a for a in ("SCMRN", "VISm", "VISpl") if a in set(dfc["area"])]
        hdr = channel if channel else "all"
        print(f"\n=== Session-mean balanced accuracy ({hdr}) ===")
        for area in areas:
            sub = dfc[dfc["area"] == area]
            print(f"\n{area}:")
            for readout in READOUT_ORDER:
                vals = sub.loc[sub["readout"] == readout, "balanced_accuracy"].dropna()
                if vals.empty:
                    continue
                print(
                    f"  {READOUT_LABELS[readout]:22s}  "
                    f"n={len(vals):2d}  mean={vals.mean():.3f}  "
                    f"median={vals.median():.3f}"
                )

        wide = pivot_readout(dfc)
        print(f"\n=== Spearman ({hdr}) ===")
        pairs = [
            ("full", "communication"),
            ("communication", "pca"),
            ("communication", "random"),
            ("communication", "communication_passive"),
            ("pca", "pca_passive"),
            ("full", "orthogonal"),
        ]
        for area in areas:
            w = wide[wide["area"] == area]
            print(f"\n{area}:")
            for a, b in pairs:
                if a not in w.columns or b not in w.columns:
                    continue
                mask = w[a].notna() & w[b].notna()
                if mask.sum() < 5:
                    continue
                rho, p = spearmanr(w.loc[mask, a], w.loc[mask, b])
                print(f"  {a} vs {b}: rho={rho:.3f}, p={p:.3g}, n={int(mask.sum())}")

        if not dfa.empty:
            print(f"\n=== Decoder alignment alpha_k ({hdr}) ===")
            for area in areas:
                sub = dfa[dfa["area"] == area]
                if sub.empty:
                    continue
                n_ap = int((sub["alpha_active"] > sub["alpha_passive"]).sum())
                print(f"\n{area}:")
                for col, name in [
                    ("alpha_active", "active"),
                    ("alpha_passive", "passive"),
                    ("alpha_random_mean", "random"),
                ]:
                    vals = sub[col].dropna()
                    if vals.empty:
                        continue
                    print(
                        f"  {name:8s}  n={len(vals):2d}  "
                        f"mean={vals.mean():.3f}  median={vals.median():.3f}"
                    )
                print(f"  active>passive in {n_ap}/{len(sub)} sessions")
                mask = sub["balanced_accuracy"].notna() & sub["alpha_active"].notna()
                if mask.sum() >= 5:
                    rho, p = spearmanr(
                        sub.loc[mask, "balanced_accuracy"],
                        sub.loc[mask, "alpha_active"],
                    )
                    print(
                        f"  Spearman(full acc, alpha_active): "
                        f"rho={rho:.3f}, p={p:.3g}"
                    )

        print(f"\n=== Split-repeat reliability of full-space accuracy ({hdr}) ===")
        full = dfr[dfr["readout"] == "full"]
        mid = N_UNDERSAMPLE_REPEATS // 2
        gkeys = ["session_id", "area"]
        g1 = (
            full[full["repeat"] < mid]
            .groupby(gkeys, as_index=False)["balanced_accuracy"]
            .mean()
            .rename(columns={"balanced_accuracy": "acc_a"})
        )
        g2 = (
            full[full["repeat"] >= mid]
            .groupby(gkeys, as_index=False)["balanced_accuracy"]
            .mean()
            .rename(columns={"balanced_accuracy": "acc_b"})
        )
        rel = g1.merge(g2, on=gkeys)
        for area in areas:
            r = rel[rel["area"] == area]
            if len(r) < 5:
                continue
            rho, p = spearmanr(r["acc_a"], r["acc_b"])
            print(f"  {area}: rho={rho:.3f}, p={p:.3g}, n={len(r)}")


def plot_readout_boxes(
    df_sess: pd.DataFrame,
    areas: Tuple[str, ...],
    filename: str,
    title: str,
    readout_order: Optional[List[str]] = None,
) -> None:
    order = readout_order if readout_order is not None else MAIN_READOUTS
    present = [r for r in order if r in set(df_sess["readout"])]
    fig, axes = plt.subplots(1, len(areas), figsize=(5.5 * len(areas), 4.4), sharey=True)
    if len(areas) == 1:
        axes = [axes]
    for ax, area in zip(axes, areas):
        sub = df_sess[df_sess["area"] == area]
        box_data, colors, labels = [], [], []
        for readout in present:
            vals = sub.loc[sub["readout"] == readout, "balanced_accuracy"].dropna()
            if vals.empty:
                continue
            box_data.append(vals.to_numpy())
            colors.append(READOUT_COLORS[readout])
            labels.append(READOUT_LABELS[readout])
        if not box_data:
            continue
        bp = ax.boxplot(
            box_data,
            positions=range(len(box_data)),
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            zorder=2,
        )
        _style_filled_boxes(bp, colors)
        ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.5, zorder=0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(area)
        ax.set_ylabel("Balanced accuracy" if area == areas[0] else "")
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.2)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save_fig(fig, filename)


def plot_shared_scatter(
    df_sess: pd.DataFrame,
    areas: Tuple[str, ...],
    x_readout: str,
    y_readout: str,
    filename: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    wide = pivot_readout(df_sess)
    if x_readout not in wide.columns or y_readout not in wide.columns:
        print(f"  Skipping {filename}: missing {x_readout} or {y_readout}.")
        return
    fig, axes = plt.subplots(
        1, len(areas), figsize=(4.2 * len(areas), 4.2), sharex=False, sharey=False
    )
    if len(areas) == 1:
        axes = [axes]
    shared_lims = []
    for ax, area in zip(axes, areas):
        d = wide[wide["area"] == area].dropna(subset=[x_readout, y_readout])
        ax.scatter(
            d[x_readout],
            d[y_readout],
            s=36,
            c=AREA_FILL.get(area, "#333333"),
            alpha=0.85,
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
        )
        vals = pd.concat([d[x_readout], d[y_readout]], ignore_index=True)
        lo = 0.4
        hi = float(max(vals.max() if not vals.empty else 0.6, 0.55))
        pad = 0.05 * (hi - lo)
        hi = min(1.0, hi + pad)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7, zorder=0)
        ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.4, zorder=0)
        ax.axvline(0.5, color="k", ls=":", lw=0.8, alpha=0.4, zorder=0)
        ax.set_title(area)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if area == areas[0] else "")
        ax.set_box_aspect(1)
        shared_lims.append((lo, hi))
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    for ax, (lo, hi) in zip(axes, shared_lims):
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ticks = ax.get_xticks()
        ticks = ticks[(ticks >= lo) & (ticks <= hi)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    _save_fig(fig, filename)


def plot_alpha_boxes(
    df_alpha: pd.DataFrame,
    areas: Tuple[str, ...],
    filename: str,
    title: str,
) -> None:
    if df_alpha.empty:
        print(f"  Skipping {filename}: no alpha rows.")
        return
    subspace_cols = ["alpha_active", "alpha_passive", "alpha_random_mean"]
    subspace_labels = ["active", "passive", f"random k={TOP_K}"]
    fig, axes = plt.subplots(1, len(areas), figsize=(5.0 * len(areas), 4.2), sharey=True)
    if len(areas) == 1:
        axes = [axes]
    for ax, area in zip(axes, areas):
        pal = AREA_PALETTE.get(area, {True: "#333333", False: "#AAAAAA"})
        box_colors = [pal[True], pal[False], RANDOM_FILL]
        box_data = [
            df_alpha.loc[df_alpha["area"] == area, col].dropna().to_numpy()
            for col in subspace_cols
        ]
        if all(len(v) == 0 for v in box_data):
            continue
        bp = ax.boxplot(
            box_data,
            positions=range(len(subspace_cols)),
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            zorder=2,
        )
        _style_filled_boxes(bp, box_colors)
        side = "V_k" if area == "SCMRN" else "U_k"
        ax.set_xticks(range(len(subspace_labels)))
        ax.set_xticklabels(subspace_labels)
        ax.set_title(f"{area} decoder vs {side}")
        ax.set_ylabel(r"$\alpha_k = \|B_k^\top w\|^2$" if area == areas[0] else "")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.2)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save_fig(fig, filename)


def plot_alpha_paired(
    df_alpha: pd.DataFrame,
    areas: Tuple[str, ...],
    filename: str,
    title: str,
) -> None:
    if df_alpha.empty:
        print(f"  Skipping {filename}: no alpha rows.")
        return
    fig, axes = plt.subplots(
        1, len(areas), figsize=(4.2 * len(areas), 4.2), sharex=False, sharey=False
    )
    if len(areas) == 1:
        axes = [axes]
    shared_lims = []
    for ax, area in zip(axes, areas):
        sub = df_alpha[df_alpha["area"] == area].dropna(
            subset=["alpha_active", "alpha_passive"]
        )
        ax.scatter(
            sub["alpha_passive"],
            sub["alpha_active"],
            s=40,
            c=AREA_FILL.get(area, "#333333"),
            alpha=0.85,
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
        )
        vals = pd.concat(
            [sub["alpha_active"], sub["alpha_passive"]], ignore_index=True
        )
        lo = 0.0
        hi = float(max(vals.max() if not vals.empty else 0.05, 0.05))
        pad = 0.05 * (hi - lo) if hi > lo else 0.01
        hi = min(1.0, hi + pad)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7, zorder=0)
        n_above = int((sub["alpha_active"] > sub["alpha_passive"]).sum())
        side = "V_k" if area == "SCMRN" else "U_k"
        ax.set_title(f"{area} (vs {side}): {n_above}/{len(sub)} above diagonal")
        ax.set_xlabel(r"$\alpha_k$ passive")
        ax.set_ylabel(r"$\alpha_k$ active" if area == areas[0] else "")
        ax.set_box_aspect(1)
        shared_lims.append((lo, hi))
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    for ax, (lo, hi) in zip(axes, shared_lims):
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ticks = ax.get_xticks()
        ticks = ticks[(ticks >= lo) & (ticks <= hi)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    _save_fig(fig, filename)


def plot_channel_figures(
    df_sess: pd.DataFrame, df_alpha: pd.DataFrame, source: str, target: str
) -> None:
    channel = _channel_tag(source, target)
    tag = _file_tag(source, target)
    areas = (target, source)
    dfc = (
        df_sess[df_sess["channel"] == channel]
        if "channel" in df_sess.columns
        else df_sess
    )
    dfa = (
        df_alpha[df_alpha["channel"] == channel]
        if (not df_alpha.empty and "channel" in df_alpha.columns)
        else df_alpha
    )
    if dfc.empty:
        print(f"No rows for channel {channel}; skipping plots.")
        return
    label = f"{source} → {target}"

    plot_readout_boxes(
        dfc,
        areas,
        filename=f"decoding_readout_box{tag}.pdf",
        title=f"Lick decoding vs communication / PCA / random k-D (k={TOP_K}; {label})",
        readout_order=MAIN_READOUTS,
    )
    plot_shared_scatter(
        dfc,
        areas,
        x_readout="pca",
        y_readout="communication",
        filename=f"decoding_comm_vs_pca{tag}.pdf",
        xlabel="Local PCA k-D accuracy",
        ylabel="Communication k-D accuracy",
        title=f"Communication vs local PCs ({label})",
    )
    plot_readout_boxes(
        dfc,
        areas,
        filename=f"decoding_passive_box{tag}.pdf",
        title=f"Active lick decoding on active vs passive axes (k={TOP_K}; {label})",
        readout_order=PASSIVE_READOUTS,
    )
    plot_shared_scatter(
        dfc,
        areas,
        x_readout="full",
        y_readout="communication",
        filename=f"decoding_comm_vs_full{tag}.pdf",
        xlabel="Full-space balanced accuracy",
        ylabel="Communication k-D accuracy",
        title=f"Communication k-D vs full space ({label})",
    )
    plot_shared_scatter(
        dfc,
        areas,
        x_readout="random",
        y_readout="communication",
        filename=f"decoding_comm_vs_random{tag}.pdf",
        xlabel="Random k-D accuracy",
        ylabel="Communication k-D accuracy",
        title=f"Communication k-D vs random k-D ({label})",
    )
    plot_shared_scatter(
        dfc,
        areas,
        x_readout="communication_passive",
        y_readout="communication",
        filename=f"decoding_comm_vs_passive{tag}.pdf",
        xlabel="Passive communication k-D accuracy",
        ylabel="Active communication k-D accuracy",
        title=f"Active vs passive communication axes ({label})",
    )
    plot_alpha_boxes(
        dfa,
        areas,
        filename=f"decoding_alpha_k{tag}.pdf",
        title=f"Choice decoder alignment with communication subspaces ({label})",
    )
    plot_alpha_paired(
        dfa,
        areas,
        filename=f"decoding_alpha_paired{tag}.pdf",
        title=f"Paired session alignment: active vs passive subspace ({label})",
    )


def cache_is_current(df_repeat: pd.DataFrame) -> bool:
    if "channel" not in df_repeat.columns or "alpha_active" not in df_repeat.columns:
        return False
    needed = {"communication_passive", "pca_passive", "full"}
    return needed.issubset(set(df_repeat["readout"].dropna()))


def load_activity() -> pd.DataFrame:
    if not activity_file.exists():
        raise FileNotFoundError(
            f"Missing {activity_file}. Run "
            "code_gendata/generate_decoding.py first."
        )
    df_activity = pd.read_pickle(activity_file)
    missing = [a for a in ALL_AREAS if a not in set(df_activity["area"].unique())]
    if missing:
        raise KeyError(
            f"df_decoding.pk is missing areas {missing}. "
            "Re-run generate_decoding.py (includes VISpl)."
        )
    data_filter = (
        (df_activity.session_number == 1)
        & (df_activity.experience_level == "Familiar")
        & (df_activity.image_name != "omitted")
        & (df_activity.is_image_novel == False)
        & (df_activity.is_change == False)
        & (df_activity.time_from_stimulus >= TIME_FROM_STIMULUS_MIN)
        & (df_activity.N >= MIN_UNITS)
        & (df_activity.area.isin(ALL_AREAS))
    )
    df_filtered = df_activity.loc[data_filter].copy()
    df_filtered["licks_bool"] = licks_to_bool(df_filtered["licks"])
    print(
        f"Loaded {len(df_filtered)} rows, "
        f"{df_filtered['session_id'].nunique()} sessions, "
        f"areas={sorted(df_filtered['area'].unique())}."
    )
    return df_filtered


def main() -> None:
    df_repeat = None
    if REPEAT_FILE.exists() and not FORCE_REBUILD:
        df_repeat = pd.read_pickle(REPEAT_FILE)
        if cache_is_current(df_repeat):
            print(f"Loaded {len(df_repeat)} rows from {REPEAT_FILE}")
        else:
            print(f"{REPEAT_FILE} is missing readouts or alpha_k; rebuilding.")
            df_repeat = None
    if df_repeat is None:
        df_filtered = load_activity()
        df_repeat = run_all_sessions(df_filtered)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_repeat.to_pickle(REPEAT_FILE, protocol=4)
        print(f"Saved {len(df_repeat)} rows to {REPEAT_FILE}")

    if df_repeat.empty:
        raise SystemExit("No decoding rows; nothing to plot.")

    df_sess = session_means(df_repeat)
    df_alpha = session_alpha(df_repeat)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_sess.to_pickle(SESSION_FILE, protocol=4)
    df_alpha.to_pickle(ALPHA_SESSION_FILE, protocol=4)
    print(f"Saved session means to {SESSION_FILE}")
    print(f"Saved alpha session means to {ALPHA_SESSION_FILE}")

    _print_diagnostics(df_sess, df_repeat, df_alpha)
    for source, target in CHANNELS:
        plot_channel_figures(df_sess, df_alpha, source, target)


if __name__ == "__main__":
    main()
# %%
