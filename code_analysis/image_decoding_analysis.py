# %%
"""
Image-identity decoding relative to VIS–SCMRN communication subspaces.

Reads clean_scripts/data/df_decoding.pk (from generate_decoding.py).
Cohort: day-1 Familiar, image_set G only, viewing bins with
0.1 s <= time_from_stimulus < 0.25 s, non-change, non-novel, non-omitted,
N >= 30. Labels are the 8 G image names, not licks.

One sample = one image presentation (flash): mean spike counts over bins
of that flash. Do not treat 50 ms bins as independent trials.

Channels: VISm → SCMRN (main), VISpl → SCMRN, and LGd → VISpl
(image identity only; not used for lick decoding).
Per session, 8-way multinomial logistic on *active* flashes, from:

  1. Full space.
  2. Active communication k-D (U_k in source, V_k in target).
  3. Orthogonal complement of that communication subspace (not PCs).
  4. Random k-D (dimensionality control).
  5. Local top-k PCs (active).
  6. Active flashes projected onto passive communication / PCA axes.

SVD and PCA are fit on flash-averaged matrices. Chance accuracy = 1/8.

Figures: clean_scripts/results/image_decoding_*.pdf
Tables:  clean_scripts/data/df_image_decoding_{repeat,session}.pk
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr, svd as scipy_svd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# %%
# -------------
#  CONFIG
# -------------

TOP_K = 5
VAR_EPS = 1e-12
N_IMAGES = 8
CHANCE = 1.0 / N_IMAGES
IMAGE_SET = "G"
ALIGN_KEYS = ["stim_onset_ms", "image_name"]

CHANNELS = [
    ("VISm", "SCMRN"),
    ("VISpl", "SCMRN"),
    ("LGd", "VISpl"),
]
ALL_AREAS = sorted({a for pair in CHANNELS for a in pair})

LOGISTIC_C = 0.1
N_RANDOM_DECODE = 5
N_CV_FOLDS = 5
MIN_PER_IMAGE = 8
MIN_UNITS = 30
TIME_FROM_STIMULUS_MIN = 0.1
TIME_FROM_STIMULUS_MAX = 0.25
RANDOM_SEED = 0
SAVE_PLOTS = True
SHOW_PLOTS = False
FORCE_REBUILD = False

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

DATA_DIR = _SCRIPT_DIR.parent / "data"
RESULTS_DIR = _SCRIPT_DIR.parent / "results"
activity_file = DATA_DIR / "df_decoding.pk"
REPEAT_FILE = DATA_DIR / "df_image_decoding_repeat.pk"
SESSION_FILE = DATA_DIR / "df_image_decoding_session.pk"

AREA_FILL = {
    "LGd": "#80B918",
    "VISpl": "#0077B6",
    "VISm": "#34A0A4",
    "SCMRN": "#C36F09",
}

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
]
READOUT_ORDER = [
    "full",
    "communication",
    "pca",
    "random",
    "orthogonal",
    "communication_passive",
    "pca_passive",
]
READOUT_LABELS = {
    "full": "full",
    "communication": "comm. k-D",
    "pca": "PCA k-D",
    "random": "random k-D",
    "orthogonal": "orth. complement",
    "communication_passive": "comm. (passive)",
    "pca_passive": "PCA (passive)",
}
READOUT_COLORS = {
    "full": "#333333",
    "communication": "#C36F09",
    "pca": "#34A0A4",
    "random": "#888888",
    "orthogonal": "#6C757D",
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


def aggregate_flashes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (session, area, active, flash): mean counts in the window."""
    work = df.copy()
    work["stim_onset_ms"] = np.round(
        (work["bin_start_times"] - work["time_from_stimulus"]) * 1000.0
    ).astype(np.int64)
    keys = ["session_id", "area", "active", "stim_onset_ms"]
    rows: List[Dict[str, Any]] = []
    for key, g in work.groupby(keys, sort=False):
        session_id, area, active, stim_onset_ms = key
        X = _stack_consistent_vectors(g["spikecounts_area"])
        if X.size == 0:
            continue
        names = g["image_name"].astype(str).unique()
        if len(names) != 1:
            continue
        row = {
            "session_id": session_id,
            "area": area,
            "active": bool(active),
            "stim_onset_ms": int(stim_onset_ms),
            "image_name": names[0],
            "spikecounts_area": X.mean(axis=0),
            "n_bins": int(X.shape[0]),
            "N": int(g["N"].iloc[0]),
        }
        if "licks_bool" in g.columns:
            row["lick_any"] = bool(g["licks_bool"].to_numpy().any())
        rows.append(row)
    return pd.DataFrame(rows)


def build_area_matrix(df: pd.DataFrame, area: str) -> Tuple[np.ndarray, pd.DataFrame]:
    sub = df[df["area"] == area].copy()
    if sub.empty:
        return np.empty((0, 0), dtype=np.float64), sub
    sub = sub.sort_values(ALIGN_KEYS, kind="mergesort")
    sub = sub.drop_duplicates(subset=ALIGN_KEYS, keep="first")
    X = _stack_consistent_vectors(sub["spikecounts_area"])
    return X, sub


def build_matched_area_pair(
    df: pd.DataFrame, source_area: str, target_area: str
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    keep_cols = ALIGN_KEYS + ["spikecounts_area"]
    extra = ["lick_any"] if "lick_any" in df.columns else []
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
        max_iter=4000,
    )


def cv_multiclass_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_CV_FOLDS,
    C: float = LOGISTIC_C,
    random_state: int = 0,
) -> Tuple[float, float]:
    y = np.asarray(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    _, counts = np.unique(y, return_counts=True)
    n_folds_eff = min(n_folds, int(counts.min()))
    if n_folds_eff < 2:
        raise ValueError("Need >=2 samples per class for stratified CV.")
    cv = StratifiedKFold(n_splits=n_folds_eff, shuffle=True, random_state=random_state)
    clf = make_logistic(C=C)
    pred = cross_val_predict(clf, X, y, cv=cv)
    return float(accuracy_score(y, pred)), float(balanced_accuracy_score(y, pred))


def _safe_cv(X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[float, float]:
    if X.size == 0 or X.shape[0] < 4 or X.shape[1] < 1:
        return np.nan, np.nan
    try:
        return cv_multiclass_accuracy(X, y, random_state=random_state)
    except ValueError:
        return np.nan, np.nan


# %%
# ---------------------------
#  SESSION LOOP
# ---------------------------


def _class_ok(y: np.ndarray) -> bool:
    classes, counts = np.unique(y, return_counts=True)
    return len(classes) >= N_IMAGES and int(counts.min()) >= MIN_PER_IMAGE


def decode_readouts_for_area(
    X: np.ndarray,
    y: np.ndarray,
    B_comm: np.ndarray,
    B_pca: np.ndarray,
    B_comm_pas: Optional[np.ndarray],
    B_pca_pas: Optional[np.ndarray],
    rng: np.random.Generator,
    session_id: str,
    area: str,
    channel: str,
    n_lick_flashes: int,
) -> List[Dict[str, Any]]:
    if not _class_ok(y):
        classes, counts = np.unique(y, return_counts=True)
        print(
            f"    Skipping {area}: need {N_IMAGES} images with "
            f">={MIN_PER_IMAGE} flashes each "
            f"(got {len(classes)} images, min n={counts.min() if len(counts) else 0})."
        )
        return []
    if B_comm.shape[0] != X.shape[1]:
        print(
            f"    Skipping {area}: neuron mismatch "
            f"(decoder {X.shape[1]} vs comm {B_comm.shape[0]})."
        )
        return []

    Xc, _ = mean_center_columns(X)
    k_comm = B_comm.shape[1]
    k_pca = B_pca.shape[1] if B_pca.size else 0
    rows: List[Dict[str, Any]] = []

    def _row(readout, acc, bal, n_rand=1):
        return {
            "session_id": session_id,
            "channel": channel,
            "area": area,
            "readout": readout,
            "k": TOP_K,
            "n_neurons": int(X.shape[1]),
            "n_flashes": int(X.shape[0]),
            "n_images": int(len(np.unique(y))),
            "n_lick_flashes": int(n_lick_flashes),
            "n_random": int(n_rand),
            "accuracy": acc,
            "balanced_accuracy": bal,
        }

    readouts: Dict[str, np.ndarray] = {
        "full": Xc,
        "communication": project_onto(Xc, B_comm),
        "orthogonal": residual_from(Xc, B_comm),
    }
    if k_pca > 0 and B_pca.shape[0] == Xc.shape[1]:
        readouts["pca"] = project_onto(Xc, B_pca)
    if B_comm_pas is not None and B_comm_pas.shape[0] == Xc.shape[1]:
        readouts["communication_passive"] = project_onto(Xc, B_comm_pas)
    if B_pca_pas is not None and B_pca_pas.size and B_pca_pas.shape[0] == Xc.shape[1]:
        readouts["pca_passive"] = project_onto(Xc, B_pca_pas)

    for readout, X_use in readouts.items():
        acc, bal = _safe_cv(X_use, y, RANDOM_SEED)
        rows.append(_row(readout, acc, bal))

    rand_acc, rand_bal = [], []
    for i_rand in range(N_RANDOM_DECODE):
        try:
            B_rand = random_orthonormal_subspace(Xc.shape[1], k_comm, rng)
        except ValueError:
            continue
        acc, bal = _safe_cv(project_onto(Xc, B_rand), y, RANDOM_SEED + i_rand)
        if np.isfinite(acc):
            rand_acc.append(acc)
            rand_bal.append(bal)
    if rand_acc:
        rows.append(
            _row("random", float(np.mean(rand_acc)), float(np.mean(rand_bal)),
                 n_rand=len(rand_acc))
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


def run_all_sessions(df_flashes: pd.DataFrame) -> pd.DataFrame:
    session_ids = sorted(df_flashes["session_id"].unique())
    rng = np.random.default_rng(RANDOM_SEED)
    rows: List[Dict[str, Any]] = []

    print(
        f"\nImage-identity decoding: 8-way multinomial, k={TOP_K}, "
        f"{N_RANDOM_DECODE} random subspaces, {N_CV_FOLDS}-fold CV, "
        f"chance={CHANCE:.3f}."
    )
    print("  Channels:", ", ".join(f"{s}→{t}" for s, t in CHANNELS))
    print("  SVD/PCA fit on flash-averaged presentations (active and passive).")

    for i_session, session_id in enumerate(session_ids):
        print(f"\nSession {i_session + 1}/{len(session_ids)}: {session_id}")
        df_sess = df_flashes[df_flashes["session_id"] == session_id]
        df_active = df_sess[df_sess["active"] == True]
        df_passive = df_sess[df_sess["active"] == False]
        if df_active.empty:
            print("  Skipping: no active flashes.")
            continue

        for source, target in CHANNELS:
            channel = f"{source}_to_{target}"
            decode_areas = (target, source)
            print(f"  Channel {channel}")

            U_comm, V_comm, X_src, X_tgt, merged = _fit_comm_svd(
                df_active, source, target
            )
            if U_comm is None:
                print("    Skipping: insufficient active aligned flashes / SVD.")
                continue
            print(
                f"    active aligned flashes={X_src.shape[0]}, "
                f"{source} N={U_comm.shape[0]}, {target} N={V_comm.shape[0]}"
            )
            B_comm_by_area = {source: U_comm, target: V_comm}

            U_pas, V_pas, X_pas_src, _, _ = _fit_comm_svd(df_passive, source, target)
            B_comm_pas_by_area: Dict[str, Optional[np.ndarray]] = {
                source: None, target: None,
            }
            if U_pas is not None:
                B_comm_pas_by_area[source] = U_pas
                B_comm_pas_by_area[target] = V_pas
                print(f"    passive aligned flashes={X_pas_src.shape[0]}")
            else:
                print("    passive communication SVD skipped.")

            for area in decode_areas:
                X_dec, meta_dec = build_area_matrix(df_active, area)
                if X_dec.size == 0:
                    print(f"    Skipping {area}: no active flash matrix.")
                    continue
                y = meta_dec["image_name"].astype(str).to_numpy()
                if len(y) != X_dec.shape[0]:
                    n = min(len(y), X_dec.shape[0])
                    X_dec, y = X_dec[:n], y[:n]
                n_lick = (
                    int(meta_dec["lick_any"].sum())
                    if "lick_any" in meta_dec.columns
                    else 0
                )
                B_pca = local_pca_basis(X_dec, top_k=TOP_K)
                X_pas_area, _ = build_area_matrix(df_passive, area)
                B_pca_pas = (
                    local_pca_basis(X_pas_area, top_k=TOP_K)
                    if X_pas_area.size
                    else None
                )
                area_rows = decode_readouts_for_area(
                    X=X_dec,
                    y=y,
                    B_comm=B_comm_by_area[area],
                    B_pca=B_pca,
                    B_comm_pas=B_comm_pas_by_area[area],
                    B_pca_pas=B_pca_pas,
                    rng=rng,
                    session_id=str(session_id),
                    area=area,
                    channel=channel,
                    n_lick_flashes=n_lick,
                )
                rows.extend(area_rows)

    return pd.DataFrame(rows)


# %%
# ---------------------------
#  SUMMARIES AND PLOTS
# ---------------------------


def session_means(df_repeat: pd.DataFrame) -> pd.DataFrame:
    return df_repeat.copy()


def pivot_readout(df_sess: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    idx = ["session_id", "channel", "area", "n_flashes", "n_neurons"]
    idx = [c for c in idx if c in df_sess.columns]
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


def _print_diagnostics(df_sess: pd.DataFrame) -> None:
    channels = sorted(df_sess["channel"].unique())
    for channel in channels:
        dfc = df_sess[df_sess["channel"] == channel]
        areas = [a for a in ("SCMRN", "VISm", "VISpl", "LGd") if a in set(dfc["area"])]
        print(f"\n=== Session-mean 8-way accuracy ({channel}; chance={CHANCE:.3f}) ===")
        for area in areas:
            sub = dfc[dfc["area"] == area]
            print(f"\n{area}:")
            for readout in READOUT_ORDER:
                vals = sub.loc[sub["readout"] == readout, "accuracy"].dropna()
                if vals.empty:
                    continue
                print(
                    f"  {READOUT_LABELS[readout]:18s}  "
                    f"n={len(vals):2d}  mean={vals.mean():.3f}  "
                    f"median={vals.median():.3f}"
                )

        wide = pivot_readout(dfc)
        print(f"\n=== Spearman accuracy ({channel}) ===")
        pairs = [
            ("full", "communication"),
            ("communication", "pca"),
            ("communication", "random"),
            ("full", "orthogonal"),
            ("communication", "communication_passive"),
            ("pca", "pca_passive"),
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


def plot_readout_boxes(
    df_sess: pd.DataFrame,
    areas: Tuple[str, ...],
    filename: str,
    title: str,
    readout_order: Optional[List[str]] = None,
) -> None:
    order = readout_order if readout_order is not None else MAIN_READOUTS
    present = [r for r in order if r in set(df_sess["readout"])]
    fig, axes = plt.subplots(1, len(areas), figsize=(5.2 * len(areas), 4.4), sharey=True)
    if len(areas) == 1:
        axes = [axes]
    for ax, area in zip(axes, areas):
        sub = df_sess[df_sess["area"] == area]
        box_data, colors, labels = [], [], []
        for readout in present:
            vals = sub.loc[sub["readout"] == readout, "accuracy"].dropna()
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
        ax.axhline(CHANCE, color="k", ls="--", lw=0.8, alpha=0.6, zorder=0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(area)
        ax.set_ylabel("8-way accuracy" if area == areas[0] else "")
        ax.set_ylim(0.0, 1.0)
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
        lo = 0.0
        hi = float(max(vals.max() if not vals.empty else 0.2, CHANCE + 0.05))
        pad = 0.05 * (hi - lo)
        hi = min(1.0, hi + pad)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7, zorder=0)
        ax.axhline(CHANCE, color="k", ls=":", lw=0.8, alpha=0.5, zorder=0)
        ax.axvline(CHANCE, color="k", ls=":", lw=0.8, alpha=0.5, zorder=0)
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


def plot_channel_figures(df_sess: pd.DataFrame, source: str, target: str) -> None:
    channel = _channel_tag(source, target)
    tag = _file_tag(source, target)
    areas = (target, source)
    dfc = df_sess[df_sess["channel"] == channel]
    if dfc.empty:
        print(f"No rows for channel {channel}; skipping plots.")
        return
    label = f"{source} → {target}"
    plot_readout_boxes(
        dfc,
        areas,
        filename=f"image_decoding_readout_box{tag}.pdf",
        title=f"Image identity (set G, 8-way) vs communication / PCA / random "
        f"(k={TOP_K}; {label})",
        readout_order=MAIN_READOUTS,
    )
    plot_readout_boxes(
        dfc,
        areas,
        filename=f"image_decoding_passive_box{tag}.pdf",
        title=f"Active image decoding on active vs passive axes "
        f"(k={TOP_K}; {label})",
        readout_order=PASSIVE_READOUTS,
    )
    plot_shared_scatter(
        dfc, areas,
        x_readout="pca", y_readout="communication",
        filename=f"image_decoding_comm_vs_pca{tag}.pdf",
        xlabel="Local PCA k-D accuracy",
        ylabel="Communication k-D accuracy",
        title=f"Image identity: communication vs local PCs ({label})",
    )
    plot_shared_scatter(
        dfc, areas,
        x_readout="full", y_readout="communication",
        filename=f"image_decoding_comm_vs_full{tag}.pdf",
        xlabel="Full-space 8-way accuracy",
        ylabel="Communication k-D accuracy",
        title=f"Image identity: communication vs full space ({label})",
    )
    plot_shared_scatter(
        dfc, areas,
        x_readout="random", y_readout="communication",
        filename=f"image_decoding_comm_vs_random{tag}.pdf",
        xlabel="Random k-D accuracy",
        ylabel="Communication k-D accuracy",
        title=f"Image identity: communication vs random k-D ({label})",
    )


def cache_is_current(df_repeat: pd.DataFrame) -> bool:
    if "channel" not in df_repeat.columns:
        return False
    needed = {"communication_passive", "pca_passive", "full", "orthogonal"}
    if not needed.issubset(set(df_repeat["readout"].dropna())):
        return False
    needed_channels = {f"{s}_to_{t}" for s, t in CHANNELS}
    return needed_channels.issubset(set(df_repeat["channel"].dropna()))


def load_activity() -> pd.DataFrame:
    if not activity_file.exists():
        raise FileNotFoundError(
            f"Missing {activity_file}. Run "
            "clean_scripts/code_gendata/generate_decoding.py first."
        )
    df_activity = pd.read_pickle(activity_file)
    missing = [a for a in ALL_AREAS if a not in set(df_activity["area"].unique())]
    if missing:
        raise KeyError(f"df_decoding.pk is missing areas {missing}.")
    if "image_set" not in df_activity.columns:
        raise KeyError("df_decoding.pk has no image_set column.")
    data_filter = (
        (df_activity.session_number == 1)
        & (df_activity.experience_level == "Familiar")
        & (df_activity.image_set == IMAGE_SET)
        & (df_activity.image_name != "omitted")
        & (df_activity.is_image_novel == False)
        & (df_activity.is_change == False)
        & (df_activity.time_from_stimulus >= TIME_FROM_STIMULUS_MIN)
        & (df_activity.time_from_stimulus < TIME_FROM_STIMULUS_MAX)
        & (df_activity.N >= MIN_UNITS)
        & (df_activity.area.isin(ALL_AREAS))
    )
    df_filtered = df_activity.loc[data_filter].copy()
    df_filtered["licks_bool"] = licks_to_bool(df_filtered["licks"])
    print(
        f"Loaded {len(df_filtered)} bins (image_set={IMAGE_SET}), "
        f"{df_filtered['session_id'].nunique()} sessions, "
        f"areas={sorted(df_filtered['area'].unique())}."
    )
    print("Aggregating bins into flashes...")
    df_flashes = aggregate_flashes(df_filtered)
    n_img = df_flashes["image_name"].nunique()
    print(
        f"  {len(df_flashes)} flashes, "
        f"{df_flashes['session_id'].nunique()} sessions, "
        f"{n_img} image names."
    )
    if "lick_any" in df_flashes.columns:
        frac = df_flashes.loc[df_flashes["active"] == True, "lick_any"].mean()
        print(f"  Fraction of active flashes with any lick in-window: {frac:.3f}")
    return df_flashes


def main() -> None:
    df_repeat = None
    if REPEAT_FILE.exists() and not FORCE_REBUILD:
        df_repeat = pd.read_pickle(REPEAT_FILE)
        if cache_is_current(df_repeat):
            print(f"Loaded {len(df_repeat)} rows from {REPEAT_FILE}")
        else:
            print(f"{REPEAT_FILE} is missing readouts; rebuilding.")
            df_repeat = None
    if df_repeat is None:
        df_flashes = load_activity()
        df_repeat = run_all_sessions(df_flashes)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_repeat.to_pickle(REPEAT_FILE, protocol=4)
        print(f"Saved {len(df_repeat)} rows to {REPEAT_FILE}")

    if df_repeat.empty:
        raise SystemExit("No image-decoding rows; nothing to plot.")

    df_sess = session_means(df_repeat)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_sess.to_pickle(SESSION_FILE, protocol=4)
    print(f"Saved session table to {SESSION_FILE}")

    _print_diagnostics(df_sess)
    for source, target in CHANNELS:
        plot_channel_figures(df_sess, source, target)


if __name__ == "__main__":
    main()
# %%
