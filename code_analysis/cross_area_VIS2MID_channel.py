# %%
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

# %%
# -------------
#  CONFIG
# -------------
SAVE_PLOT = True
# True: only SESSION_IDS. False: every session that survives the image/area filters.
USE_MANUAL_SESSIONS = True
# SESSION_IDS = ["1093638203"] # Clean example for 6 panel plot
SESSION_IDS = ["1095138995"] # Clean example for pooled SVD trial clouds

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

import config

DATA_DIR = config.DATA_DIR
RESULTS_DIR = config.results_dir("cross_area_VIS2MID_channel")

IMAGE_NAME = "im078_r"
TIME_FROM_STIMULUS_MIN = 0.1

# Diverging: blue (negative) → white (0) → red (positive); symmetric vmin/vmax centers at 0.
MATRIX_CMAP = "RdBu_r"
DIAGONAL_BAD_COLOR = "#000000"
CROSS_AREA_SVD_RANK = 1
# Pooled active+passive cross-cov SVD: trial clouds in top mode subspaces.
SVD_CLOUD_N_DIMS_2D = 2
SVD_CLOUD_N_DIMS_3D = 3
# Trial-cloud point style (see neural_clouds.py).
SVD_CLOUD_SCATTER_ALPHA = 0.7
SVD_CLOUD_SCATTER_SIZE = 10
ACTIVE_COLORS_BY_AREA = {
    "VISm": {True: "#2A7FFF", False: "#AA0000"},
    "SCMRN": {True: "#2A7FFF", False: "#AA0000"},
}

# %%
# -------------
#  LOAD DATASET
# -------------
activity_file = DATA_DIR / "df_cortex2mid.pk"
df_activity = pd.read_pickle(activity_file)

data_filter = (
    (df_activity.session_number == 1)
    & (df_activity.experience_level == "Familiar")
    & (df_activity.image_name != "omitted")
    & (df_activity.is_image_novel == False)
    & (df_activity.is_change == False)
    & (df_activity.time_from_stimulus >= TIME_FROM_STIMULUS_MIN)
    & (df_activity.N >= 50)
    & (df_activity.area.isin(["VISm", "SCMRN"]))
    & (df_activity.image_name == IMAGE_NAME)
)
if USE_MANUAL_SESSIONS:
    data_filter = data_filter & (
        df_activity.session_id.astype(str).isin([str(s) for s in SESSION_IDS])
    )
df_activity_filtered = df_activity[data_filter].drop(
    ["session_number", "experience_level", "is_image_novel", "is_change", "image_set"],
    axis=1,
)

# %%
# ---------------------------
#  HELPER FUNCTIONS
# ---------------------------
def _stack_consistent_vectors(series: pd.Series) -> np.ndarray:
    vecs = [np.asarray(v, dtype=float).ravel() for v in series]
    lengths = pd.Series([len(v) for v in vecs])
    common_len = int(lengths.mode().iloc[0])
    vecs = [v for v in vecs if len(v) == common_len]
    if len(vecs) == 0:
        return np.empty((0, 0))
    return np.vstack(vecs)


def build_matched_area_pair_matrices(
    df: pd.DataFrame, session_id: str, active_state: bool, area_a: str, area_b: str
) -> Tuple[np.ndarray, np.ndarray]:
    session_block = df[(df["session_id"].astype(str) == str(session_id)) & (df["active"] == active_state)]

    if area_a == area_b:
        area_block = session_block[session_block["area"] == area_a].copy()
        if area_block.empty:
            return np.empty((0, 0)), np.empty((0, 0))
        area_block = area_block.sort_values("bin_start_times")
        x_area = _stack_consistent_vectors(area_block["spikecounts_area"])
        if x_area.size == 0:
            return np.empty((0, 0)), np.empty((0, 0))
        return x_area, x_area

    key_cols = ["bin_start_times"]
    keep_cols = key_cols + ["spikecounts_area"]
    area_a_block = session_block[session_block["area"] == area_a][keep_cols].copy()
    area_b_block = session_block[session_block["area"] == area_b][keep_cols].copy()
    merged = area_a_block.merge(area_b_block, on=key_cols, suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        return np.empty((0, 0)), np.empty((0, 0))

    x_a = _stack_consistent_vectors(merged["spikecounts_area_a"])
    x_b = _stack_consistent_vectors(merged["spikecounts_area_b"])
    return x_a, x_b


def build_area_pair_matrix(
    df: pd.DataFrame,
    session_id: str,
    active_state: bool,
    area_a: str,
    area_b: str,
) -> np.ndarray:
    """Mean-centered sample covariance (np.cov; no per-neuron z-scoring)."""
    x_a, x_b = build_matched_area_pair_matrices(
        df,
        session_id=str(session_id),
        active_state=active_state,
        area_a=area_a,
        area_b=area_b,
    )
    if x_a.size == 0 or x_b.size == 0:
        return np.empty((0, 0))

    if area_a == area_b:
        return np.cov(x_a, rowvar=False)

    x_joint = np.hstack([x_a, x_b])
    joint_matrix = np.cov(x_joint, rowvar=False)
    n_a = x_a.shape[1]
    return joint_matrix[:n_a, n_a:]


def _symmetric_limits_from_matrices(
    matrices: List[np.ndarray],
    clip_percentile: float = 99.0,
) -> Tuple[float, float]:
    pieces = []
    for m in matrices:
        v = np.asarray(m, dtype=float).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            pieces.append(v)
    if not pieces:
        return -1e-12, 1e-12
    vals = np.concatenate(pieces)
    vmax = float(np.nanpercentile(np.abs(vals), clip_percentile))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = float(np.nanmax(np.abs(vals)))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1e-12
    return -vmax, vmax


def svd_rank_k_approximation(c: np.ndarray, k: int) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    if c.size == 0:
        return c
    m, n = c.shape
    k_eff = int(min(max(k, 0), m, n))
    if k_eff == 0:
        return np.zeros_like(c)
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    return (u[:, :k_eff] * s[:k_eff]) @ vt[:k_eff, :]


def _combine_symmetric_within_area_split(
    active: np.ndarray,
    passive: np.ndarray,
) -> np.ndarray:
    if active.shape != passive.shape or active.ndim != 2 or active.shape[0] != active.shape[1]:
        raise ValueError("active and passive must be equal square matrices for triangle split.")
    n = active.shape[0]
    out = np.full((n, n), np.nan, dtype=float)
    iu = np.triu_indices(n, k=1)
    il = np.tril_indices(n, k=-1)
    out[iu] = passive[iu]
    out[il] = active[il]
    return out


def plot_session_composite_six_panel(
    session_id: str,
    cov_vism_scmrn_active: np.ndarray,
    cov_vism_vism_active: np.ndarray,
    cov_vism_vism_passive: np.ndarray,
    cov_scmrn_scmrn_active: np.ndarray,
    cov_scmrn_scmrn_passive: np.ndarray,
    cov_scmrn_vism_passive: np.ndarray,
    *,
    rank_k: int,
    save_plot: bool,
    results_dir: Path,
    image_name: str,
    cmap_name: str = MATRIX_CMAP,
    clip_percentile: float = 99.0,
) -> None:
    metric_label = "mean-centered covariance"

    svd_vism_scmrn_active = svd_rank_k_approximation(cov_vism_scmrn_active, rank_k)
    svd_scmrn_vism_passive = svd_rank_k_approximation(cov_scmrn_vism_passive, rank_k)
    panel_bm = np.asarray(cov_scmrn_vism_passive, dtype=float)

    combined_vism = _combine_symmetric_within_area_split(
        cov_vism_vism_active,
        cov_vism_vism_passive,
    )
    combined_scmrn = _combine_symmetric_within_area_split(
        cov_scmrn_scmrn_active,
        cov_scmrn_scmrn_passive,
    )

    vmin, vmax = _symmetric_limits_from_matrices(
        [
            svd_vism_scmrn_active,
            cov_vism_scmrn_active,
            combined_vism,
            combined_scmrn,
            panel_bm,
            svd_scmrn_vism_passive,
        ],
        clip_percentile=clip_percentile,
    )

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=DIAGONAL_BAD_COLOR)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    panels = [
        (axes[0, 0], svd_vism_scmrn_active,
         f"Rank-{rank_k} SVD | VISm–SCMRN active", "SCMRN neurons", "VISm neurons"),
        (axes[0, 1], cov_vism_scmrn_active,
         f"VISm–SCMRN | active | session {session_id}", "SCMRN neurons", "VISm neurons"),
        (axes[0, 2], combined_vism,
         "VISm–VISm | active above diag / passive below", "VISm neurons", "VISm neurons"),
        (axes[1, 0], combined_scmrn,
         "SCMRN–SCMRN | active above diag / passive below", "SCMRN neurons", "SCMRN neurons"),
        (axes[1, 1], panel_bm,
         "SCMRN–VISm | passive", "VISm neurons", "SCMRN neurons"),
        (axes[1, 2], svd_scmrn_vism_passive,
         f"Rank-{rank_k} SVD | SCMRN–VISm passive", "VISm neurons", "SCMRN neurons"),
    ]

    last_im = None
    for ax, data, title, xlabel, ylabel in panels:
        last_im = ax.imshow(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            origin="lower",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_box_aspect(1)

    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.85, label=metric_label)
    fig.suptitle(
        f"Composite {metric_label} — session {session_id}, image {image_name!r}, "
        f"t ≥ {TIME_FROM_STIMULUS_MIN}s "
        f"(cmap={cmap_name!r}, symmetric limits; rank-k={rank_k})"
    )

    if save_plot:
        out_dir = results_dir / "composite_six_panel_cross_cov"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            out_dir
            / f"mean_centered_cov_rank{rank_k}_session_{session_id}_{image_name}_t{TIME_FROM_STIMULUS_MIN}.pdf"
        )
        fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _session_matrices(df: pd.DataFrame, sid: str) -> tuple:
    return (
        build_area_pair_matrix(df, sid, True, "VISm", "SCMRN"),
        build_area_pair_matrix(df, sid, True, "VISm", "VISm"),
        build_area_pair_matrix(df, sid, False, "VISm", "VISm"),
        build_area_pair_matrix(df, sid, True, "SCMRN", "SCMRN"),
        build_area_pair_matrix(df, sid, False, "SCMRN", "SCMRN"),
        build_area_pair_matrix(df, sid, False, "SCMRN", "VISm"),
    )


def build_cross_area_neuron_by_trial(
    df: pd.DataFrame,
    session_id: str,
    active_state: bool,
    area_a: str = "VISm",
    area_b: str = "SCMRN",
) -> Tuple[np.ndarray, np.ndarray]:
    """Aligned trial matrices: shape (n_neurons, n_trials) per area."""
    x_a, x_b = build_matched_area_pair_matrices(
        df, session_id=str(session_id), active_state=active_state, area_a=area_a, area_b=area_b
    )
    if x_a.size == 0:
        return np.empty((0, 0)), np.empty((0, 0))
    return x_a.T, x_b.T


def pooled_cross_cov_svd(
    x_a: np.ndarray,
    x_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean-center each area across pooled trials (columns), cross-cov SVD.
    x_a, x_b: (n_neurons, n_trials). Returns U, s, V (columns are modes), mu_a, mu_b, C.
    """
    if x_a.size == 0 or x_b.size == 0:
        raise ValueError("Empty activity matrices.")
    if x_a.shape[1] != x_b.shape[1]:
        raise ValueError("Trial counts must match between areas.")
    t = x_a.shape[1]
    if t < 2:
        raise ValueError("Need at least 2 pooled trials for cross-covariance SVD.")

    mu_a = x_a.mean(axis=1)
    mu_b = x_b.mean(axis=1)
    x_a_c = x_a - mu_a[:, np.newaxis]
    x_b_c = x_b - mu_b[:, np.newaxis]
    c = (x_a_c @ x_b_c.T) / (t - 1)
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    v = vt.T
    return u, s, v, mu_a, mu_b, c


def trial_scores_in_modes(
    x_area: np.ndarray,
    modes: np.ndarray,
    mu: np.ndarray,
    n_dims: int,
) -> np.ndarray:
    """Project trials onto top ``n_dims`` mode columns; returns (n_trials, n_dims)."""
    n_dims = int(min(n_dims, modes.shape[1], x_area.shape[1]))
    if n_dims < 1:
        return np.empty((x_area.shape[1], 0))
    x_c = x_area - mu[:, np.newaxis]
    return (modes[:, :n_dims].T @ x_c).T


def _symmetric_limits_per_dim(
    score_sets: Sequence[np.ndarray],
    n_dims: int,
    pad: float = 0.08,
) -> List[Tuple[float, float]]:
    """Symmetric (lo, hi) per mode axis, pooled over all provided score matrices."""
    limits: List[Tuple[float, float]] = []
    for d in range(n_dims):
        parts = [s[:, d] for s in score_sets if s.size and s.shape[1] > d]
        if not parts:
            limits.append((-1.0, 1.0))
            continue
        vals = np.concatenate(parts)
        lim = float(np.max(np.abs(vals)))
        if lim <= 0:
            lim = 1e-12
        lim *= 1.0 + pad
        limits.append((-lim, lim))
    return limits


def _active_colors_for_area(area_label: str) -> dict:
    if area_label not in ACTIVE_COLORS_BY_AREA:
        raise KeyError(f"Unknown area {area_label!r}; expected one of {list(ACTIVE_COLORS_BY_AREA)}.")
    return ACTIVE_COLORS_BY_AREA[area_label]


def _trial_cloud_scatter(ax, xyz: Tuple[np.ndarray, ...], color: str, label: str, *, ndim: int) -> None:
    """Scatter trial points (neural_clouds style: small, rasterized, no edges)."""
    kw = dict(color=color, alpha=SVD_CLOUD_SCATTER_ALPHA, s=SVD_CLOUD_SCATTER_SIZE, rasterized=True, label=label)
    if ndim == 2:
        ax.scatter(xyz[0], xyz[1], **kw)
    else:
        ax.scatter(xyz[0], xyz[1], xyz[2], **kw)


def plot_pooled_svd_trial_clouds(
    *,
    session_id: str,
    image_name: str,
    area_label: str,
    scores_active: np.ndarray,
    scores_passive: np.ndarray,
    singular_values: np.ndarray,
    n_dims: int,
    save_plot: bool,
    results_dir: Path,
    axis_limits: Optional[List[Tuple[float, float]]] = None,
) -> None:
    """One figure: active vs passive trial clouds in pooled cross-area SVD axes."""
    colors = _active_colors_for_area(area_label)
    color_active = colors[True]
    color_passive = colors[False]

    s_str = ", ".join(f"σ{i + 1}={singular_values[i]:.3g}" for i in range(min(n_dims, len(singular_values))))
    base_title = (
        f"{area_label} — pooled VISm–SCMRN cross-cov modes\n"
        f"session {session_id}, image {image_name!r}, t ≥ {TIME_FROM_STIMULUS_MIN}s ({s_str})"
    )
    fname_suffix = f"session_{session_id}_{image_name}_t{TIME_FROM_STIMULUS_MIN}"

    if save_plot:
        out_dir = results_dir / "pooled_svd_trial_clouds"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    if n_dims == 2:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        _trial_cloud_scatter(
            ax,
            (scores_active[:, 0], scores_active[:, 1]),
            color_active,
            "active",
            ndim=2,
        )
        _trial_cloud_scatter(
            ax,
            (scores_passive[:, 0], scores_passive[:, 1]),
            color_passive,
            "passive",
            ndim=2,
        )
        if axis_limits is None:
            axis_limits = _symmetric_limits_per_dim(
                [scores_active, scores_passive], n_dims=2
            )
        ax.set_xlim(axis_limits[0])
        ax.set_ylim(axis_limits[1])
        ax.set_xlabel("Pooled cross-mode 1 score")
        ax.set_ylabel("Pooled cross-mode 2 score")
        ax.set_title(base_title)
        ax.legend(loc="best")
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, c=".5", lw=0.6, zorder=0)
        ax.axvline(0, c=".5", lw=0.6, zorder=0)
        if save_plot:
            path = out_dir / f"{area_label}_trial_cloud_2d_{fname_suffix}.pdf"
            fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {path}")
        else:
            plt.show()
        return

    if n_dims == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        _trial_cloud_scatter(
            ax,
            (scores_active[:, 0], scores_active[:, 1], scores_active[:, 2]),
            color_active,
            "active",
            ndim=3,
        )
        _trial_cloud_scatter(
            ax,
            (scores_passive[:, 0], scores_passive[:, 1], scores_passive[:, 2]),
            color_passive,
            "passive",
            ndim=3,
        )
        if axis_limits is None:
            axis_limits = _symmetric_limits_per_dim(
                [scores_active, scores_passive], n_dims=3
            )
        ax.set_xlim(axis_limits[0])
        ax.set_ylim(axis_limits[1])
        ax.set_zlim(axis_limits[2])
        ax.set_xlabel("Pooled cross-mode 1")
        ax.set_ylabel("Pooled cross-mode 2")
        ax.set_zlabel("Pooled cross-mode 3")
        ax.set_title(base_title)
        ax.legend(loc="best")
        if save_plot:
            path = out_dir / f"{area_label}_trial_cloud_3d_{fname_suffix}.pdf"
            fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {path}")
        else:
            plt.show()
        return

    raise ValueError(f"n_dims must be 2 or 3, got {n_dims}")


def run_pooled_svd_trial_cloud_analysis(
    df: pd.DataFrame,
    session_id: str,
    image_name: str,
    *,
    save_plot: bool,
    results_dir: Path,
) -> None:
    """Pooled cross-cov SVD basis; project active/passive trials per area (2D and 3D)."""
    x_vism_act, x_scmrn_act = build_cross_area_neuron_by_trial(df, session_id, True)
    x_vism_pas, x_scmrn_pas = build_cross_area_neuron_by_trial(df, session_id, False)

    if x_vism_act.size == 0 or x_vism_pas.size == 0:
        raise RuntimeError("Missing cross-area trial data for active or passive.")

    if x_vism_act.shape[0] != x_vism_pas.shape[0] or x_scmrn_act.shape[0] != x_scmrn_pas.shape[0]:
        raise RuntimeError(
            "Neuron counts differ between active and passive; cannot pool cross-covariance."
        )

    x_vism_pool = np.hstack([x_vism_act, x_vism_pas])
    x_scmrn_pool = np.hstack([x_scmrn_act, x_scmrn_pas])
    u, s, v, mu_vism, mu_scmrn, _ = pooled_cross_cov_svd(x_vism_pool, x_scmrn_pool)

    for n_dims in (SVD_CLOUD_N_DIMS_2D, SVD_CLOUD_N_DIMS_3D):
        if n_dims > len(s):
            print(f"  skip {n_dims}D clouds: only {len(s)} singular values available.")
            continue
        scores_vism_act = trial_scores_in_modes(x_vism_act, u, mu_vism, n_dims)
        scores_vism_pas = trial_scores_in_modes(x_vism_pas, u, mu_vism, n_dims)
        scores_scmrn_act = trial_scores_in_modes(x_scmrn_act, v, mu_scmrn, n_dims)
        scores_scmrn_pas = trial_scores_in_modes(x_scmrn_pas, v, mu_scmrn, n_dims)

        all_scores = [scores_vism_act, scores_vism_pas, scores_scmrn_act, scores_scmrn_pas]
        shared_axis_limits = _symmetric_limits_per_dim(all_scores, n_dims=n_dims)

        plot_pooled_svd_trial_clouds(
            session_id=session_id,
            image_name=image_name,
            area_label="VISm",
            scores_active=scores_vism_act,
            scores_passive=scores_vism_pas,
            singular_values=s,
            n_dims=n_dims,
            save_plot=save_plot,
            results_dir=results_dir,
            axis_limits=shared_axis_limits,
        )
        plot_pooled_svd_trial_clouds(
            session_id=session_id,
            image_name=image_name,
            area_label="SCMRN",
            scores_active=scores_scmrn_act,
            scores_passive=scores_scmrn_pas,
            singular_values=s,
            n_dims=n_dims,
            save_plot=save_plot,
            results_dir=results_dir,
            axis_limits=shared_axis_limits,
        )


def _session_ids_to_run(df: pd.DataFrame) -> List[str]:
    """Manual SESSION_IDS, or all session ids present after the image/area filters."""
    available = sorted({str(s) for s in df["session_id"].astype(str).unique()})
    if not USE_MANUAL_SESSIONS:
        return available
    wanted = [str(s) for s in SESSION_IDS]
    missing = [s for s in wanted if s not in set(available)]
    if missing:
        raise RuntimeError(
            f"Manual session(s) not in filtered data for image {IMAGE_NAME!r}: {missing}"
        )
    return wanted


def _session_has_required_areas(df: pd.DataFrame, session_id: str) -> bool:
    block = df[df["session_id"].astype(str) == str(session_id)]
    required = {"VISm", "SCMRN"}
    areas_active = set(block[block["active"] == True]["area"].unique())
    areas_passive = set(block[block["active"] == False]["area"].unique())
    return required.issubset(areas_active) and required.issubset(areas_passive)


def _session_mats_ok(mats: tuple) -> bool:
    (
        cov_vism_scmrn_active,
        cov_vism_vism_active,
        cov_vism_vism_passive,
        cov_scmrn_scmrn_active,
        cov_scmrn_scmrn_passive,
        cov_scmrn_vism_passive,
    ) = mats
    if any(m.size == 0 for m in mats):
        return False
    if cov_vism_vism_active.shape != cov_vism_vism_passive.shape:
        return False
    if cov_scmrn_scmrn_active.shape != cov_scmrn_scmrn_passive.shape:
        return False
    return True


# %%
# ---------------------------
#  SIX-PANEL PLOT (per session, fixed image)
# ---------------------------
session_ids = _session_ids_to_run(df_activity_filtered)
n_six_panel = 0
for sid in session_ids:
    if not _session_has_required_areas(df_activity_filtered, sid):
        msg = (
            f"Session {sid} / image {IMAGE_NAME}: need VISm and SCMRN in both active and passive."
        )
        if USE_MANUAL_SESSIONS:
            raise RuntimeError(msg)
        print(f"  skip six-panel: {msg}")
        continue

    mats = _session_matrices(df_activity_filtered, sid)
    if not _session_mats_ok(mats):
        msg = f"Could not build all six matrices for session {sid}, image {IMAGE_NAME}."
        if USE_MANUAL_SESSIONS:
            raise RuntimeError(msg)
        print(f"  skip six-panel: {msg}")
        continue

    plot_session_composite_six_panel(
        session_id=sid,
        cov_vism_scmrn_active=mats[0],
        cov_vism_vism_active=mats[1],
        cov_vism_vism_passive=mats[2],
        cov_scmrn_scmrn_active=mats[3],
        cov_scmrn_scmrn_passive=mats[4],
        cov_scmrn_vism_passive=mats[5],
        rank_k=CROSS_AREA_SVD_RANK,
        save_plot=SAVE_PLOT,
        results_dir=RESULTS_DIR,
        image_name=IMAGE_NAME,
    )
    n_six_panel += 1

print(
    f"Done. Six-panel plot(s) for {n_six_panel} session(s), image {IMAGE_NAME!r}, "
    f"t ≥ {TIME_FROM_STIMULUS_MIN}s (mean-centered cov; rank-k={CROSS_AREA_SVD_RANK})."
)
if SAVE_PLOT:
    print(f"  six-panel: {(RESULTS_DIR / 'composite_six_panel_cross_cov').resolve()}")

# %%
# ---------------------------
#  POOLED SVD TRIAL CLOUDS (2D and 3D, per session, fixed image)
# ---------------------------
session_ids = _session_ids_to_run(df_activity_filtered)
n_clouds = 0
print("Pooled cross-cov SVD trial clouds (active vs passive, shared basis)...")
for sid in session_ids:
    try:
        run_pooled_svd_trial_cloud_analysis(
            df_activity_filtered,
            sid,
            IMAGE_NAME,
            save_plot=SAVE_PLOT,
            results_dir=RESULTS_DIR,
        )
    except RuntimeError as exc:
        if USE_MANUAL_SESSIONS:
            raise
        print(f"  skip trial clouds session {sid}: {exc}")
        continue
    n_clouds += 1

print(
    f"Done. Trial clouds for {n_clouds} session(s), image {IMAGE_NAME!r} "
    f"({SVD_CLOUD_N_DIMS_2D}D + {SVD_CLOUD_N_DIMS_3D}D, VISm and SCMRN)."
)
if SAVE_PLOT:
    print(f"  trial clouds: {(RESULTS_DIR / 'pooled_svd_trial_clouds').resolve()}")

# %%
