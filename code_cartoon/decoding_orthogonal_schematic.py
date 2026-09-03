# %%
"""
Two-panel cartoon of lick-decoder weight on three orthogonal readouts.

Not a data figure: the three subspaces are drawn as an orthogonal 3-frame
so the components of w are readable. In the data they need not be orthogonal.

Each panel is a room-corner: three axes (communication, local PCs, random k-D)
and the three coordinate planes they span. The red arrow is one full-space
decoder w; dashed drops and axis ticks are its components |e^T w|.

  VISm  — large communication component, smaller local-PC, tiny random.
  SCMRN — communication ≈ local PCs, random a bit shorter (distributed).

Writes clean_scripts/results/decoding_orthogonal_schematic.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Polygon

# %%
SAVE_PLOTS = True
SHOW_PLOTS = False

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

RESULTS_DIR = _SCRIPT_DIR.parent / "results"

COMM_COLOR = "#C36F09"
PCA_COLOR = "#34A0A4"
RANDOM_COLOR = "#6C757D"
LICK_COLOR = "#9B2226"
ORIGIN_COLOR = "#222222"

# Camera: look from the first octant into the origin (room corner).
# x = communication, y = local PCs, z = random (vertical).
AZIM_DEG = 38.0
ELEV_DEG = 22.0
AXIS_LEN = 1.0

# w = (communication, local PCs, random). Schematic, not fitted.
# VISm: comm > PCs >> random. SCMRN: comm ≈ PCs, random a bit shorter.
VISM_W = np.array([0.88, 0.54, 0.16], dtype=float)
SCMRN_W = np.array([0.76, 0.72, 0.58], dtype=float)


def _camera_basis(
    azim_deg: float = AZIM_DEG, elev_deg: float = ELEV_DEG
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.deg2rad(azim_deg)
    e = np.deg2rad(elev_deg)
    view = np.array(
        [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)],
        dtype=float,
    )
    view = view / np.linalg.norm(view)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, view)
    right = right / np.linalg.norm(right)
    up = np.cross(view, right)
    up = up / np.linalg.norm(up)
    return right, up, view


_RIGHT, _UP, _VIEW = _camera_basis()


def proj(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.array([np.dot(p, _RIGHT), np.dot(p, _UP)], dtype=float)


def depth(p) -> float:
    """Larger = closer to the camera."""
    return float(np.dot(np.asarray(p, dtype=float), _VIEW))


def add_plane(
    ax,
    corners3d: np.ndarray,
    color: str,
    face_alpha: float = 0.22,
) -> None:
    pts = np.vstack([proj(c) for c in corners3d])
    patch = Polygon(
        pts,
        closed=True,
        facecolor=to_rgba(color, face_alpha),
        edgecolor=to_rgba(color, 0.85),
        linewidth=1.35,
        zorder=1,
    )
    ax.add_patch(patch)


def add_arrow(
    ax,
    start3d,
    end3d,
    color: str,
    linewidth: float = 1.8,
    mutation_scale: float = 12.0,
    alpha: float = 1.0,
    zorder: float = 5.0,
    linestyle: str = "solid",
) -> None:
    p0 = proj(start3d)
    p1 = proj(end3d)
    arrow = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        zorder=zorder,
        capstyle="round",
        shrinkA=0.0,
        shrinkB=0.0,
        linestyle=linestyle,
    )
    ax.add_patch(arrow)


def add_segment(
    ax,
    start3d,
    end3d,
    color: str,
    linewidth: float = 1.1,
    linestyle: str = (0, (3.2, 2.0)),
    alpha: float = 0.85,
    zorder: float = 4.0,
) -> None:
    p0 = proj(start3d)
    p1 = proj(end3d)
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
    )


def _label(ax, p3d, text: str, color: str, **kwargs) -> None:
    p = proj(p3d)
    ax.text(
        p[0],
        p[1],
        text,
        color=color,
        fontsize=kwargs.pop("fontsize", 10),
        ha=kwargs.pop("ha", "center"),
        va=kwargs.pop("va", "center"),
        zorder=8,
        **kwargs,
    )


def _style_axes(ax, title: str, subtitle: str) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=8, fontweight="medium")
    ax.text(
        0.5,
        -0.04,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#333333",
    )


def draw_frame(ax, comm_label: str) -> None:
    L = AXIS_LEN
    xy = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [L, L, 0.0], [0.0, L, 0.0]])
    xz = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [L, 0.0, L], [0.0, 0.0, L]])
    yz = np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0], [0.0, L, L], [0.0, 0.0, L]])
    # Face tint follows the axis that is most distinctive in this view.
    planes = [
        (xy, COMM_COLOR, 0.24),
        (yz, PCA_COLOR, 0.20),
        (xz, RANDOM_COLOR, 0.16),
    ]
    planes.sort(key=lambda item: depth(item[0].mean(axis=0)))
    for corners, color, alpha in planes:
        add_plane(ax, corners, color=color, face_alpha=alpha)

    # Thin full-length frame, then the subspace identity lives on the axes.
    add_arrow(
        ax, (0, 0, 0), (L, 0, 0), COMM_COLOR, linewidth=1.7, mutation_scale=12, zorder=3
    )
    add_arrow(
        ax, (0, 0, 0), (0, L, 0), PCA_COLOR, linewidth=1.7, mutation_scale=12, zorder=3
    )
    add_arrow(
        ax, (0, 0, 0), (0, 0, L), RANDOM_COLOR, linewidth=1.7, mutation_scale=12, zorder=3
    )

    o = proj((0.0, 0.0, 0.0))
    ax.scatter(
        [o[0]], [o[1]], s=18, color=ORIGIN_COLOR, zorder=6, linewidths=0
    )

    _label(ax, (1.18, -0.02, 0.0), comm_label, COMM_COLOR, ha="left", va="top")
    _label(ax, (-0.02, 1.16, 0.0), "local PCs", PCA_COLOR, ha="right", va="bottom")
    _label(ax, (0.0, -0.04, 1.14), r"random $k$-D", RANDOM_COLOR, ha="center", va="bottom")


def draw_decoder(ax, w: np.ndarray) -> None:
    wx, wy, wz = (float(w[0]), float(w[1]), float(w[2]))
    tip = (wx, wy, wz)
    components = [
        ((wx, 0.0, 0.0), COMM_COLOR),
        ((0.0, wy, 0.0), PCA_COLOR),
        ((0.0, 0.0, wz), RANDOM_COLOR),
    ]
    for foot, color in components:
        add_segment(
            ax,
            tip,
            foot,
            color=LICK_COLOR,
            linewidth=1.05,
            linestyle=(0, (2.6, 1.8)),
            alpha=0.75,
            zorder=4,
        )
        # Projection magnitude on this subspace (same length as the dashed drop).
        add_segment(
            ax,
            (0.0, 0.0, 0.0),
            foot,
            color=color,
            linewidth=4.2,
            linestyle="solid",
            alpha=0.95,
            zorder=5,
        )
        p = proj(foot)
        ax.scatter(
            [p[0]], [p[1]], s=22, color=color, zorder=6, edgecolors=LICK_COLOR, linewidths=0.6
        )
    add_arrow(
        ax,
        (0.0, 0.0, 0.0),
        tip,
        color=LICK_COLOR,
        linewidth=2.5,
        mutation_scale=14.0,
        zorder=7,
    )


def draw_panel(
    ax,
    w: np.ndarray,
    comm_label: str,
    title: str,
    subtitle: str,
) -> None:
    draw_frame(ax, comm_label=comm_label)
    draw_decoder(ax, w)
    corners = [
        (-0.08, -0.08, -0.08),
        (1.28, 0.0, 0.0),
        (0.0, 1.28, 0.0),
        (0.0, 0.0, 1.28),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        w * 1.05,
    ]
    pts = np.vstack([proj(c) for c in corners])
    pad = 0.10
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + 0.18)
    _style_axes(ax, title, subtitle)


def legend_handles():
    return [
        Patch(
            facecolor=to_rgba(COMM_COLOR, 0.45), edgecolor=COMM_COLOR,
            label="communication $k$-D",
        ),
        Patch(
            facecolor=to_rgba(PCA_COLOR, 0.40), edgecolor=PCA_COLOR,
            label="local PCs",
        ),
        Patch(
            facecolor=to_rgba(RANDOM_COLOR, 0.38), edgecolor=RANDOM_COLOR,
            label="random $k$-D",
        ),
        Line2D(
            [0], [0],
            color=LICK_COLOR, lw=2.2,
            label=r"decoder $w$",
        ),
        Line2D(
            [0], [0],
            color=LICK_COLOR, lw=1.4, linestyle=(0, (2.6, 1.8)),
            label=r"components of $w$",
        ),
    ]


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.8))
    draw_panel(
        axes[0],
        VISM_W,
        comm_label=r"$U_k$",
        title="VISm",
        subtitle=(
            "Large weight on communication, some on local PCs,"
            "\nlittle in a random $k$-D slice."
        ),
    )
    draw_panel(
        axes[1],
        SCMRN_W,
        comm_label=r"$V_k$",
        title="SCMRN",
        subtitle=(
            r"Similar weight on communication and local PCs;"
            "\nrandom directions still carry a substantial component."
        ),
    )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.04),
        handlelength=2.2,
        columnspacing=1.4,
    )
    fig.suptitle(
        "Where lick information sits (orthogonal readouts)",
        fontsize=13.5,
        y=1.02,
    )
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "decoding_orthogonal_schematic.pdf"
    if SAVE_PLOTS:
        fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Saved figure: {path}")
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
# %%
