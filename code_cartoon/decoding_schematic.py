# %%
"""
Two-panel cartoon of where lick information sits relative to
VISm–SCMRN readout subspaces.

Not a data figure: subspaces are schematic, not fitted. The circle is a
direction plot (angle = a direction in population space). Each readout
is a cone/slice from the origin to the circle edge. Red marks are
components of one full-space decoder w (length ~ |e_θ^T w|).

  VISm  — only the communication slice contains w.
  SCMRN — w has weight in many directions: longer in communication and
          local PCs, shorter in a random k-D slice and in the complement.

Writes clean_scripts/results/decoding_schematic.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Wedge

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
FULL_FILL = "#F7F7F7"
FULL_EDGE = "#222222"

RADIUS = 1.0

# Slice = (center_deg, width_deg). Matplotlib Wedge uses CCW from +x.
VISM_COMM = (20.0, 52.0)
VISM_PCA = (105.0, 48.0)
VISM_RANDOM = (175.0, 40.0)

SCMRN_COMM = (40.0, 58.0)
SCMRN_PCA = (62.0, 52.0)
SCMRN_RANDOM = (155.0, 40.0)


def _unit(angle_deg: float) -> np.ndarray:
    t = np.deg2rad(angle_deg)
    return np.array([np.cos(t), np.sin(t)], dtype=float)


def _span(center: float, width: float) -> tuple[float, float]:
    return center - 0.5 * width, center + 0.5 * width


def _in_slice(angle_deg: float, center: float, width: float) -> bool:
    t1, t2 = _span(center, width)
    x = np.cos(np.deg2rad(angle_deg))
    y = np.sin(np.deg2rad(angle_deg))
    a1 = np.deg2rad(t1)
    a2 = np.deg2rad(t2)
    # Angle from start of slice, CCW, must be in [0, width].
    start = np.array([np.cos(a1), np.sin(a1)])
    cross = start[0] * y - start[1] * x
    dot = start[0] * x + start[1] * y
    delta = np.rad2deg(np.arctan2(cross, dot)) % 360.0
    return delta <= width + 1e-6


def add_full_space(ax) -> None:
    fill = Circle(
        (0.0, 0.0),
        RADIUS,
        facecolor=FULL_FILL,
        edgecolor="none",
        zorder=0,
    )
    edge = Circle(
        (0.0, 0.0),
        RADIUS,
        facecolor="none",
        edgecolor=FULL_EDGE,
        linewidth=1.8,
        zorder=6,
    )
    ax.add_patch(fill)
    ax.add_patch(edge)


def add_slice(
    ax,
    center: float,
    width: float,
    color: str,
    dashed: bool = False,
    alpha: float = 0.40,
) -> Wedge:
    t1, t2 = _span(center, width)
    wedge = Wedge(
        (0.0, 0.0),
        RADIUS,
        t1,
        t2,
        facecolor="none" if dashed else color,
        edgecolor=color,
        linewidth=1.6,
        linestyle=(0, (4.0, 2.4)) if dashed else "-",
        alpha=1.0 if dashed else alpha,
        zorder=2,
    )
    ax.add_patch(wedge)
    return wedge


def add_w_component(
    ax,
    angle_deg: float,
    length: float,
    linewidth: float = 1.8,
    mutation_scale: float = 11.0,
    alpha: float = 1.0,
) -> None:
    if length < 0.04:
        return
    end = _unit(angle_deg) * min(length, 0.92 * RADIUS)
    arrow = FancyArrowPatch(
        (0.0, 0.0),
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=LICK_COLOR,
        alpha=alpha,
        zorder=5,
        capstyle="round",
        shrinkA=0.0,
        shrinkB=0.0,
    )
    ax.add_patch(arrow)


def _label_slice(ax, center: float, text: str, color: str) -> None:
    pos = 1.12 * _unit(center)
    ax.text(
        pos[0],
        pos[1],
        text,
        ha="center",
        va="center",
        fontsize=10,
        color=color,
        zorder=7,
    )


def _style_axes(ax, title: str, subtitle: str) -> None:
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.38, 1.28)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=6, fontweight="medium")
    ax.text(
        0.0,
        -1.24,
        subtitle,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#333333",
    )


def draw_vism(ax) -> None:
    add_full_space(ax)
    add_slice(ax, *VISM_RANDOM, color=RANDOM_COLOR, dashed=True)
    add_slice(ax, *VISM_PCA, color=PCA_COLOR, alpha=0.38)
    add_slice(ax, *VISM_COMM, color=COMM_COLOR, alpha=0.48)
    add_w_component(
        ax, angle_deg=VISM_COMM[0], length=0.88, linewidth=2.4, mutation_scale=14
    )
    ax.text(
        -0.15,
        -0.62,
        "complement",
        ha="center",
        va="center",
        fontsize=8,
        color="#555555",
        style="italic",
        zorder=7,
    )
    _label_slice(ax, VISM_COMM[0], r"$U_k$", COMM_COLOR)
    _style_axes(
        ax,
        "VISm",
        "Only the communication slice contains the lick decoder.",
    )


def _scmrn_component_length(angle_deg: float) -> float:
    in_comm = _in_slice(angle_deg, *SCMRN_COMM)
    in_pca = _in_slice(angle_deg, *SCMRN_PCA)
    in_rand = _in_slice(angle_deg, *SCMRN_RANDOM)
    if in_comm or in_pca:
        return 0.82 if (in_comm and in_pca) else 0.70
    if in_rand:
        return 0.38
    return 0.26


def draw_scmrn(ax) -> None:
    add_full_space(ax)
    add_slice(ax, *SCMRN_RANDOM, color=RANDOM_COLOR, dashed=True)
    add_slice(ax, *SCMRN_PCA, color=PCA_COLOR, alpha=0.36)
    add_slice(ax, *SCMRN_COMM, color=COMM_COLOR, alpha=0.42)
    for ang in np.arange(0.0, 360.0, 22.5):
        add_w_component(
            ax,
            angle_deg=float(ang),
            length=_scmrn_component_length(float(ang)),
            linewidth=1.35,
            mutation_scale=9.0,
            alpha=0.92,
        )
    _label_slice(ax, SCMRN_COMM[0], r"$V_k$", COMM_COLOR)
    _style_axes(
        ax,
        "SCMRN",
        r"One decoder; $w$ has weight in many directions."
        "\nLonger in communication / local PCs, shorter elsewhere.",
    )


def legend_handles():
    return [
        Line2D(
            [0], [0],
            color=FULL_EDGE, lw=1.8, label="full space",
        ),
        Patch(
            facecolor=COMM_COLOR, edgecolor=COMM_COLOR, alpha=0.45,
            label="communication $k$-D",
        ),
        Patch(
            facecolor=PCA_COLOR, edgecolor=PCA_COLOR, alpha=0.38,
            label="local PCs",
        ),
        Line2D(
            [0], [0],
            color=RANDOM_COLOR, lw=1.6, linestyle=(0, (4.0, 2.4)),
            label="random $k$-D",
        ),
        Line2D(
            [0], [0],
            color=LICK_COLOR, lw=2.0,
            label=r"components of $w$",
        ),
    ]


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))
    draw_vism(axes[0])
    draw_scmrn(axes[1])
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=2.2,
        columnspacing=1.4,
    )
    fig.suptitle(
        "Where lick information sits (VISm → SCMRN)",
        fontsize=13.5,
        y=1.02,
    )
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "decoding_schematic.pdf"
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
