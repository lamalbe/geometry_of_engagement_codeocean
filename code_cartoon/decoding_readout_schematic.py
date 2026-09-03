# %%
"""
Two-panel cartoon of lick vs no-lick *readout* after restricting activity
to communication, local-PC, or random k-D subspaces.

Not a data figure. The three faces are schematic views of those readouts:
separability of the two ellipses on a face = how well a linear decoder
works after projecting activity into that subspace. That is the main
decoding result (balanced accuracy), not the orientation of one
full-space decoder w (alpha_k).

The 3D pair is full-space geometry. Face ellipses are the same two
classes viewed in each k-D. The bottom insets are the communication
complement (remove U_k / V_k).

  VISm  — communication face almost as split as full space; local PCs
          weaker; random overlapping; complement collapses.
  SCMRN — all three faces split (random a bit more overlap); complement
          still splits (distributed midbrain lick signal).

Writes clean_scripts/results/decoding_readout_schematic.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Patch, Polygon

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
LICK_COLOR = "#2E7D4F"
NOLICK_COLOR = "#7B5EA7"
ORIGIN_COLOR = "#222222"
DECISION_COLOR = "#333333"

AZIM_DEG = 38.0
ELEV_DEG = 22.0
AXIS_LEN = 1.0

# Full-space ellipsoids: elongated along the directions that actually
# separate the classes. VISm is flattened off the random axis.
RADII_VISM = np.array([0.20, 0.11, 0.09], dtype=float)
RADII_SCMRN = np.array([0.16, 0.15, 0.14], dtype=float)
VISM_LICK = np.array([0.70, 0.58, 0.46], dtype=float)
VISM_NOLICK = np.array([0.32, 0.44, 0.42], dtype=float)
SCMRN_LICK = np.array([0.68, 0.66, 0.64], dtype=float)
SCMRN_NOLICK = np.array([0.34, 0.36, 0.36], dtype=float)

# Face = (lick_uv, nolick_uv, (ru, rv, angle_rad)).
# xy = communication, yz = local PCs, xz = random.
VISM_FACES = {
    "xy": (np.array([0.72, 0.62]), np.array([0.28, 0.38]), (0.16, 0.08, 0.35)),
    "yz": (np.array([0.66, 0.58]), np.array([0.38, 0.44]), (0.14, 0.08, 0.50)),
    "xz": (np.array([0.54, 0.54]), np.array([0.48, 0.48]), (0.16, 0.10, 0.20)),
}
SCMRN_FACES = {
    "xy": (np.array([0.70, 0.66]), np.array([0.32, 0.36]), (0.15, 0.08, 0.40)),
    "yz": (np.array([0.68, 0.64]), np.array([0.34, 0.38]), (0.15, 0.08, 0.50)),
    "xz": (np.array([0.66, 0.62]), np.array([0.38, 0.40]), (0.14, 0.09, 0.25)),
}

# Complement insets (2D). VISm ~ chance; SCMRN still informative.
VISM_COMP = (np.array([0.0, 0.04]), np.array([0.0, -0.03]), (0.34, 0.28))
SCMRN_COMP = (np.array([0.28, 0.08]), np.array([-0.28, -0.06]), (0.26, 0.22))


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
    return float(np.dot(np.asarray(p, dtype=float), _VIEW))


def uv_to_world(face: str, uv) -> np.ndarray:
    u, v = float(uv[0]), float(uv[1])
    if face == "xy":
        return np.array([u, v, 0.0])
    if face == "xz":
        return np.array([u, 0.0, v])
    if face == "yz":
        return np.array([0.0, u, v])
    raise ValueError(face)


def add_plane(ax, corners3d: np.ndarray, color: str, face_alpha: float = 0.18) -> None:
    pts = np.vstack([proj(c) for c in corners3d])
    patch = Polygon(
        pts,
        closed=True,
        facecolor=to_rgba(color, face_alpha),
        edgecolor=to_rgba(color, 0.80),
        linewidth=1.25,
        zorder=1,
    )
    ax.add_patch(patch)


def add_arrow(
    ax,
    start3d,
    end3d,
    color: str,
    linewidth: float = 1.7,
    mutation_scale: float = 12.0,
    zorder: float = 3.0,
) -> None:
    arrow = FancyArrowPatch(
        proj(start3d),
        proj(end3d),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        zorder=zorder,
        capstyle="round",
        shrinkA=0.0,
        shrinkB=0.0,
    )
    ax.add_patch(arrow)


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
        zorder=9,
        **kwargs,
    )


def _ellipse_uv(
    center, ru: float, rv: float, n: int = 72, angle: float = 0.0
) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    c, s = np.cos(angle), np.sin(angle)
    x = ru * np.cos(t)
    y = rv * np.sin(t)
    return np.column_stack(
        [center[0] + c * x - s * y, center[1] + s * x + c * y]
    )


def add_face_ellipse(
    ax,
    face: str,
    uv_center,
    ru: float,
    rv: float,
    color: str,
    alpha: float = 0.42,
    zorder: float = 2.0,
    angle: float = 0.0,
) -> None:
    pts3 = np.vstack(
        [uv_to_world(face, uv) for uv in _ellipse_uv(uv_center, ru, rv, angle=angle)]
    )
    pts = np.vstack([proj(p) for p in pts3])
    ax.add_patch(
        Polygon(
            pts,
            closed=True,
            facecolor=to_rgba(color, alpha),
            edgecolor=color,
            linewidth=1.15,
            zorder=zorder,
        )
    )


def add_decision_line(ax, face: str, uv_a, uv_b, zorder: float = 2.4) -> None:
    a = np.asarray(uv_a, dtype=float)
    b = np.asarray(uv_b, dtype=float)
    sep = np.linalg.norm(a - b)
    if sep < 0.14:
        return
    mid = 0.5 * (a + b)
    d = (a - b) / sep
    n = np.array([-d[1], d[0]])
    half = 0.22
    p0 = uv_to_world(face, mid - half * n)
    p1 = uv_to_world(face, mid + half * n)
    q0, q1 = proj(p0), proj(p1)
    ax.plot(
        [q0[0], q1[0]],
        [q0[1], q1[1]],
        color=DECISION_COLOR,
        linewidth=1.05,
        linestyle=(0, (3.0, 2.0)),
        zorder=zorder,
        alpha=0.85,
        solid_capstyle="round",
    )


def add_drop(ax, p3d, foot3d) -> None:
    a, b = proj(p3d), proj(foot3d)
    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        color="#666666",
        linewidth=0.7,
        linestyle=(0, (2.2, 1.8)),
        alpha=0.45,
        zorder=4,
    )


def add_silhouette(ax, mu, radii, color: str, alpha: float, zorder: float) -> None:
    pmat = np.vstack([_RIGHT, _UP])
    cov = pmat @ np.diag(np.square(radii)) @ pmat.T
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 1e-12, None)
    evecs = evecs[:, order]
    angle = np.rad2deg(np.arctan2(evecs[1, 0], evecs[0, 0]))
    c = proj(mu)
    ax.add_patch(
        Ellipse(
            xy=c,
            width=2.0 * np.sqrt(evals[0]),
            height=2.0 * np.sqrt(evals[1]),
            angle=angle,
            facecolor=to_rgba(color, alpha),
            edgecolor=color,
            linewidth=1.45,
            zorder=zorder,
        )
    )


def add_rings(ax, mu, radii, color: str, zorder: float) -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 80)
    ct, st = np.cos(t), np.sin(t)
    rings = [
        np.column_stack([mu[0] + radii[0] * ct, mu[1] + radii[1] * st, np.full_like(t, mu[2])]),
        np.column_stack([mu[0] + radii[0] * ct, np.full_like(t, mu[1]), mu[2] + radii[2] * st]),
        np.column_stack([np.full_like(t, mu[0]), mu[1] + radii[1] * ct, mu[2] + radii[2] * st]),
    ]
    for ring in rings:
        pts = np.vstack([proj(p) for p in ring])
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=0.85,
            alpha=0.70,
            zorder=zorder,
        )


def draw_ellipsoid(ax, mu, radii, color: str, zorder: float) -> None:
    add_silhouette(ax, mu, radii, color, alpha=0.38, zorder=zorder)
    add_rings(ax, mu, radii, color, zorder=zorder + 0.2)
    c = proj(mu)
    ax.scatter([c[0]], [c[1]], s=14, color=color, zorder=zorder + 0.4, linewidths=0)


def draw_frame(ax, comm_label: str) -> None:
    L = AXIS_LEN
    xy = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [L, L, 0.0], [0.0, L, 0.0]])
    xz = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [L, 0.0, L], [0.0, 0.0, L]])
    yz = np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0], [0.0, L, L], [0.0, 0.0, L]])
    planes = [
        (xy, COMM_COLOR, 0.20, "xy"),
        (yz, PCA_COLOR, 0.17, "yz"),
        (xz, RANDOM_COLOR, 0.14, "xz"),
    ]
    planes.sort(key=lambda item: depth(item[0].mean(axis=0)))
    for corners, color, alpha, _face in planes:
        add_plane(ax, corners, color=color, face_alpha=alpha)

    add_arrow(ax, (0, 0, 0), (L, 0, 0), COMM_COLOR)
    add_arrow(ax, (0, 0, 0), (0, L, 0), PCA_COLOR)
    add_arrow(ax, (0, 0, 0), (0, 0, L), RANDOM_COLOR)
    o = proj((0.0, 0.0, 0.0))
    ax.scatter([o[0]], [o[1]], s=16, color=ORIGIN_COLOR, zorder=6, linewidths=0)
    _label(ax, (1.16, -0.02, 0.0), comm_label, COMM_COLOR, ha="left", va="top")
    _label(ax, (-0.02, 1.14, 0.0), "local PCs", PCA_COLOR, ha="right", va="bottom")
    _label(ax, (0.0, -0.04, 1.12), r"random $k$-D", RANDOM_COLOR, ha="center", va="bottom")


def draw_face_readouts(ax, faces: dict) -> None:
    order = [("xy", 2.0), ("yz", 2.1), ("xz", 2.2)]
    order.sort(key=lambda item: depth(uv_to_world(item[0], (0.5, 0.5))))
    for face, z0 in order:
        lick_uv, nolick_uv, shape = faces[face]
        ru, rv, ang = float(shape[0]), float(shape[1]), float(shape[2])
        add_face_ellipse(
            ax, face, lick_uv, ru, rv, LICK_COLOR, alpha=0.40, zorder=z0, angle=ang
        )
        add_face_ellipse(
            ax, face, nolick_uv, ru, rv, NOLICK_COLOR, alpha=0.40, zorder=z0, angle=ang
        )
        add_decision_line(ax, face, lick_uv, nolick_uv, zorder=z0 + 0.3)


def draw_full_space(ax, lick_mu, nolick_mu, faces: dict, radii) -> None:
    blobs = [
        (lick_mu, LICK_COLOR, 0),
        (nolick_mu, NOLICK_COLOR, 1),
    ]
    blobs.sort(key=lambda item: depth(item[0]))
    for mu, color, which in blobs:
        uv_lick_xy, uv_nolick_xy, _ = faces["xy"]
        uv_lick_yz, uv_nolick_yz, _ = faces["yz"]
        uv_lick_xz, uv_nolick_xz, _ = faces["xz"]
        feet = [
            uv_to_world("xy", uv_lick_xy if which == 0 else uv_nolick_xy),
            uv_to_world("yz", uv_lick_yz if which == 0 else uv_nolick_yz),
            uv_to_world("xz", uv_lick_xz if which == 0 else uv_nolick_xz),
        ]
        for foot in feet:
            add_drop(ax, mu, foot)
        z = 7.0 if depth(mu) > depth(0.5 * (lick_mu + nolick_mu)) else 6.4
        draw_ellipsoid(ax, mu, radii, color, zorder=z)


def _style_main(ax, title: str, subtitle: str) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=8, fontweight="medium")
    ax.text(
        0.5,
        -0.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.3,
        color="#333333",
    )


def draw_panel(
    ax,
    lick_mu,
    nolick_mu,
    faces: dict,
    radii,
    comm_label: str,
    title: str,
    subtitle: str,
) -> None:
    draw_frame(ax, comm_label=comm_label)
    draw_face_readouts(ax, faces)
    draw_full_space(ax, lick_mu, nolick_mu, faces, radii)
    corners = [
        (-0.08, -0.08, -0.08),
        (1.26, 0.0, 0.0),
        (0.0, 1.26, 0.0),
        (0.0, 0.0, 1.26),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        lick_mu,
        nolick_mu,
    ]
    pts = np.vstack([proj(c) for c in corners])
    pad = 0.10
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + 0.16)
    _style_main(ax, title, subtitle)


def draw_complement(ax, lick_xy, nolick_xy, radii, comm_label: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-0.72, 0.72)
    ax.set_ylim(-0.52, 0.58)
    ax.axis("off")
    frame = plt.Rectangle(
        (-0.68, -0.46),
        1.36,
        0.98,
        fill=True,
        facecolor="#F4F4F4",
        edgecolor="#6C757D",
        linewidth=1.15,
        linestyle=(0, (3.5, 2.0)),
        zorder=0,
    )
    ax.add_patch(frame)
    for xy, color in ((lick_xy, LICK_COLOR), (nolick_xy, NOLICK_COLOR)):
        ax.add_patch(
            Ellipse(
                xy=xy,
                width=2.0 * radii[0],
                height=2.0 * radii[1],
                facecolor=to_rgba(color, 0.40),
                edgecolor=color,
                linewidth=1.3,
                zorder=2,
            )
        )
        ax.scatter([xy[0]], [xy[1]], s=12, color=color, zorder=3, linewidths=0)
    sep = np.linalg.norm(np.asarray(lick_xy) - np.asarray(nolick_xy))
    if sep >= 0.18:
        mid = 0.5 * (np.asarray(lick_xy) + np.asarray(nolick_xy))
        d = (np.asarray(lick_xy) - np.asarray(nolick_xy)) / sep
        n = np.array([-d[1], d[0]])
        ax.plot(
            [mid[0] - 0.38 * n[0], mid[0] + 0.38 * n[0]],
            [mid[1] - 0.38 * n[1], mid[1] + 0.38 * n[1]],
            color=DECISION_COLOR,
            linewidth=1.05,
            linestyle=(0, (3.0, 2.0)),
            zorder=4,
        )
    ax.set_title(
        rf"after removing {comm_label}",
        fontsize=10,
        pad=2,
        color="#333333",
    )


def legend_handles():
    return [
        Patch(
            facecolor=to_rgba(LICK_COLOR, 0.45), edgecolor=LICK_COLOR,
            label="lick",
        ),
        Patch(
            facecolor=to_rgba(NOLICK_COLOR, 0.45), edgecolor=NOLICK_COLOR,
            label="no-lick",
        ),
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
            color=DECISION_COLOR, lw=1.15, linestyle=(0, (3.0, 2.0)),
            label="linear readout",
        ),
    ]


def main() -> None:
    fig = plt.figure(figsize=(9.6, 6.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.55, 1.28], hspace=0.18, wspace=0.16)
    ax_vism = fig.add_subplot(gs[0, 0])
    ax_scmrn = fig.add_subplot(gs[0, 1])
    ax_vism_c = fig.add_subplot(gs[1, 0])
    ax_scmrn_c = fig.add_subplot(gs[1, 1])

    draw_panel(
        ax_vism,
        VISM_LICK,
        VISM_NOLICK,
        VISM_FACES,
        RADII_VISM,
        comm_label=r"$U_k$",
        title="VISm",
        subtitle=(
            "Communication recovers nearly full-space decoding;"
            "\nlocal PCs are weaker, random $k$-D is near chance."
        ),
    )
    draw_panel(
        ax_scmrn,
        SCMRN_LICK,
        SCMRN_NOLICK,
        SCMRN_FACES,
        RADII_SCMRN,
        comm_label=r"$V_k$",
        title="SCMRN",
        subtitle=(
            "Communication $\\approx$ local PCs; random directions"
            "\nstill separate lick from no-lick."
        ),
    )
    draw_complement(ax_vism_c, *VISM_COMP, comm_label=r"$U_k$")
    draw_complement(ax_scmrn_c, *SCMRN_COMP, comm_label=r"$V_k$")

    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8.4,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=2.1,
        columnspacing=1.25,
    )
    fig.suptitle(
        "Where lick information sits (subspace readouts)",
        fontsize=13.5,
        y=0.98,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "decoding_readout_schematic.pdf"
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
