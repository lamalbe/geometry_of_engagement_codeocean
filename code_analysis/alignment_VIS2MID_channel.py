# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# LGd: {True: '#80B918', False: '#BFD200'},
# VIS: {True: '#34A0A4', False: '#48CAE4'},
# SCMRN: {True: '#C36F09', False: '#EEBA0B'}


# %%
SAVE_PLOTS = False
CHANNEL = "VISm_to_SCMRN"
PC_K_MAX = 5
ACTIVE_COLOR = "#2A7FFF"
PASSIVE_COLOR = "#AA0000"
AP_PALETTE = {True: ACTIVE_COLOR, False: PASSIVE_COLOR}

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

DATA_DIR = _SCRIPT_DIR.parent / "data"
RESULTS_DIR = _SCRIPT_DIR.parent / "results"
results_dir = RESULTS_DIR

# %%
df_basis_vectors = pd.read_pickle(DATA_DIR / "df_basis_vectors.pk")
df_max_cosine_active_passive = pd.read_pickle(DATA_DIR / "df_max_cosine_active_passive.pk")
df_max_cosine_within_condition = pd.read_pickle(DATA_DIR / "df_max_cosine_within_condition.pk")
df_subspace_overlap_pc = pd.read_pickle(DATA_DIR / "df_subspace_overlap_pc.pk")
df_subspace_overlap_svd = pd.read_pickle(DATA_DIR / "df_subspace_overlap_svd.pk")
df_subspace_overlap_cross_in_local = pd.read_pickle(DATA_DIR / "df_cross_covariance_in_local_subspace.pk") # nuclear_norm(P_src_k.T @ C_cross @ P_tgt_k) / sum(s[:k])
df_subspace_overlap_local_in_cross = pd.read_pickle(DATA_DIR / "df_local_variance_in_cross_subspace.pk") # trace(U_k.T @ C_source @ U_k) / sum(λ_source[:k]) and trace(V_k.T @ C_target @ V_k) / sum(λ_target[:k])

# Rebuild SVD spectrum from df_basis_vectors (singular values of the cross-covariance).
_sv_rows = []
_df_sv = df_basis_vectors[
    (df_basis_vectors["basis_type"] == "SV_source")
    & (df_basis_vectors["channel"] == CHANNEL)
]
for (session_id, image_name, channel, active), grp in _df_sv.groupby(
    ["session_id", "image_name", "channel", "active"], sort=False
):
    s = grp.sort_values("component_rank")["singular_value"].to_numpy(dtype=float)
    denom = float(np.nansum(s ** 2))
    pr = float((np.nansum(s) ** 2) / denom) if denom > 0 else np.nan
    _sv_rows.append(
        {
            "session_id": session_id,
            "image_name": image_name,
            "channel": channel,
            "active": active,
            "singular_values_top10": s,
            "participation_ratio_sv": pr,
        }
    )
df_svd_spectrum = pd.DataFrame(_sv_rows)

# %%
# --------------------------------------
#   SVD PARTICIPATION RATIO AND SPECTRUM
# --------------------------------------

plot_filter = df_svd_spectrum.channel == CHANNEL
df_plot = df_svd_spectrum[plot_filter]

g_pr_sv = sns.catplot(
    data=df_plot,
    y="participation_ratio_sv",
    x="active",
    kind="box",
)

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_pr_sv.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_participation_ratio_sv_box.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(g_pr_sv.figure)
else:
    plt.show()

sv_ratio_rows = []
for (session_id, image_name, channel), grp in df_plot.groupby(
    ["session_id", "image_name", "channel"], sort=False
):
    act_rows = grp.loc[grp["active"] == True, "singular_values_top10"]
    pas_rows = grp.loc[grp["active"] == False, "singular_values_top10"]
    if act_rows.empty or pas_rows.empty:
        continue
    s_active = np.asarray(act_rows.iloc[0], dtype=float).ravel()
    s_passive = np.asarray(pas_rows.iloc[0], dtype=float).ravel()
    if s_active.size != s_passive.size:
        continue
    ratios = np.divide(
        s_active,
        s_passive,
        out=np.full(s_active.shape, np.nan, dtype=float),
        where=s_passive != 0,
    )
    for k, ratio in enumerate(ratios, start=1):
        if not np.isfinite(ratio):
            continue
        sv_ratio_rows.append(
            {
                "session_id": session_id,
                "image_name": image_name,
                "channel": channel,
                "k": k,
                "sv_ratio": float(ratio),
            }
        )

df_sv_ratio = pd.DataFrame(sv_ratio_rows)
df_sv_ratio["k"] = pd.Categorical(
    df_sv_ratio["k"], categories=list(range(1, 11)), ordered=True
)

g_sv_ratio = sns.catplot(
    data=df_sv_ratio,
    x="k",
    y="sv_ratio",
    kind="point",
    errorbar="se",
    linestyle="none",
    capsize=0.12,
    height=4.0,
    aspect=1.2,
)
g_sv_ratio.set_axis_labels("Singular value rank k", "Active / passive singular value ratio")
g_sv_ratio.figure.suptitle(
    f"Cross-SVD spectrum ratio ({CHANNEL}); error bars = SE over session×image",
    y=1.02,
)
g_sv_ratio.ax.axhline(1.0, ls="--", c=".4", lw=1, zorder=0)

if SAVE_PLOTS:
    g_sv_ratio.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_singular_value_ratio_point.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(g_sv_ratio.figure)
else:
    plt.show()

# %%
# --------------------------------------
#   PC SPECTRA (first 5 eigenvalues)
# --------------------------------------

df_pc_spectrum = df_basis_vectors[
    (df_basis_vectors["basis_type"] == "PC")
    & (df_basis_vectors["channel"] == CHANNEL)
    & (df_basis_vectors["component_rank"] <= PC_K_MAX)
].copy()
df_pc_spectrum["component_rank"] = pd.Categorical(
    df_pc_spectrum["component_rank"],
    categories=list(range(1, PC_K_MAX + 1)),
    ordered=True,
)

g_pc_spectrum = sns.catplot(
    data=df_pc_spectrum,
    x="component_rank",
    y="eigenvalue",
    hue="active",
    hue_order=[True, False],
    col="area",
    kind="point",
    errorbar="se",
    palette=AP_PALETTE,
    linestyle="none",
    capsize=0.12,
    height=4.0,
    aspect=1.2,
)
g_pc_spectrum.set_axis_labels("PC rank k", "Eigenvalue")
g_pc_spectrum.figure.suptitle(
    f"Local PC spectra ({CHANNEL}); first {PC_K_MAX} eigenvalues; "
    "blue = active, red = passive; error bars = SE over session×image",
    y=1.02,
)

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_pc_spectrum.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_pc_spectrum_point.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(g_pc_spectrum.figure)
else:
    plt.show()

pc_ratio_rows = []
for (session_id, image_name, channel, area), grp in df_pc_spectrum.groupby(
    ["session_id", "image_name", "channel", "area"], sort=False
):
    for k in range(1, PC_K_MAX + 1):
        act_vals = grp.loc[
            (grp["active"] == True) & (grp["component_rank"] == k), "eigenvalue"
        ]
        pas_vals = grp.loc[
            (grp["active"] == False) & (grp["component_rank"] == k), "eigenvalue"
        ]
        if act_vals.empty or pas_vals.empty:
            continue
        lam_act = float(act_vals.iloc[0])
        lam_pas = float(pas_vals.iloc[0])
        if lam_pas == 0 or not np.isfinite(lam_act) or not np.isfinite(lam_pas):
            continue
        pc_ratio_rows.append(
            {
                "session_id": session_id,
                "image_name": image_name,
                "channel": channel,
                "area": area,
                "k": k,
                "eigenvalue_ratio": lam_act / lam_pas,
            }
        )

df_pc_ratio = pd.DataFrame(pc_ratio_rows)
df_pc_ratio["k"] = pd.Categorical(
    df_pc_ratio["k"], categories=list(range(1, PC_K_MAX + 1)), ordered=True
)

g_pc_ratio = sns.catplot(
    data=df_pc_ratio,
    x="k",
    y="eigenvalue_ratio",
    col="area",
    kind="point",
    errorbar="se",
    linestyle="none",
    capsize=0.12,
    height=4.0,
    aspect=1.2,
)
g_pc_ratio.set_axis_labels("PC rank k", "Active / passive eigenvalue ratio")
g_pc_ratio.figure.suptitle(
    f"Local PC spectrum ratio ({CHANNEL}); first {PC_K_MAX} eigenvalues; "
    "error bars = SE over session×image",
    y=1.02,
)
for ax in g_pc_ratio.axes.flat:
    ax.axhline(1.0, ls="--", c=".4", lw=1, zorder=0)

if SAVE_PLOTS:
    g_pc_ratio.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_pc_spectrum_ratio_point.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(g_pc_ratio.figure)
else:
    plt.show()

# %%
# ------------------
#  COSINE SIMILARITY ACROSS CONDITIONS
# ------------------

COSINE_YLIM = (-0.02, 1.02)

df_max_cosine_active_passive.keys()
plot_filter = (
    (df_max_cosine_active_passive.comparison != "PC_SV")
    & (df_max_cosine_active_passive.comparison != "SV_PC")
    & (df_max_cosine_active_passive.channel == CHANNEL)
)
df_plot = df_max_cosine_active_passive[plot_filter]

INNER_KWS = dict(
    box_width=15,
    whis_width=2,
    color=".8",
    solid_capstyle="round",
    solid_joinstyle="round",
)

g_violin = sns.catplot(
    data=df_plot,
    y="abs_cosine",
    x="area",
    row="comparison",
    kind="violin",
    cut=0,
    inner="box",
    inner_kws=INNER_KWS,
    sharey=True,
    bw_adjust=1.3,
)
for ax in g_violin.axes.flat:
    ax.set_ylim(COSINE_YLIM)

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_violin.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_across_abs_cosine_violin.pdf",
        format="pdf",
        bbox_inches="tight",
    )
else:
    plt.show()

# %%
# ------------------
#  COSINE SIMILARITY WITHIN CONDITIONS
# -----------------

df_max_cosine_within_condition.keys()
plot_filter = df_max_cosine_within_condition.channel == CHANNEL
df_plot = df_max_cosine_within_condition[plot_filter]

INNER_KWS = dict(
    box_width=15,
    whis_width=2,
    color=".8",
    solid_capstyle="round",
    solid_joinstyle="round",
)

g_violin = sns.catplot(
    data=df_plot,
    y="abs_cosine",
    x="area",
    row="active",
    kind="violin",
    cut=0,
    inner="box",
    inner_kws=INNER_KWS,
    sharey=True,
    bw_adjust=1.3,
)
for ax in g_violin.axes.flat:
    ax.set_ylim(COSINE_YLIM)

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_violin.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_within_abs_cosine_violin.pdf",
        format="pdf",
        bbox_inches="tight",
    )
else:
    plt.show()

g_kde = sns.displot(
    data=df_plot[df_plot.area == "SCMRN"],
    x="abs_cosine",
    row="active",
    kind="kde",
    bw_adjust=1.1,
    clip=COSINE_YLIM,
)
for ax in g_kde.axes.flat:
    ax.set_xlim(COSINE_YLIM)

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_kde.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_within_SCMRN_abs_cosine_kde.pdf",
        format="pdf",
        bbox_inches="tight",
    )
else:
    plt.show()

# %%
# ------------------
#  SVD SUBSPACE OVERLAP ACROSS CONDITIONS
# ------------------

plot_filter = (df_subspace_overlap_svd.channel == CHANNEL) & (
    df_subspace_overlap_svd.direction == "active_captured_by_passive"
)
df_plot = df_subspace_overlap_svd[plot_filter]

INNER_KWS = dict(
    box_width=15,
    whis_width=2,
    color=".8",
    solid_capstyle="round",
    solid_joinstyle="round",
)

g_violin = sns.catplot(
    data=df_plot,
    y="overlap",
    kind="violin",
    cut=0,
    inner="box",
    inner_kws=INNER_KWS,
    sharey=True,
    bw_adjust=1.3,
)
for ax in g_violin.axes.flat:
    ax.set_ylim(0.0, 1.0)

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_violin.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_svd_overlap_violin.pdf",
        format="pdf",
        bbox_inches="tight",
    )
else:
    plt.show()

OVERLAP_LIM = (0.0, 1.0)

g_kde = sns.displot(
    data=df_plot,
    x="overlap",
    kind="kde",
    bw_adjust=1.1,
    # cut=0,
    clip=OVERLAP_LIM,
)
for ax in g_kde.axes.flat:
    ax.set_xlim(OVERLAP_LIM)

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_kde.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_svd_overlap_kde.pdf",
        format="pdf",
        bbox_inches="tight",
    )
else:
    plt.show()

# %%
# ------------------
#  PC SUBSPACE OVERLAP ACROSS CONDITIONS
# ------------------

plot_filter = (df_subspace_overlap_pc.channel == CHANNEL) & (
    df_subspace_overlap_pc.direction == "active_captured_by_passive"
)
df_plot = df_subspace_overlap_pc[plot_filter]

INNER_KWS = dict(
    box_width=15,
    whis_width=2,
    color=".8",
    solid_capstyle="round",
    solid_joinstyle="round",
)

g_violin = sns.catplot(
    data=df_plot,
    y="overlap",
    x="area",
    kind="violin",
    cut=0,
    inner="box",
    inner_kws=INNER_KWS,
    sharey=True,
    bw_adjust=1.3,
)
for ax in g_violin.axes.flat:
    ax.set_ylim(0.0, 1.0)

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_violin.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_pc_overlap_violin.pdf",
        format="pdf",
        bbox_inches="tight",
    )
else:
    plt.show()

OVERLAP_LIM = (0.0, 1.0)

g_kde = sns.displot(
    data=df_plot,
    x="overlap",
    row="area",
    kind="kde",
    bw_adjust=1.1,
    # cut=0,
    clip=OVERLAP_LIM,
)
for ax in g_kde.axes.flat:
    ax.set_xlim(OVERLAP_LIM)

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_kde.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_pc_overlap_kde.pdf",
        format="pdf",
        bbox_inches="tight",
    )
else:
    plt.show()

# %%
# ------------------
#  CROSS COVARIANCE IN LOCAL SUBSPACE OVERLAP WITHIN CONDITION
# ------------------

plot_filter = df_subspace_overlap_cross_in_local.channel == CHANNEL
df_plot = df_subspace_overlap_cross_in_local[plot_filter]

INNER_KWS = dict(
    box_width=15,
    whis_width=2,
    color=".8",
    solid_capstyle="round",
    solid_joinstyle="round",
)

g_violin = sns.catplot(
    data=df_plot,
    y="overlap",
    hue="active",
    kind="violin",
    cut=0,
    inner="box",
    inner_kws=INNER_KWS,
    sharey=True,
    bw_adjust=1.3,
)
for ax in g_violin.axes.flat:
    ax.set_ylim(0.0, 1.0)

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_violin.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_cross_covariance_in_local_subspace_violin.pdf",
        format="pdf",
        bbox_inches="tight",
    )
else:
    plt.show()

OVERLAP_LIM = (0.0, 1.0)

g_kde = sns.displot(
    data=df_plot,
    x="overlap",
    hue="active",
    kind="kde",
    bw_adjust=1.1,
    # cut=0,
    clip=OVERLAP_LIM,
)
for ax in g_kde.axes.flat:
    ax.set_xlim(OVERLAP_LIM)

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_kde.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_cross_covariance_in_local_subspace_kde.pdf",
        format="pdf",
        bbox_inches="tight",
    )
else:
    plt.show()

# %%
# ------------------
#  LOCAL VARIANCE IN CROSS SUBSPACE OVERLAP WITHIN CONDITION
# ------------------

plot_filter = df_subspace_overlap_local_in_cross.channel == CHANNEL
df_plot = df_subspace_overlap_local_in_cross[plot_filter]

INNER_KWS = dict(
    box_width=15,
    whis_width=2,
    color=".8",
    solid_capstyle="round",
    solid_joinstyle="round",
)

g_violin = sns.catplot(
    data=df_plot,
    y="overlap",
    x="area",
    hue="active",
    kind="violin",
    cut=0,
    inner="box",
    inner_kws=INNER_KWS,
    sharey=True,
    bw_adjust=1.3,
)
for ax in g_violin.axes.flat:
    ax.set_ylim(0.0, 1.0)

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_violin.figure.savefig(
        results_dir
        / f"modes_alignment_{CHANNEL}_local_variance_in_cross_subspace_violin.pdf",
        format="pdf",
        bbox_inches="tight",
    )
else:
    plt.show()

# %%
# ---------------------
#  OVERLAP SCATTERPLOTS (direction: active ↔ passive capture)
# ---------------------

from scipy import stats


def get_asterisk(p_value):
    """Convert p-value to asterisk notation."""
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""

X_DIR = "active_captured_by_passive"
Y_DIR = "passive_captured_by_active"
OVERLAP_LIM = (0.0, 1.0)
OVERLAP_SCATTER_DPI = 300
RUG_KWS = dict(height=0.025, alpha=0.5, linewidth=0.8)
OVERLAP_SCATTER_COLORS = {
    "svd": "#5AA02C",
    "pc_VISm": "#0079BC",
    "pc_SCMRN": "#F0A600",
}


def _rasterize_overlap_scatter_rugs(ax):
    for artist in ax.collections:
        artist.set_rasterized(True)
    for line in ax.lines:
        line.set_rasterized(True)


plot_filter = df_subspace_overlap_svd.channel == CHANNEL
df_svd_long = df_subspace_overlap_svd[plot_filter]

df_scatter_svd = (
    df_svd_long.pivot_table(
        index=["session_id", "image_name", "channel"],
        columns="direction",
        values="overlap",
        aggfunc="first",
    )
    .reset_index()
    .dropna(subset=[X_DIR, Y_DIR])
)
df_scatter_svd["overlap_ratio"] = df_scatter_svd[X_DIR] / df_scatter_svd[Y_DIR]
df_scatter_svd = df_scatter_svd[np.isfinite(df_scatter_svd["overlap_ratio"])]

plot_filter = df_subspace_overlap_pc.channel == CHANNEL
df_pc_long = df_subspace_overlap_pc[plot_filter]

df_scatter_pc = (
    df_pc_long.pivot_table(
        index=["session_id", "image_name", "channel", "area"],
        columns="direction",
        values="overlap",
        aggfunc="first",
    )
    .reset_index()
    .dropna(subset=[X_DIR, Y_DIR])
)
df_scatter_pc["overlap_ratio"] = df_scatter_pc[X_DIR] / df_scatter_pc[Y_DIR]
df_scatter_pc = df_scatter_pc[np.isfinite(df_scatter_pc["overlap_ratio"])]

overlap_panels = [
    ("svd", df_scatter_svd, f"SVD subspace overlap ({CHANNEL})"),
    (
        "pc_VISm",
        df_scatter_pc[df_scatter_pc["area"] == "VISm"],
        f"PC subspace overlap VISm ({CHANNEL})",
    ),
    (
        "pc_SCMRN",
        df_scatter_pc[df_scatter_pc["area"] == "SCMRN"],
        f"PC subspace overlap SCMRN ({CHANNEL})",
    ),
]

for slug, df_panel, scatter_title in overlap_panels:
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=OVERLAP_SCATTER_DPI)
    scatter_color = OVERLAP_SCATTER_COLORS[slug]
    sns.scatterplot(
        data=df_panel,
        x=X_DIR,
        y=Y_DIR,
        color=scatter_color,
        alpha=0.65,
        s=36,
        ax=ax,
        rasterized=True,
    )
    sns.rugplot(
        data=df_panel,
        x=X_DIR,
        ax=ax,
        color=scatter_color,
        **RUG_KWS,
    )
    sns.rugplot(
        data=df_panel,
        y=Y_DIR,
        ax=ax,
        color=scatter_color,
        **RUG_KWS,
    )
    _rasterize_overlap_scatter_rugs(ax)
    ax.plot([0, 1], [0, 1], ls="--", c=".4", lw=1, zorder=0)
    ax.set_xlim(OVERLAP_LIM)
    ax.set_ylim(OVERLAP_LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Active captured by passive")
    ax.set_ylabel("Passive captured by active")
    ax.set_title(scatter_title)

    if SAVE_PLOTS:
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            results_dir / f"modes_alignment_{CHANNEL}_{slug}_overlap_scatter.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=OVERLAP_SCATTER_DPI,
        )
        plt.close(fig)
    else:
        plt.show()

PANEL_ORDER = ["svd", "pc_VISm", "pc_SCMRN"]
PANEL_LABELS = {
    "svd": "SVD",
    "pc_VISm": "PC VISm",
    "pc_SCMRN": "PC SCMRN",
}

ratio_frames = []
for slug, df_panel, _ in overlap_panels:
    chunk = df_panel[["overlap_ratio"]].copy()
    chunk["panel"] = slug
    ratio_frames.append(chunk)
df_ratio = pd.concat(ratio_frames, ignore_index=True)
df_ratio["panel"] = pd.Categorical(
    df_ratio["panel"], categories=PANEL_ORDER, ordered=True
)

_ratio_ymin = float(df_ratio["overlap_ratio"].min())
_ratio_ymax = float(df_ratio["overlap_ratio"].max())
_ratio_yrange = _ratio_ymax - _ratio_ymin
RATIO_Y_STAR = _ratio_ymax + 0.04 * _ratio_yrange
RATIO_YLIM = (_ratio_ymin, RATIO_Y_STAR + 0.06 * _ratio_yrange)

g_box = sns.catplot(
    data=df_ratio,
    y="overlap_ratio",
    x="panel",
    order=PANEL_ORDER,
    kind="box",
    height=4.0,
    aspect=1.1,
    sharey=True,
)
g_box.set_axis_labels("", "Active / passive overlap ratio")
g_box.set_xticklabels([PANEL_LABELS[p] for p in PANEL_ORDER])
g_box.figure.suptitle(
    f"Overlap ratio vs symmetry (H0: mean = 1, {CHANNEL})", y=1.02
)

ax_box = g_box.ax
ax_box.axhline(1.0, ls="--", c=".4", lw=1, zorder=0)
ax_box.set_ylim(RATIO_YLIM)

for i, panel in enumerate(PANEL_ORDER):
    vals = df_ratio.loc[df_ratio["panel"] == panel, "overlap_ratio"].dropna().to_numpy()
    if vals.size < 2:
        p_value = np.nan
    else:
        _, p_value = stats.ttest_1samp(vals, popmean=1.0)
    ast = get_asterisk(p_value)
    if ast:
        ax_box.text(
            i,
            RATIO_Y_STAR,
            ast,
            ha="center",
            va="bottom",
            color="black",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )

if SAVE_PLOTS:
    results_dir.mkdir(parents=True, exist_ok=True)
    g_box.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_overlap_ratio_box.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(g_box.figure)
else:
    plt.show()

g_ratio_violin = sns.catplot(
    data=df_ratio,
    y="overlap_ratio",
    x="panel",
    order=PANEL_ORDER,
    kind="violin",
    color=".9",
    inner=None,
    cut=0,
    bw_adjust=1.1,
    height=4.0,
    aspect=1.1,
    sharey=True,
)
_n_coll_before_swarm = len(g_ratio_violin.ax.collections)
sns.swarmplot(
    data=df_ratio,
    y="overlap_ratio",
    x="panel",
    order=PANEL_ORDER,
    hue="panel",
    hue_order=PANEL_ORDER,
    palette=OVERLAP_SCATTER_COLORS,
    dodge=False,
    legend=False,
    size=3,
    ax=g_ratio_violin.ax,
    zorder=3,
)
for artist in g_ratio_violin.ax.collections[_n_coll_before_swarm:]:
    artist.set_rasterized(True)
g_ratio_violin.set_axis_labels("", "Active / passive overlap ratio")
g_ratio_violin.set_xticklabels([PANEL_LABELS[p] for p in PANEL_ORDER])
g_ratio_violin.figure.suptitle(
    f"Overlap ratio vs symmetry (H0: mean = 1, {CHANNEL})", y=1.02
)

ax_violin = g_ratio_violin.ax
ax_violin.axhline(1.0, ls="--", c=".4", lw=1, zorder=0)
ax_violin.set_ylim(RATIO_YLIM)

for i, panel in enumerate(PANEL_ORDER):
    vals = df_ratio.loc[df_ratio["panel"] == panel, "overlap_ratio"].dropna().to_numpy()
    if vals.size < 2:
        p_value = np.nan
    else:
        _, p_value = stats.ttest_1samp(vals, popmean=1.0)
    ast = get_asterisk(p_value)
    if ast:
        ax_violin.text(
            i,
            RATIO_Y_STAR,
            ast,
            ha="center",
            va="bottom",
            color="black",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )

if SAVE_PLOTS:
    g_ratio_violin.figure.savefig(
        results_dir / f"modes_alignment_{CHANNEL}_overlap_ratio_violin_swarm.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(g_ratio_violin.figure)
else:
    plt.show()

# %%
# ---------------------
#  OVERLAP SCATTERPLOTS (active vs passive overlap)
# ---------------------

OVERLAP_ACTIVE = True
OVERLAP_PASSIVE = False
OVERLAP_SCATTER_DPI = 300
OVERLAP_SCATTER_COLORS = {
    "cross_in_local": "#6F1D1B",
    "local_in_cross_VISm": "#34A0A4",
    "local_in_cross_SCMRN": "#C36F09",
}


def _rasterize_overlap_scatter_rugs(ax):
    for artist in ax.collections:
        artist.set_rasterized(True)
    for line in ax.lines:
        line.set_rasterized(True)


plot_filter = df_subspace_overlap_cross_in_local.channel == CHANNEL
df_cross_local_long = df_subspace_overlap_cross_in_local[plot_filter]

df_scatter_cross_local = (
    df_cross_local_long.pivot_table(
        index=["session_id", "image_name", "channel"],
        columns="active",
        values="overlap",
        aggfunc="first",
    )
    .reset_index()
    .dropna(subset=[OVERLAP_ACTIVE, OVERLAP_PASSIVE])
)
df_scatter_cross_local["overlap_ratio"] = (
    df_scatter_cross_local[OVERLAP_ACTIVE] / df_scatter_cross_local[OVERLAP_PASSIVE]
)
df_scatter_cross_local = df_scatter_cross_local[
    np.isfinite(df_scatter_cross_local["overlap_ratio"])
]

plot_filter = df_subspace_overlap_local_in_cross.channel == CHANNEL
df_local_cross_long = df_subspace_overlap_local_in_cross[plot_filter]

df_scatter_local_cross = (
    df_local_cross_long.pivot_table(
        index=["session_id", "image_name", "channel", "area"],
        columns="active",
        values="overlap",
        aggfunc="first",
    )
    .reset_index()
    .dropna(subset=[OVERLAP_ACTIVE, OVERLAP_PASSIVE])
)
df_scatter_local_cross["overlap_ratio"] = (
    df_scatter_local_cross[OVERLAP_ACTIVE] / df_scatter_local_cross[OVERLAP_PASSIVE]
)
df_scatter_local_cross = df_scatter_local_cross[
    np.isfinite(df_scatter_local_cross["overlap_ratio"])
]

active_overlap_panels = [
    (
        "cross_in_local",
        df_scatter_cross_local,
        f"Cross cov. in local subspace ({CHANNEL})",
    ),
    (
        "local_in_cross_VISm",
        df_scatter_local_cross[df_scatter_local_cross["area"] == "VISm"],
        f"Local var. in cross subspace VISm ({CHANNEL})",
    ),
    (
        "local_in_cross_SCMRN",
        df_scatter_local_cross[df_scatter_local_cross["area"] == "SCMRN"],
        f"Local var. in cross subspace SCMRN ({CHANNEL})",
    ),
]

for slug, df_panel, scatter_title in active_overlap_panels:
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=OVERLAP_SCATTER_DPI)
    scatter_color = OVERLAP_SCATTER_COLORS[slug]
    sns.scatterplot(
        data=df_panel,
        x=OVERLAP_ACTIVE,
        y=OVERLAP_PASSIVE,
        color=scatter_color,
        alpha=0.65,
        s=36,
        ax=ax,
        rasterized=True,
    )
    sns.rugplot(
        data=df_panel,
        x=OVERLAP_ACTIVE,
        ax=ax,
        color=scatter_color,
        **RUG_KWS,
    )
    sns.rugplot(
        data=df_panel,
        y=OVERLAP_PASSIVE,
        ax=ax,
        color=scatter_color,
        **RUG_KWS,
    )
    _rasterize_overlap_scatter_rugs(ax)
    ax.plot([0, 1], [0, 1], ls="--", c=".4", lw=1, zorder=0)
    ax.set_xlim(OVERLAP_LIM)
    ax.set_ylim(OVERLAP_LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Overlap (active)")
    ax.set_ylabel("Overlap (passive)")
    ax.set_title(scatter_title)

    if SAVE_PLOTS:
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            results_dir
            / f"modes_alignment_{CHANNEL}_{slug}_active_overlap_scatter.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=OVERLAP_SCATTER_DPI,
        )
        plt.close(fig)
    else:
        plt.show()

ACTIVE_PANEL_ORDER = [
    "cross_in_local",
    "local_in_cross_VISm",
    "local_in_cross_SCMRN",
]
ACTIVE_PANEL_LABELS = {
    "cross_in_local": "Cross in local",
    "local_in_cross_VISm": "Local in cross VISm",
    "local_in_cross_SCMRN": "Local in cross SCMRN",
}

ratio_frames = []
for slug, df_panel, _ in active_overlap_panels:
    chunk = df_panel[["overlap_ratio"]].copy()
    chunk["panel"] = slug
    ratio_frames.append(chunk)
df_ratio_active = pd.concat(ratio_frames, ignore_index=True)
df_ratio_active["panel"] = pd.Categorical(
    df_ratio_active["panel"], categories=ACTIVE_PANEL_ORDER, ordered=True
)

g_box = sns.catplot(
    data=df_ratio_active,
    y="overlap_ratio",
    x="panel",
    order=ACTIVE_PANEL_ORDER,
    kind="box",
    height=4.0,
    aspect=1.1,
    sharey=True,
)
g_box.set_axis_labels("", "Active / passive overlap ratio")
g_box.set_xticklabels([ACTIVE_PANEL_LABELS[p] for p in ACTIVE_PANEL_ORDER])
g_box.figure.suptitle(
    f"Active overlap ratio vs symmetry (H0: mean = 1, {CHANNEL})", y=1.02
)

ax_box = g_box.ax
ax_box.axhline(1.0, ls="--", c=".4", lw=1, zorder=0)
ax_box.set_ylim(RATIO_YLIM)

for i, panel in enumerate(ACTIVE_PANEL_ORDER):
    vals = df_ratio_active.loc[
        df_ratio_active["panel"] == panel, "overlap_ratio"
    ].dropna().to_numpy()
    if vals.size < 2:
        p_value = np.nan
    else:
        _, p_value = stats.ttest_1samp(vals, popmean=1.0)
    ast = get_asterisk(p_value)
    if ast:
        ax_box.text(
            i,
            RATIO_Y_STAR,
            ast,
            ha="center",
            va="bottom",
            color="black",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )

if SAVE_PLOTS:
    g_box.figure.savefig(
        results_dir
        / f"modes_alignment_{CHANNEL}_active_overlap_ratio_box.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(g_box.figure)
else:
    plt.show()

g_ratio_violin = sns.catplot(
    data=df_ratio_active,
    y="overlap_ratio",
    x="panel",
    order=ACTIVE_PANEL_ORDER,
    kind="violin",
    color=".9",
    inner=None,
    cut=0,
    bw_adjust=1.1,
    height=4.0,
    aspect=1.1,
    sharey=True,
)
_n_coll_before_swarm = len(g_ratio_violin.ax.collections)
sns.swarmplot(
    data=df_ratio_active,
    y="overlap_ratio",
    x="panel",
    order=ACTIVE_PANEL_ORDER,
    hue="panel",
    hue_order=ACTIVE_PANEL_ORDER,
    palette=OVERLAP_SCATTER_COLORS,
    dodge=False,
    legend=False,
    size=3,
    ax=g_ratio_violin.ax,
    zorder=3,
)
for artist in g_ratio_violin.ax.collections[_n_coll_before_swarm:]:
    artist.set_rasterized(True)
g_ratio_violin.set_axis_labels("", "Active / passive overlap ratio")
g_ratio_violin.set_xticklabels([ACTIVE_PANEL_LABELS[p] for p in ACTIVE_PANEL_ORDER])
g_ratio_violin.figure.suptitle(
    f"Active overlap ratio vs symmetry (H0: mean = 1, {CHANNEL})", y=1.02
)

ax_violin = g_ratio_violin.ax
ax_violin.axhline(1.0, ls="--", c=".4", lw=1, zorder=0)
ax_violin.set_ylim(RATIO_YLIM)

for i, panel in enumerate(ACTIVE_PANEL_ORDER):
    vals = df_ratio_active.loc[
        df_ratio_active["panel"] == panel, "overlap_ratio"
    ].dropna().to_numpy()
    if vals.size < 2:
        p_value = np.nan
    else:
        _, p_value = stats.ttest_1samp(vals, popmean=1.0)
    ast = get_asterisk(p_value)
    if ast:
        ax_violin.text(
            i,
            RATIO_Y_STAR,
            ast,
            ha="center",
            va="bottom",
            color="black",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )

if SAVE_PLOTS:
    g_ratio_violin.figure.savefig(
        results_dir
        / f"modes_alignment_{CHANNEL}_active_overlap_ratio_violin_swarm.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(g_ratio_violin.figure)
else:
    plt.show()

# %%
