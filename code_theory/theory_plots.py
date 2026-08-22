# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE_DIR = os.getcwd()

# Project-level results/ (sibling of tempt/)
_PROJECT_ROOT = os.path.dirname(_BASE_DIR) if os.path.basename(_BASE_DIR) == "tempt" else _BASE_DIR
output_dir = os.path.join(_PROJECT_ROOT, "results")

# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N = 1000
M = 1000
P = 5  # connectivity realizations per sweep point

D0 = 1.0
sigma_D = 1.0 / np.sqrt(2.0)
Delta0 = sigma_D**2
E = 0.0

G_VALS = [0.3, 0.5, 0.7]
GAMMA_S_VALS = np.linspace(1.0, float(M), 30)

# Mean source eigenvalue Tr(Sigma_s)/M; total variance = mean * M
TR_OVER_M_VALS = np.linspace(0.0, 2.0, 30)
V_SCALE_VALS = [0.1, 1.0, 5.0]  # V = scale / M
G_FIXED_V_SWEEPS = 0.5
G_FIXED_GAMMA_SWEEP = 0.6
GAMMA_S_FIXED = 0.2 * M
TR_OVER_M_FIXED = 1.0

PALETTE_G = {
    0.3: "#2E5F8A",
    0.5: "#378151",
    0.7: "#B5651D",
}

# PALETTE_V = {
#     5.0: "#880D1E",  # highest V
#     1.0: "#F26A8D",
#     0.1: "#CBEEF3",  # lowest V
# }

PALETTE_V = {
    5.0: "#CE4257",  # highest V
    1.0: "#F26A8D",
    0.1: "#880D1E",  # lowest V
}

# %%
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalized_dimensionality(C):
    """Return D(C) = DPR(C) / N for a covariance matrix C."""
    cii = np.diag(C)
    cij = C[~np.eye(C.shape[0], dtype=bool)]
    numerator = np.sum(cii) ** 2
    denominator = np.sum(cii**2) + np.sum(cij**2)
    if denominator == 0.0:
        return 0.0
    return numerator / N / denominator


def unnormalized_dimensionality(C):
    """Return N * D(C), the unnormalized target participation ratio."""
    return normalized_dimensionality(C) * N


def draw_lognormal_source_eigenvalues(xi_e, gamma_s, m):
    """Draw source eigenvalues with Tr(Sigma_s) = xi_e and Gamma_s = gamma_s."""
    if xi_e <= 0.0:
        return np.zeros(m)
    # Cap Gamma_s at M for a valid lognormal spectrum
    gamma_s = min(float(gamma_s), float(m) * (1.0 - 1e-12))
    mu_lambda = np.log(xi_e / m) - 0.5 * np.log(m / gamma_s)
    sigma_lambda = np.sqrt(np.log(m / gamma_s))
    return np.random.lognormal(mu_lambda, sigma_lambda, m)


def v_label(v_scale):
    if v_scale == 1.0:
        return r"$V=1/M$"
    if v_scale == 0.1:
        return r"$V=0.1/M$"
    return rf"$V={v_scale:g}/M$"


def make_square_axes():
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.set_box_aspect(1)
    ax.tick_params(axis="both", which="both", direction="out", length=4, width=0.8, bottom=True, left=True)
    return fig, ax


def save_figure(fig, filename):
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, filename)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.show()
    return pdf_path


# %%
# ---------------------------------------------------------------------------
# Analytical formulas (Eqs. 47 and 53)
# ---------------------------------------------------------------------------

def mean_cii_theory(g, tr_sigma_s, v, d0=D0):
    """Mean diagonal covariance from Eq. (47): <C_ii> = (D0 + Tr(Sigma_s) V) / (1 - g^2)."""
    return (d0 + tr_sigma_s * v) / (1.0 - g**2)


def eq53_dimensionality(g, gamma_s, d0=D0, delta0=Delta0, n=N, tr_sigma_s=None, v=None):
    """Normalized target dimensionality D(C) from Eq. (53)."""
    if tr_sigma_s is None:
        tr_sigma_s = float(M)
    if v is None:
        v = 1.0 / M
    mean_scale = d0 + tr_sigma_s * v
    hetero = delta0 + n * (v**2) * (tr_sigma_s**2) / gamma_s
    return (1.0 - g**2) ** 2 * mean_scale**2 / (mean_scale**2 + (1.0 - g**2) ** 2 * hetero)


# %%
# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class LinearAfferentSimulation:
    """Linear model simulation corresponding to Eqs. (47) and (53)."""

    def __init__(self, n, g, m, xi_e, gamma_s, mu_d=D0, sigma_d=sigma_D, e=E, v=None):
        if v is None:
            v = 1.0 / m
        self.n = n
        self.g = g
        self.m = m
        self.xi_e = xi_e
        self.gamma_s = gamma_s

        self.w = np.random.normal(0.0, g / np.sqrt(n), (n, n))

        mean = np.log(mu_d**2 / np.sqrt(mu_d**2 + sigma_d**2))
        var = np.log(1.0 + (sigma_d**2 / mu_d**2))
        self.d_intrinsic = np.diag(np.random.lognormal(mean, np.sqrt(var), n))

        lambdas = draw_lognormal_source_eigenvalues(xi_e, gamma_s, m)
        self.d_source = np.diag(lambdas)
        self.lambda_mat = np.random.normal(e, np.sqrt(v), (n, m))

    def covariance_matrix(self):
        r_mat = np.linalg.inv(np.eye(self.n) - self.w)
        total_noise = self.d_intrinsic + self.lambda_mat @ self.d_source @ self.lambda_mat.T
        return r_mat @ total_noise @ r_mat.T


def run_gamma_sweep(g_vals, gamma_s_vals, tr_over_m, v_scale, p=P, seed=9):
    """Sweep Gamma_s across g values at fixed Tr(Sigma_s)/M and V = v_scale/M."""
    np.random.seed(seed)
    tr_sigma_s = tr_over_m * M
    v = v_scale / M

    rows = []
    for g in g_vals:
        for gamma_s in gamma_s_vals:
            for _ in range(p):
                sim = LinearAfferentSimulation(
                    n=N, m=M, g=g, xi_e=tr_sigma_s, gamma_s=gamma_s, v=v,
                )
                c = sim.covariance_matrix()
                rows.append({
                    "g": g,
                    "Gamma_s": gamma_s,
                    "tr_over_m": tr_over_m,
                    "v_scale": v_scale,
                    "target_dim": unnormalized_dimensionality(c),
                    "mean_cii": float(np.mean(np.diag(c))),
                    "theory": False,
                })

            rows.append({
                "g": g,
                "Gamma_s": gamma_s,
                "tr_over_m": tr_over_m,
                "v_scale": v_scale,
                "target_dim": eq53_dimensionality(g, gamma_s, tr_sigma_s=tr_sigma_s, v=v) * N,
                "mean_cii": mean_cii_theory(g, tr_sigma_s, v),
                "theory": True,
            })

    return pd.DataFrame(rows)


def run_tr_sweep(tr_over_m_vals, v_scales, g, gamma_s, p=P, seed=11):
    """Sweep Tr(Sigma_s)/M across V scales at fixed g and Gamma_s."""
    np.random.seed(seed)

    rows = []
    for v_scale in v_scales:
        v = v_scale / M
        for tr_over_m in tr_over_m_vals:
            tr_sigma_s = tr_over_m * M
            for _ in range(p):
                sim = LinearAfferentSimulation(
                    n=N, m=M, g=g, xi_e=tr_sigma_s, gamma_s=gamma_s, v=v,
                )
                c = sim.covariance_matrix()
                rows.append({
                    "g": g,
                    "Gamma_s": gamma_s,
                    "tr_over_m": tr_over_m,
                    "v_scale": v_scale,
                    "target_dim": unnormalized_dimensionality(c),
                    "mean_cii": float(np.mean(np.diag(c))),
                    "theory": False,
                })

            rows.append({
                "g": g,
                "Gamma_s": gamma_s,
                "tr_over_m": tr_over_m,
                "v_scale": v_scale,
                "target_dim": eq53_dimensionality(g, gamma_s, tr_sigma_s=tr_sigma_s, v=v) * N,
                "mean_cii": mean_cii_theory(g, tr_sigma_s, v),
                "theory": True,
            })

    return pd.DataFrame(rows)


def run_gamma_vs_v_sweep(gamma_s_vals, v_scales, g, tr_over_m, p=P, seed=13):
    """Sweep Gamma_s across V scales at fixed g and Tr(Sigma_s)/M."""
    np.random.seed(seed)
    tr_sigma_s = tr_over_m * M

    rows = []
    for v_scale in v_scales:
        v = v_scale / M
        for gamma_s in gamma_s_vals:
            for _ in range(p):
                sim = LinearAfferentSimulation(
                    n=N, m=M, g=g, xi_e=tr_sigma_s, gamma_s=gamma_s, v=v,
                )
                c = sim.covariance_matrix()
                rows.append({
                    "g": g,
                    "Gamma_s": gamma_s,
                    "tr_over_m": tr_over_m,
                    "v_scale": v_scale,
                    "target_dim": unnormalized_dimensionality(c),
                    "mean_cii": float(np.mean(np.diag(c))),
                    "theory": False,
                })

            rows.append({
                "g": g,
                "Gamma_s": gamma_s,
                "tr_over_m": tr_over_m,
                "v_scale": v_scale,
                "target_dim": eq53_dimensionality(g, gamma_s, tr_sigma_s=tr_sigma_s, v=v) * N,
                "mean_cii": mean_cii_theory(g, tr_sigma_s, v),
                "theory": True,
            })

    return pd.DataFrame(rows)


# %%
# ---------------------------------------------------------------------------
# Run sweeps
# ---------------------------------------------------------------------------
np.random.seed(9)

df_dim_vs_gamma_g = run_gamma_sweep(
    g_vals=G_VALS,
    gamma_s_vals=GAMMA_S_VALS,
    tr_over_m=TR_OVER_M_FIXED,
    v_scale=1.0,
)

df_vs_tr = run_tr_sweep(
    tr_over_m_vals=TR_OVER_M_VALS,
    v_scales=V_SCALE_VALS,
    g=G_FIXED_V_SWEEPS,
    gamma_s=GAMMA_S_FIXED,
)

df_dim_vs_gamma_v = run_gamma_vs_v_sweep(
    gamma_s_vals=GAMMA_S_VALS,
    v_scales=V_SCALE_VALS,
    g=G_FIXED_GAMMA_SWEEP,
    tr_over_m=TR_OVER_M_FIXED,
)

# %%
# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_curves(
    df,
    *,
    x,
    y,
    hue_key,
    hue_vals,
    palette,
    hue_formatter,
    xlabel,
    ylabel,
    title,
    filename,
    xlim=None,
    yscale="linear",
):
    fig, ax = make_square_axes()

    for hv in hue_vals:
        color = palette[hv]
        label_base = hue_formatter(hv)
        theory = df[(df["theory"]) & (df[hue_key] == hv)]
        sim = df[(~df["theory"]) & (df[hue_key] == hv)]

        sns.lineplot(
            data=theory,
            x=x,
            y=y,
            ax=ax,
            color=color,
            linewidth=2.0,
            label=f"{label_base} (theory)",
            legend=False,
        )
        sns.scatterplot(
            data=sim,
            x=x,
            y=y,
            ax=ax,
            color=color,
            s=22,
            alpha=0.35,
            edgecolor=None,
            label=f"{label_base} (sim)",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)

    y_vals = df[y].to_numpy()
    y_vals = y_vals[np.isfinite(y_vals)]
    if yscale == "log":
        ax.set_yscale("log")
        positive = y_vals[y_vals > 0]
        y_min = float(np.nanmin(positive)) if positive.size else 1e-3
        y_max = float(np.nanmax(y_vals)) if y_vals.size else 1.0
        ax.set_ylim(y_min / 1.2, y_max * 1.2)
        ax.grid(alpha=0.25, which="both")
    else:
        y_min = float(np.nanmin(y_vals)) if y_vals.size else 0.0
        y_max = float(np.nanmax(y_vals)) if y_vals.size else 1.0
        pad = 0.05 * (y_max - y_min if y_max > y_min else max(abs(y_max), 1.0))
        ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
        ax.grid(alpha=0.25)

    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, filename)


plot_curves(
    df_dim_vs_gamma_g,
    x="Gamma_s",
    y="target_dim",
    hue_key="g",
    hue_vals=G_VALS,
    palette=PALETTE_G,
    hue_formatter=lambda g: rf"$g_*={g}$",
    xlabel=r"Source dimensionality $\Gamma_s$",
    ylabel=r"Target dimensionality $N\,D(C)$",
    title=rf"Eq. (53): $N D(C)$ vs $\Gamma_s$ ($\mathrm{{Tr}}(\Sigma_s)/M={TR_OVER_M_FIXED}$, $V=1/M$)",
    filename="eq53_target_vs_source_dimensionality.pdf",
    xlim=(0.0, float(M)),
)

plot_curves(
    df_vs_tr,
    x="tr_over_m",
    y="mean_cii",
    hue_key="v_scale",
    hue_vals=V_SCALE_VALS,
    palette=PALETTE_V,
    hue_formatter=v_label,
    xlabel=r"Mean source variance $\mathrm{Tr}(\Sigma_s)/M$",
    ylabel=r"Mean diagonal covariance $\langle C_{ii}\rangle$",
    title=rf"Eq. (47): $\langle C_{{ii}}\rangle$ vs $\mathrm{{Tr}}(\Sigma_s)/M$ ($g_*={G_FIXED_V_SWEEPS}$)",
    filename="eq47_mean_cii_vs_tr.pdf",
    xlim=(0.0, 2.0),
)

plot_curves(
    df_vs_tr,
    x="tr_over_m",
    y="mean_cii",
    hue_key="v_scale",
    hue_vals=V_SCALE_VALS,
    palette=PALETTE_V,
    hue_formatter=v_label,
    xlabel=r"Mean source variance $\mathrm{Tr}(\Sigma_s)/M$",
    ylabel=r"Mean diagonal covariance $\langle C_{ii}\rangle$ (log)",
    title=rf"Eq. (47): $\langle C_{{ii}}\rangle$ vs $\mathrm{{Tr}}(\Sigma_s)/M$ (log-y, $g_*={G_FIXED_V_SWEEPS}$)",
    filename="eq47_mean_cii_vs_tr_log.pdf",
    xlim=(0.0, 2.0),
    yscale="log",
)

plot_curves(
    df_vs_tr,
    x="tr_over_m",
    y="target_dim",
    hue_key="v_scale",
    hue_vals=V_SCALE_VALS,
    palette=PALETTE_V,
    hue_formatter=v_label,
    xlabel=r"Mean source variance $\mathrm{Tr}(\Sigma_s)/M$",
    ylabel=r"Target dimensionality $N\,D(C)$",
    title=rf"Eq. (53): $N D(C)$ vs $\mathrm{{Tr}}(\Sigma_s)/M$ ($g_*={G_FIXED_V_SWEEPS}$, $\Gamma_s=0.2M$)",
    filename="eq53_target_vs_tr.pdf",
    xlim=(0.0, 2.0),
)

plot_curves(
    df_dim_vs_gamma_v,
    x="Gamma_s",
    y="target_dim",
    hue_key="v_scale",
    hue_vals=V_SCALE_VALS,
    palette=PALETTE_V,
    hue_formatter=v_label,
    xlabel=r"Source dimensionality $\Gamma_s$",
    ylabel=r"Target dimensionality $N\,D(C)$",
    title=rf"Eq. (53): $N D(C)$ vs $\Gamma_s$ ($g_*={G_FIXED_GAMMA_SWEEP}$, $\mathrm{{Tr}}(\Sigma_s)/M={TR_OVER_M_FIXED}$)",
    filename="eq53_target_vs_gamma_by_V.pdf",
    xlim=(0.0, float(M)),
)
# %%
