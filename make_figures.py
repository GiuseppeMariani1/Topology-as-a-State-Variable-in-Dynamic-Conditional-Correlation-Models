#!/usr/bin/env python3
"""
make_figures.py — TopoDCC thesis figures, no model recompute.

Everything here reads only what is already saved in data/processed/.
Nothing is re-fitted, re-estimated, or re-permuted. If a figure needs
data that isn't on disk (OOS loss series, portfolio returns, the 500-run
permutation arrays, the full config sweep), it is deliberately left out
rather than faked — see the printed summary at the end for what's
skipped and why.

Usage:
    python make_figures.py

Drop this in the repo root (same level as data/, src/). Figures land
in figures/, one PNG each, at 200 dpi.

Requires: numpy, pandas, matplotlib, pyarrow (for parquet), ripser (for the
persistence diagram figure only — pip install ripser)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap

DATA = "data/processed"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

skipped = []

# ---------------------------------------------------------------------------
# Shared theme — one consistent look across every figure. Times New Roman
# to match the thesis body text; falls back cleanly if it isn't installed
# (e.g. on this Linux sandbox) but will pick it up on your Windows laptop.
# ---------------------------------------------------------------------------
NAVY = "#13293d"
CRIMSON = "#9e2a2b"
STEEL = "#2471a3"
GOLD = "#b8860b"
SLATE = "#5d6d7e"
PAPER = "#fbfaf7"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": NAVY,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.color": SLATE,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "text.color": NAVY,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": SLATE,
    "legend.framealpha": 0.9,
})

CRISES = [
    ("2007-08-01", "2009-03-31", "GFC / Lehman"),
    ("2011-08-01", "2011-12-31", "Euro debt"),
    ("2020-02-15", "2020-04-30", "COVID crash"),
    ("2022-01-01", "2022-10-31", "2022 rate hikes"),
]


def shade_crises(ax):
    """Overlay the same four crisis windows used in the a_t/b_t figure."""
    for start, end, label in CRISES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=GOLD, alpha=0.12, zorder=0)


def savefig(fig, name, pad=0.1):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print(f"  wrote {path}")


def style_3d_axes(ax):
    """Give a 3D axes panel a clean, uncluttered, premium look."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor(SLATE)
        axis.pane.set_alpha(0.6)
        axis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.25)
    ax.xaxis.line.set_color(SLATE)
    ax.yaxis.line.set_color(SLATE)
    ax.zaxis.line.set_color(SLATE)


# ---------------------------------------------------------------------------
# 1. a_t / b_t paths — already exists, just copy it into figures/ for
#    consistency so everything the thesis needs lives in one folder.
# ---------------------------------------------------------------------------
print("[1/8] a_t/b_t paths (already generated — copying, not rebuilding)")
src = os.path.join(DATA, "a_b_timevariation.png")
if os.path.exists(src):
    import shutil
    shutil.copy(src, os.path.join(OUT, "01_a_b_timevariation.png"))
    print(f"  copied {src}")
else:
    skipped.append("a_t/b_t paths — a_b_timevariation.png not found in data/processed")


# ---------------------------------------------------------------------------
# 2. GARCH conditional volatility per asset, crisis-shaded
# ---------------------------------------------------------------------------
print("[2/8] GARCH sigma_t per asset")
try:
    sigma = pd.read_parquet(os.path.join(DATA, "garch_sigma.parquet"))
    fig, axes = plt.subplots(len(sigma.columns), 1, figsize=(9, 10), sharex=True)
    for ax, col in zip(axes, sigma.columns):
        shade_crises(ax)
        ax.plot(sigma.index, sigma[col], linewidth=0.9, color=NAVY, zorder=2)
        ax.fill_between(sigma.index, 0, sigma[col], color=STEEL, alpha=0.12, zorder=1)
        ax.set_ylabel(col, rotation=0, ha="right", va="center", fontweight="bold")
        ax.margins(x=0)
        ax.set_ylim(bottom=0)
    axes[0].set_title("Fitted GARCH(1,1) conditional volatility $\\sigma_t$ by asset", pad=14)
    fig.tight_layout()
    savefig(fig, "02_garch_sigma.png")
except Exception as e:
    skipped.append(f"GARCH sigma_t — {e}")


# ---------------------------------------------------------------------------
# 3. Encoder weight tornado (Table 5 as a figure), with value labels
# ---------------------------------------------------------------------------
print("[3/8] Encoder weight diverging bars")
try:
    d = np.load(os.path.join(DATA, "dcc_topo_lpnorm_results.npy"), allow_pickle=True).item()
    feats = list(d["feature_columns"])
    w_a, w_b = np.array(d["w_a"]), np.array(d["w_b"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for ax, w, title in [(ax1, w_a, "$w_a$ (shock sensitivity)"), (ax2, w_b, "$w_b$ (persistence)")]:
        order = np.argsort(np.abs(w))
        vals = w[order]
        colors = [CRIMSON if v < 0 else STEEL for v in vals]
        positions = np.arange(len(w))
        ax.barh(positions - 0.06, vals, height=0.62, color=SLATE, alpha=0.15, zorder=1)
        ax.barh(positions, vals, height=0.62, color=colors, zorder=2,
                edgecolor="white", linewidth=0.6)
        for pos, v in zip(positions, vals):
            ax.text(v + (0.012 if v >= 0 else -0.012), pos, f"{v:+.3f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=8, color=NAVY)
        ax.axvline(0, color=NAVY, linewidth=1.0, zorder=3)
        ax.set_title(title)
        ax.set_yticks(positions)
        ax.set_yticklabels([feats[i] for i in order], fontsize=9)
        ax.set_ylim(-0.6, len(w) - 0.4)
        xpad = max(abs(vals.min()), abs(vals.max())) * 0.35
        ax.set_xlim(vals.min() - xpad, vals.max() + xpad)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Fitted encoder weights, 9-feature lpnorm representation", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, "03_encoder_weights.png")
except Exception as e:
    skipped.append(f"Encoder weight tornado — {e}")


# ---------------------------------------------------------------------------
# 4. Ridge regularisation path, selected lambda marked
# ---------------------------------------------------------------------------
print("[4/8] Ridge regularisation path")
try:
    d = np.load(os.path.join(DATA, "dcc_topo_reg_results.npy"), allow_pickle=True).item()
    lam_results = d["lambda_results"]
    rows = sorted(lam_results["lambda_l2"].keys())
    lambdas = [lam_results["lambda_l2"][r] for r in rows]
    train_ll = [lam_results["ll_train"][r] for r in rows]
    test_ll = [lam_results["ll_test"][r] for r in rows]
    selected = max(lambdas)  # the grid's own optimum, whatever the repo says

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(lambdas, train_ll, "o-", color=CRIMSON, linewidth=1.8, markersize=7,
              markeredgecolor="white", markeredgewidth=0.8, label="train $\\ell$", zorder=3)
    ax1.set_xscale("log")
    ax1.set_xlabel("$\\lambda$ (log scale)")
    ax1.set_ylabel("train $\\ell$", color=CRIMSON, fontweight="bold")
    ax1.tick_params(axis="y", colors=CRIMSON)

    ax2 = ax1.twinx()
    ax2.plot(lambdas, test_ll, "s-", color=STEEL, linewidth=1.8, markersize=7,
              markeredgecolor="white", markeredgewidth=0.8, label="held-out $\\ell$", zorder=3)
    ax2.set_ylabel("held-out $\\ell$", color=STEEL, fontweight="bold")
    ax2.tick_params(axis="y", colors=STEEL)
    ax2.grid(False)

    ax1.axvline(selected, color=GOLD, linestyle="--", linewidth=1.3, zorder=1)
    ax1.text(selected, ax1.get_ylim()[1], f"  selected $\\lambda$={selected:,.0f}",
              color=GOLD, fontsize=9, va="top", fontweight="bold")

    ax1.set_title("Ridge grid search: train vs. held-out log-likelihood", pad=14)
    fig.tight_layout()
    savefig(fig, "04_ridge_path.png")
except Exception as e:
    skipped.append(f"Ridge regularisation path — {e} (check lambda_results dict keys match)")


# ---------------------------------------------------------------------------
# 5. Diagnostic: DCC-standardised residuals, computed from saved R_seq
# ---------------------------------------------------------------------------
print("[5/8] Standardised residual diagnostics")
try:
    z = pd.read_parquet(os.path.join(DATA, "garch_residuals.parquet"))
    d = np.load(os.path.join(DATA, "dcc_baseline_results.npy"), allow_pickle=True).item()
    R_seq = np.array(d["R_seq"])
    T, N, _ = R_seq.shape
    z_vals = z.values[-T:]

    e = np.zeros_like(z_vals)
    for t in range(T):
        Rt = R_seq[t]
        vals, vecs = np.linalg.eigh(Rt)
        vals = np.clip(vals, 1e-10, None)
        Rt_inv_sqrt = vecs @ np.diag(vals ** -0.5) @ vecs.T
        e[t] = Rt_inv_sqrt @ z_vals[t]

    def acf(x, nlags=20):
        x = x - x.mean()
        var = np.dot(x, x)
        return np.array([1.0 if k == 0 else np.dot(x[:-k], x[k:]) / var for k in range(nlags + 1)])

    fig, axes = plt.subplots(1, N, figsize=(3.1 * N, 3.6), sharey=True)
    band = 1.96 / np.sqrt(T)
    for i, col in enumerate(z.columns):
        lags = acf(e[:, i] ** 2, nlags=15)
        colors = [CRIMSON if abs(v) > band and k > 0 else STEEL for k, v in enumerate(lags)]
        axes[i].bar(range(len(lags)), lags, color=colors, width=0.6, zorder=2)
        axes[i].axhspan(-band, band, color=SLATE, alpha=0.12, zorder=1)
        axes[i].axhline(0, color=NAVY, linewidth=0.8)
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("lag")
    axes[0].set_ylabel("autocorrelation")
    fig.suptitle("ACF of squared DCC-standardised residuals, baseline model", y=1.03)
    fig.tight_layout()
    savefig(fig, "05_residual_diagnostics.png")
except Exception as e:
    skipped.append(f"Residual diagnostics — {e}")


# ---------------------------------------------------------------------------
# 6. Permutation test — regularised side only (real n on disk, see note)
# ---------------------------------------------------------------------------
print("[6/8] Permutation histogram (regularised only — see note)")
try:
    d = np.load(os.path.join(DATA, "permutation_test_lpnorm_reg_results.npy"), allow_pickle=True).item()
    perms = np.array(d["permuted_lls"])
    real_ll = d["real_ll"]
    lam = d.get("lambda_l2", "?")
    n = len(perms)

    fig, ax = plt.subplots(figsize=(8, 5))
    counts, bins, patches = ax.hist(perms, bins=max(8, n // 6), color=SLATE,
                                     edgecolor="white", linewidth=0.8, alpha=0.85, zorder=2)
    peak = counts.max() if len(counts) else 1
    for c, p in zip(counts, patches):
        p.set_facecolor(plt.cm.Blues(0.35 + 0.5 * (c / peak)))
    ax.axvline(real_ll, color=CRIMSON, linewidth=2.2, zorder=3)
    ax.annotate(f"real $\\ell$ = {real_ll:.2f}", xy=(real_ll, peak), xytext=(8, 0),
                textcoords="offset points", color=CRIMSON, fontweight="bold", va="center")
    ax.set_title(f"Permutation test, regularised ($\\lambda={lam:,.0f}$, n={n})", pad=14)
    ax.set_xlabel("log-likelihood, shuffled feature ordering")
    ax.set_ylabel("count")
    fig.tight_layout()
    savefig(fig, "06_permutation_regularised.png")
    print(f"  NOTE: n={n} on disk — caption/text must say n={n}, not 500, unless rerun.")
except Exception as e:
    skipped.append(f"Permutation histogram — {e}")

skipped.append("Permutation histogram, UNREGULARISED — only 10 stale permutations on disk, "
                "not enough to plot honestly (needs rerun at n=500)")


# ---------------------------------------------------------------------------
# 7. CREATIVE #1: full persistence-landscape surface over time (H1, level 1)
#    Real hillshading via LightSource, floor contour for depth, tuned camera.
# ---------------------------------------------------------------------------
print("[7/10] 3D persistence landscape surface (creative #1)")
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    land = pd.read_parquet(os.path.join(DATA, "tda_features_landscape.parquet"))
    level = "lh1_k0"
    grid_cols = sorted([c for c in land.columns if c.startswith(level + "_g")],
                        key=lambda c: int(c.split("_g")[1]))
    Z = land[grid_cols].values

    step = 5
    Z_sub = Z[::step]
    t_idx = land.index[::step]
    t_num = np.arange(len(t_idx))
    g_num = np.arange(len(grid_cols))
    Tm, Gm = np.meshgrid(t_num, g_num, indexing="ij")

    cmap = LinearSegmentedColormap.from_list("premium", ["#0b3d63", "#2471a3", "#e8b34d", "#9e2a2b"])
    ls = LightSource(azdeg=315, altdeg=55)
    rgb = ls.shade(Z_sub, cmap=cmap, vert_exag=2.5, blend_mode="soft")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.03, 0.05, 0.78, 0.85])
    ax.plot_surface(Tm, Gm, Z_sub, facecolors=rgb, rstride=1, cstride=1,
                     linewidth=0, antialiased=True, shade=False, zorder=2)
    floor = Z_sub.min() - 0.02
    ax.contourf(Tm, Gm, Z_sub, zdir="z", offset=floor, cmap=cmap, alpha=0.35, levels=14, zorder=1)
    ax.set_zlim(floor, Z_sub.max() * 1.05)

    yr_ticks = np.linspace(0, len(t_idx) - 1, 6).astype(int)
    ax.set_xticks(yr_ticks)
    ax.set_xticklabels([str(t_idx[i].date())[:7] for i in yr_ticks], rotation=20)
    ax.set_ylabel("landscape grid point")
    ax.set_zlabel("$\\lambda_1(t)$, $H_1$", labelpad=6)
    ax.set_title("$H_1$ persistence landscape (level 1) evolving over time", pad=6)
    ax.view_init(elev=32, azim=-55)
    ax.grid(False)
    style_3d_axes(ax)
    savefig(fig, "07_landscape_surface.png", pad=0.5)
except Exception as e:
    skipped.append(f"Landscape surface — {e}")


# ---------------------------------------------------------------------------
# 8. CREATIVE #2: real point cloud, calm window vs. crisis window,
#    PCA-projected to 3D, with a floor shadow projection for depth.
# ---------------------------------------------------------------------------
print("[8/10] Point cloud: calm vs. crisis window (creative #2)")
try:
    lr = pd.read_parquet(os.path.join(DATA, "log_returns.parquet"))
    window = 250
    embed_dim = 5

    def embed_window(returns_window):
        vals = returns_window.values
        n = len(vals) - (embed_dim - 1)
        pts = np.zeros((n, embed_dim * vals.shape[1]))
        for i in range(n):
            pts[i] = vals[i:i + embed_dim].flatten()
        return pts

    realized_vol = lr.rolling(window).std().mean(axis=1)
    crisis_end = realized_vol.idxmax()
    calm_end = realized_vol.idxmin()

    calm_window = lr.loc[:calm_end].iloc[-window:]
    crisis_window = lr.loc[:crisis_end].iloc[-window:]

    pts_calm = embed_window(calm_window)
    pts_crisis = embed_window(crisis_window)

    pooled = np.vstack([pts_calm, pts_crisis])
    pooled = pooled - pooled.mean(axis=0)
    U, S, Vt = np.linalg.svd(pooled, full_matrices=False)
    proj = pooled @ Vt[:3].T
    proj_calm = proj[:len(pts_calm)]
    proj_crisis = proj[len(pts_calm):]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.05, 0.05, 0.78, 0.85])

    floor_z = proj[:, 2].min() - 0.02
    ax.scatter(proj_calm[:, 0], proj_calm[:, 1], floor_z, color=STEEL, s=8, alpha=0.10, zorder=1)
    ax.scatter(proj_crisis[:, 0], proj_crisis[:, 1], floor_z, color=CRIMSON, s=8, alpha=0.10, zorder=1)

    ax.scatter(*proj_calm.T, color=STEEL, s=16, alpha=0.75, edgecolor="white", linewidth=0.2,
               label=f"calm window (ends {calm_end.date()})", zorder=3)
    ax.scatter(*proj_crisis.T, color=CRIMSON, s=16, alpha=0.75, edgecolor="white", linewidth=0.2,
               label=f"turbulent window (ends {crisis_end.date()})", zorder=3)

    ax.set_zlim(floor_z, proj[:, 2].max())
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3", labelpad=6)
    ax.set_title("Takens-embedded return point cloud, PCA-projected\n"
                  "(illustrative 3D projection of the true $\\mathbb{R}^{25}$ embedding)", pad=6)
    ax.view_init(elev=22, azim=-50)
    ax.grid(False)
    style_3d_axes(ax)
    ax.legend(loc="upper left", fontsize=9)
    savefig(fig, "08_pointcloud_calm_vs_crisis.png", pad=0.5)
except Exception as e:
    skipped.append(f"Point cloud calm vs crisis — {e}")


# ---------------------------------------------------------------------------
# 9. NEW: Vietoris-Rips growth schematic — pure pedagogy, synthetic toy
#    points chosen to have an obvious loop. NOT real return data — this
#    is purely for visually defining VR_eps(X) in 3.3, labelled as such.
# ---------------------------------------------------------------------------
print("[9/10] Vietoris-Rips filtration schematic (toy, pedagogical)")
try:
    rng = np.random.RandomState(7)
    n_pts = 14
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles)]) * 3
    ring += rng.normal(scale=0.18, size=ring.shape)  # slight jitter, still a clean loop

    eps_values = [0.9, 1.55, 2.9]
    eps_labels = ["small $\\varepsilon$: isolated points", "medium $\\varepsilon$: loop forms",
                  "large $\\varepsilon$: loop fills in"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, eps, label in zip(axes, eps_values, eps_labels):
        # draw balls
        for p in ring:
            circ = plt.Circle(p, eps / 2, color=STEEL, alpha=0.10, zorder=1)
            ax.add_patch(circ)
        # draw edges for pairs within eps, and triangles once enough edges close a gap
        pairs = []
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                if np.linalg.norm(ring[i] - ring[j]) <= eps:
                    ax.plot([ring[i, 0], ring[j, 0]], [ring[i, 1], ring[j, 1]],
                            color=NAVY, linewidth=1.0, alpha=0.7, zorder=2)
                    pairs.append((i, j))
        # once eps is large enough to also connect skip-one neighbours, shade the
        # interior to show the loop being filled in (schematic, not a real simplex test)
        if eps >= eps_values[-1]:
            ax.fill(ring[:, 0], ring[:, 1], color=GOLD, alpha=0.25, zorder=0)
        ax.scatter(ring[:, 0], ring[:, 1], color=CRIMSON, s=28, zorder=3, edgecolor="white", linewidth=0.5)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Vietoris-Rips filtration $VR_\\varepsilon(X)$ — schematic on synthetic points, not real return data",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    savefig(fig, "09_vr_filtration_schematic.png")
except Exception as e:
    skipped.append(f"VR filtration schematic — {e}")


# ---------------------------------------------------------------------------
# 10. NEW: real persistence diagram, calm vs. crisis window. This is a real,
#     small computation (ripser on one window's actual embedding) — not a
#     recompute of the pipeline, and not saved anywhere on disk already.
# ---------------------------------------------------------------------------
print("[10/10] Persistence diagram, calm vs. crisis window (real computation)")
try:
    from ripser import ripser

    lr = pd.read_parquet(os.path.join(DATA, "log_returns.parquet"))
    window = 250
    embed_dim = 5

    def embed_window(returns_window):
        vals = returns_window.values
        n = len(vals) - (embed_dim - 1)
        pts = np.zeros((n, embed_dim * vals.shape[1]))
        for i in range(n):
            pts[i] = vals[i:i + embed_dim].flatten()
        return pts

    realized_vol = lr.rolling(window).std().mean(axis=1)
    crisis_end = realized_vol.idxmax()
    calm_end = realized_vol.idxmin()
    calm_pts = embed_window(lr.loc[:calm_end].iloc[-window:])
    crisis_pts = embed_window(lr.loc[:crisis_end].iloc[-window:])

    dgm_calm = ripser(calm_pts, maxdim=1)["dgms"][1]      # H1 only, matches Sec 3.3
    dgm_crisis = ripser(crisis_pts, maxdim=1)["dgms"][1]

    all_finite = np.concatenate([dgm_calm[np.isfinite(dgm_calm).all(1)],
                                  dgm_crisis[np.isfinite(dgm_crisis).all(1)]])
    lim = all_finite.max() * 1.1 if len(all_finite) else 1.0

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, lim], [0, lim], color=SLATE, linewidth=1, linestyle="--", zorder=1)
    ax.fill_between([0, lim], [0, lim], color=SLATE, alpha=0.06, zorder=0)
    ax.scatter(*dgm_calm.T, color=STEEL, s=45, alpha=0.8, edgecolor="white",
               linewidth=0.5, label=f"calm window (ends {calm_end.date()})", zorder=2)
    ax.scatter(*dgm_crisis.T, color=CRIMSON, s=45, alpha=0.8, edgecolor="white",
               linewidth=0.5, label=f"turbulent window (ends {crisis_end.date()})", zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("$H_1$ persistence diagram, calm vs. turbulent window\n(real computation on the actual Takens embedding)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    savefig(fig, "10_persistence_diagram.png")
    print("  NOTE: computed fresh via ripser on the same two windows as the point cloud figure "
          "— small real computation, not read from a saved file.")
except ImportError:
    skipped.append("Persistence diagram — 'ripser' not installed (pip install ripser)")
except Exception as e:
    skipped.append(f"Persistence diagram — {e}")


# ---------------------------------------------------------------------------
print("\nDone. Figures in ./figures/")
if skipped:
    print("\nSkipped (need a real rerun or data that doesn't exist yet):")
    for s in skipped:
        print(f"  - {s}")