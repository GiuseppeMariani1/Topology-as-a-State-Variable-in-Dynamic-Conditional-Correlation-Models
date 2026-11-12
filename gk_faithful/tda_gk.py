# -*- coding: utf-8 -*-
"""
Gidea & Katz (2018) faithful persistence-landscape norms.

WHAT IS DIFFERENT FROM src/topology/tda_pipeline.py
  Thesis pipeline                      | This module (G&K)
  -------------------------------------|---------------------------------------
  delay embedding, embed_dim = 5       | none: one day = one point in R^d
  per-window standardisation           | none (optional, for sensitivity only)
  PCA to 10 dims before Rips           | none
  window 250 (100-400 in sweeps)       | w = 50 (main) and 100 (robustness)
  first 3 landscape levels on a        | the FULL landscape, all levels k >= 1,
  30-point grid set from the 95%       | integrated exactly from the diagram
  quantile of ALL diagrams             | (no grid, so no full-sample quantile)
  L2 norm per level                    | L1 and L2 norm of the whole landscape

EXACT NORMS WITHOUT A GRID
  At each filtration value t, the landscape levels lambda_1(t) >= lambda_2(t)
  >= ... are the tent values f_i(t) = max(0, min(t - b_i, d_i - t)) of the
  bars, sorted. Sorting is a permutation, so for every t

      sum_k lambda_k(t)^p = sum_i f_i(t)^p.

  Integrating, the G&K norm ||lambda||_p^p = sum_k ||lambda_k||_p^p is

      sum_i  integral f_i(t)^p dt  =  sum_i  2 h_i^(p+1) / (p+1),

  with h_i = (d_i - b_i) / 2 the half-life of bar i. For p = 1 this is
  sum h_i^2 and for p = 2 it is (2/3) sum h_i^3. This is the continuous
  quantity in the paper, exactly, and it uses nothing outside the window.
  tests/test_gk.py checks it against grid integration of the thesis's own
  compute_persistence_landscape().

USAGE
  python -m gk_faithful.tda_gk              # all windows in config_gk.yaml
  python -m gk_faithful.tda_gk --window 50
"""

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ripser import ripser

from gk_faithful.common import load_gk_config, load_thesis_paths, norms_path
from src.topology.tda_pipeline import normalize_pointcloud  # original, read-only use


def landscape_norm_from_diagram(dgm, p):
    """Exact ||lambda||_p of the full persistence landscape of diagram dgm."""
    if dgm is None or len(dgm) == 0:
        return 0.0
    finite = np.isfinite(dgm[:, 1])
    h = (dgm[finite, 1] - dgm[finite, 0]) / 2.0
    h = h[h > 0]
    if len(h) == 0:
        return 0.0
    return float((np.sum(2.0 * h ** (p + 1) / (p + 1))) ** (1.0 / p))


def h1_diagram(points, standardise=False):
    """H1 persistence diagram of the Rips filtration on a point cloud."""
    X = normalize_pointcloud(points) if standardise else points
    return ripser(X, maxdim=1)['dgms'][1]


def _window_norms(s, e, returns, ps, standardise):
    win = returns[s:e]
    if np.isnan(win).any():
        return None
    dgm = h1_diagram(win, standardise)
    row = {f'norm_l{p}': landscape_norm_from_diagram(dgm, p) for p in ps}
    row['n_h1_bars'] = int(np.isfinite(dgm[:, 1]).sum()) if len(dgm) else 0
    return row


def compute_norm_series(log_returns_df, window, ps=(1, 2), step=1,
                        standardise=False, n_jobs=-1, verbose=True):
    """
    One row per window, dated on the window's LAST day (same convention as
    the thesis pipeline; the one-day lag is applied later, in data.py).
    """
    returns = log_returns_df.values.astype(float)
    dates = log_returns_df.index
    slices = [(s, s + window) for s in range(0, len(returns) - window + 1, step)]

    if verbose:
        print(f"G&K norms: w={window}, {returns.shape[1]} assets, "
              f"{len(slices)} windows, standardise={standardise}, p={list(ps)}")

    t0 = time.time()
    rows = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_window_norms)(s, e, returns, ps, standardise) for s, e in slices
    )
    keep = [(r, dates[e - 1]) for r, (s, e) in zip(rows, slices) if r is not None]
    out = pd.DataFrame([r for r, _ in keep], index=pd.DatetimeIndex([d for _, d in keep]))

    if verbose:
        print(f"  {len(out)} valid windows in {time.time() - t0:.1f}s  "
              f"({out.index[0].date()} -> {out.index[-1].date()})")
        print(out.describe().round(6).to_string())
    return out


def main():
    gk = load_gk_config()
    parser = argparse.ArgumentParser(description="Gidea & Katz faithful landscape norms")
    parser.add_argument('--window', type=int, default=None,
                        help="point-cloud window; default: every window in config_gk.yaml")
    args = parser.parse_args()

    pc = gk['point_cloud']
    windows = [args.window] if args.window else pc['windows']
    log_returns = pd.read_parquet(load_thesis_paths()['log_returns'])

    for w in windows:
        df = compute_norm_series(log_returns, w, ps=gk['norms']['p'], step=pc['step'],
                                 standardise=pc['standardise_window'], n_jobs=pc['n_jobs'])
        path = norms_path(w, gk)
        df.to_parquet(path)
        print(f"Saved {path}\n")


if __name__ == "__main__":
    main()
