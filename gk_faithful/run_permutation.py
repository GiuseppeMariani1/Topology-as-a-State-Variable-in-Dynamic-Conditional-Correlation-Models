# -*- coding: utf-8 -*-
"""
Permutation test for TopoDCC on the G&K features.

Calls src.models.permutation_test.run_permutation_test unchanged; it takes
the residual and feature DataFrames directly, so no patching is needed.
Same design as the thesis test: fit on the real, time-aligned features,
then on n shuffles of the feature rows, and compare log-likelihoods.

USAGE
  python -m gk_faithful.run_permutation --features gk_l2 --window 50 --m 500
  python -m gk_faithful.run_permutation --features gk_l2 --window 50 --m 500 --reg --lambda-l2 4000
"""

import argparse

import numpy as np

from gk_faithful.common import load_gk_config, result_path, run_tag
from gk_faithful.data import FEATURE_SETS, load_aligned_gk
from src.data.loader import load_config  # original, read-only use
from src.models.permutation_test import run_permutation_test  # original, read-only use


def run(features, w, m, reg=False, lambda_l2=None, n_permutations=None, lag=None):
    if reg and lambda_l2 is None:
        raise SystemExit("--reg needs --lambda-l2: no lambda has been tuned on G&K features.")
    gk = load_gk_config()
    n_permutations = n_permutations or gk['experiments']['n_permutations']

    z, X, _ = load_aligned_gk(features, w, m, lag=lag)
    cfg = load_config()['models']['topo_reg' if reg else 'topo']

    real_ll, permuted_lls, results_df = run_permutation_test(
        z, X, regularized=reg, lambda_l2=lambda_l2,
        n_permutations=n_permutations, n_iter=cfg['n_iter'], lr=cfg['lr'],
        patience=cfg.get('patience', 30), min_delta=cfg.get('min_delta', 1e-4),
    )

    n_beat = int((permuted_lls >= real_ll).sum())
    # n_beat / n matches how the thesis reported p (e.g. 30/500 = 0.060);
    # (n_beat + 1) / (n + 1) is the finite-sample-exact version.
    p = n_beat / len(permuted_lls)
    p_exact = (n_beat + 1) / (len(permuted_lls) + 1)
    out = result_path('perm_reg' if reg else 'perm', run_tag(features, w, m, 'full'))
    np.save(out, {
        'features': features, 'window': w, 'spectral_window': m,
        'feature_columns': list(X.columns),
        'real_ll': real_ll, 'permuted_lls': permuted_lls,
        'n_beat_real': n_beat, 'p_value': p, 'p_value_exact': p_exact,
        'results': results_df.to_dict(), 'regularized': reg, 'lambda_l2': lambda_l2,
    }, allow_pickle=True)
    print(f"\n{n_beat}/{len(permuted_lls)} shuffles >= real  ->  p = {p:.4f} (exact {p_exact:.4f})")
    print(f"Saved to {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Permutation test on G&K features")
    parser.add_argument('--features', default='gk_l2', choices=FEATURE_SETS)
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--m', type=int, default=500)
    parser.add_argument('--reg', action='store_true')
    parser.add_argument('--lambda-l2', type=float, default=None)
    parser.add_argument('--n-permutations', type=int, default=None)
    parser.add_argument('--lag', type=int, default=None)
    args = parser.parse_args()
    run(args.features, args.window, args.m, args.reg, args.lambda_l2,
        args.n_permutations, args.lag)


if __name__ == "__main__":
    main()
