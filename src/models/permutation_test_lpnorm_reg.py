"""
Permutation test for TopoDCC — L^p-norm features, REGULARIZED (Ridge) model.

Why this version exists:
  permutation_test_lpnorm.py tests the UNREGULARIZED fit_dcc_topo model
  (no penalty on w_a/w_b) against shuffled features. But you already
  established, in dcc_topo_reg.py, that Ridge regularization improves
  out-of-sample generalization for these same features (ll_test improved
  monotonically up to lambda_l2=1000). Testing the regularized model
  against permutations is the more honest comparison, since it's the
  regularized model you'd actually want to report/use -- carrying that
  established improvement into the permutation test is consistent with
  what you already know, not an attempt to move the number.

  This version:
    - Uses fit_dcc_topo_reg (from dcc_topo_reg.py) instead of the
      unregularized fit_dcc_topo, fit on ALL data (same as the "final
      model on full data" step in dcc_topo_reg.py), at the SAME best
      lambda found by that script's grid search.
    - Reads the best lambda from the saved dcc_topo_reg results file
      instead of hardcoding it, so this can't silently drift out of
      sync if the grid search is rerun and finds a different optimum.
    - Runs 100 permutations instead of 10, for a much finer-resolution
      p-value estimate (10 permutations can only resolve p to the
      nearest 0.1; 100 resolves to the nearest 0.01).

Procedure (same logic as permutation_test_lpnorm.py otherwise):
  1. Fit the regularized model once on the real, correctly time-aligned
     L^p-norm features -> real_ll.
  2. Shuffle the *rows* of the feature matrix 100 times (breaks the day
     <-> topology link, keeps each feature's own marginal distribution
     and cross-feature correlations intact).
  3. Refit the regularized model from scratch on each shuffled version,
     at the same fixed best lambda -> permuted_ll[i].
  4. Compare real_ll against the distribution of permuted_ll.

Usage:
  python src/models/permutation_test_lpnorm_reg.py
"""

import os
import sys
import time
import concurrent.futures
import multiprocessing as mp

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.dcc_topo import compute_Q_bar
from src.models.dcc_topo_reg import fit_dcc_topo_reg


def _fit_full_data_reg(z_df, X_df, lambda_l2, n_iter, lr, patience, min_delta):
    """
    Fit the regularized model on ALL rows (no train/test split -- this
    mirrors the "final model on full data" step in dcc_topo_reg.py, since
    that's the number that's actually comparable across real vs shuffled).
    Returns the final training ll (post early-stopping, best-checkpoint
    restored weights).
    """
    z_t = torch.tensor(z_df.values, dtype=torch.float32)

    X_raw  = X_df.values
    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0) + 1e-8
    X_t    = torch.tensor((X_raw - X_mean) / X_std, dtype=torch.float32)

    Q_bar = compute_Q_bar(z_t)

    _, ll_hist, _, _ = fit_dcc_topo_reg(
        z_t, X_t, Q_bar,
        lambda_l2=lambda_l2, n_iter=n_iter, lr=lr,
        patience=patience, min_delta=min_delta,
        device=torch.device('cpu'), verbose=False
    )
    return ll_hist[-1]


def _permuted_fit_worker(seed, z_np, X_np, lambda_l2, n_iter, lr, patience, min_delta):
    """
    Runs in a separate process. Shuffles the rows of X with this seed,
    fits the SAME regularized model at the SAME fixed lambda used for
    the real fit, returns only the final ll.
    """
    torch.set_num_threads(1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X_np.shape[0])
    X_shuffled = X_np[perm]

    z_df = pd.DataFrame(z_np)
    X_df = pd.DataFrame(X_shuffled)

    ll_final = _fit_full_data_reg(z_df, X_df, lambda_l2, n_iter, lr, patience, min_delta)
    return {'seed': seed, 'll_final': ll_final}


def run_permutation_test_reg(garch_residuals_df, tda_features_df,
                             lambda_l2,
                             n_permutations=100, n_iter=500, lr=0.01,
                             patience=30, min_delta=1e-4,
                             n_jobs=None, seed0=0, real_ll=None):
    """
    Returns: real_ll (float), permuted_lls (np.ndarray), results_df (pd.DataFrame)

    real_ll: pass a cached value ONLY if it came from a previous run at
      this exact lambda_l2, on these exact features, same n_iter/lr/
      patience/min_delta. Otherwise leave as None to fit fresh here.
    """
    z_np = garch_residuals_df.values.astype('float32')
    X_np = tda_features_df.values.astype('float32')

    if real_ll is None:
        print(f"Fitting REGULARIZED model (lambda_l2={lambda_l2:.0e}) on REAL "
              f"(correctly aligned) features...")
        t0 = time.time()
        real_ll = _fit_full_data_reg(
            garch_residuals_df, tda_features_df, lambda_l2, n_iter, lr, patience, min_delta
        )
        print(f"  real_ll = {real_ll:.2f}  ({time.time() - t0:.1f}s)")
    else:
        print(f"Using cached real_ll = {real_ll:.2f} (skipping real fit)")

    if n_jobs is None:
        n_jobs = max(1, min(n_permutations, os.cpu_count() or 1))

    seeds = [seed0 + i for i in range(n_permutations)]
    rows = []

    print(f"\nFitting {n_permutations} SHUFFLED versions (regularized, "
          f"lambda_l2={lambda_l2:.0e}) across {n_jobs} processes...")
    t0 = time.time()
    if n_jobs > 1:
        ctx = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            futures = {
                ex.submit(_permuted_fit_worker, s, z_np, X_np, lambda_l2,
                          n_iter, lr, patience, min_delta): s
                for s in seeds
            }
            n_done = 0
            for fut in concurrent.futures.as_completed(futures):
                row = fut.result()
                rows.append(row)
                n_done += 1
                if n_done % 10 == 0 or n_done == n_permutations:
                    print(f"    {n_done}/{n_permutations} done "
                          f"(last: seed={row['seed']:3d}  ll_final={row['ll_final']:.2f})")
    else:
        for i, s in enumerate(seeds, start=1):
            row = _permuted_fit_worker(s, z_np, X_np, lambda_l2, n_iter, lr, patience, min_delta)
            rows.append(row)
            if i % 10 == 0 or i == n_permutations:
                print(f"    {i}/{n_permutations} done "
                      f"(last: seed={row['seed']:3d}  ll_final={row['ll_final']:.2f})")
    print(f"  done in {time.time() - t0:.1f}s")

    rows.sort(key=lambda r: r['seed'])
    permuted_lls = np.array([r['ll_final'] for r in rows])

    print("\nRESULTS")
    print("_" * 60)
    print(f"  lambda_l2 (fixed)    = {lambda_l2:.0e}")
    print(f"  real_ll              = {real_ll:.2f}")
    print(f"  permuted mean        = {permuted_lls.mean():.2f}")
    print(f"  permuted std         = {permuted_lls.std():.2f}")
    print(f"  permuted min / max   = {permuted_lls.min():.2f} / {permuted_lls.max():.2f}")
    print(f"  real - permuted mean = {real_ll - permuted_lls.mean():.2f}")
    n_as_good = int((permuted_lls >= real_ll).sum())
    print(f"  permuted runs >= real_ll: {n_as_good}/{n_permutations}  "
          f"(p ~= {n_as_good / n_permutations:.3f})")
    if n_as_good >= 1:
        print("  -> at least one shuffled-feature run matched or beat the real")
        print("     features, even under regularization. Still a signal that")
        print("     some of the gap is about capacity rather than topology")
        print("     content, though regularization should shrink that gap")
        print("     relative to the unregularized permutation test.")
    else:
        print("  -> real features clearly outperformed every shuffled version")
        print("     under regularization too. Stronger evidence the L^p-norm")
        print("     topology features carry real signal, not just capacity.")

    return real_ll, permuted_lls, pd.DataFrame(rows)


if __name__ == "__main__":
    from src.data.loader import load_config

    config = load_config()
    paths = config['paths']
    topo_reg_cfg = config['models']['topo_reg']

    garch_residuals = pd.read_parquet(paths['garch_residuals'])
    tda_features = pd.read_parquet(
        paths.get('tda_features_lpnorm', 'data/processed/tda_features_lpnorm.parquet')
    )

    common = garch_residuals.index.intersection(tda_features.index)
    garch_residuals = garch_residuals.loc[common]
    tda_features = tda_features.loc[common]

    assert (garch_residuals.index == tda_features.index).all(), \
        "Index mismatch — re-run lp_norm_features.py"

    print(f"Residuals: {garch_residuals.shape}")
    print(f"Features:  {tda_features.shape}")
    print(f"Feature columns: {list(tda_features.columns)}")

    # Read the best lambda from the actual saved dcc_topo_reg.py output,
    # instead of hardcoding it, so this can't silently go stale if the
    # grid search is rerun and lands on a different optimum.
    reg_results_path = paths.get('dcc_topo_reg', 'data/processed/dcc_topo_reg_results.npy')
    try:
        reg_results = np.load(reg_results_path, allow_pickle=True).item()
        best_lambda = float(reg_results['lambda_l2'])
        print(f"\nUsing best_lambda={best_lambda:.0e} from {reg_results_path}")
    except (FileNotFoundError, KeyError):
        raise SystemExit(
            f"Could not load best lambda from {reg_results_path} — "
            f"run dcc_topo_reg.py first so this test can use the actual "
            f"validated best lambda instead of guessing one."
        )

    patience  = topo_reg_cfg.get('patience', 30)
    min_delta = topo_reg_cfg.get('min_delta', 1e-4)

    real_ll, permuted_lls, results_df = run_permutation_test_reg(
        garch_residuals,
        tda_features,
        lambda_l2=best_lambda,
        n_permutations=100,
        n_iter=topo_reg_cfg['n_iter'],
        lr=topo_reg_cfg['lr'],
        patience=patience,
        min_delta=min_delta,
    )

    out_path = paths.get('permutation_test_lpnorm_reg',
                         'data/processed/permutation_test_lpnorm_reg_results.npy')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, {
        'real_ll':      real_ll,
        'permuted_lls': permuted_lls,
        'lambda_l2':    best_lambda,
        'results':      results_df.to_dict(),
    })
    print(f"\nSaved to {out_path}")
