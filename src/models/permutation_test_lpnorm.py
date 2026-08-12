"""
Permutation test for TopoDCC — L^p-norm (Gidea & Katz style) features.

Does the L^p-norm landscape summary carry real signal, or is the
+17.48 nat in-sample improvement over baseline just the ~20-30 free
parameters (9 features) fitting noise the same way the raw 186-feature
version did?

This is the literature-matched follow-up to the original permutation
test, which found the raw 186-feature TopoDCC lost to shuffled features
10/10 — evidence that version was overfitting day-to-day flexibility
rather than capturing real topology signal. This version uses the
L^p-norm reduced features instead (3 H1 landscape-level norms + 6
scalar features = 9 total), which leaves far less spare capacity for
shuffled features to exploit.

Procedure:
  1. Fit TopoDCC once on the real (correctly time-aligned) L^p-norm
     features -> real_ll.
  2. Shuffle the *rows* of the feature matrix n_permutations times. This
     breaks the day <-> topology link (each day gets a random other day's
     features) while keeping every feature's own marginal distribution,
     scale, and cross-feature correlation structure intact. Same ~20-30
     parameters, same optimizer, same n_iter -- the only thing that
     changes is whether the features carry information about *when*
     things happened.
  3. Refit TopoDCC from scratch on each shuffled version -> permuted_ll[i].
  4. Compare real_ll against the distribution of permuted_ll.

Reading the result:
  - real_ll far above the permuted distribution -> the L^p-norm features
    carry real, time-aligned information; the model is doing more than
    just exploiting extra parameters.
  - real_ll inside or barely above the permuted distribution -> even at
    9 features, the model is fitting noise rather than topology content
    that's earning its keep. Given how little capacity is left to overfit
    with at this size, this outcome would be a stronger negative result
    than the 186-feature case was.

Usage:
  python src/models/permutation_test_lpnorm.py
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

from src.models.dcc_topo import fit_dcc_topo


def _permuted_fit_worker(seed, z_np, X_np, n_iter, lr):
    """
    Runs in a separate process. Shuffles the rows of X with this seed,
    fits TopoDCC via the exact same fit_dcc_topo used for the real run,
    returns only the final ll (cheap to ship back across the process
    boundary -- no need to return the whole model or history).
    """
    torch.set_num_threads(1)  # avoid every worker fighting for all cores

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X_np.shape[0])
    X_shuffled = X_np[perm]

    z_df = pd.DataFrame(z_np)
    X_df = pd.DataFrame(X_shuffled)

    _, _, _, _, ll_hist = fit_dcc_topo(z_df, X_df, n_iter=n_iter, lr=lr, verbose=False)
    return {'seed': seed, 'll_final': ll_hist[-1]}


def run_permutation_test(garch_residuals_df, tda_features_df,
                         n_permutations=20, n_iter=500, lr=0.01,
                         n_jobs=None, seed0=0, real_ll=None):
    """
    Returns: real_ll (float), permuted_lls (np.ndarray), results_df (pd.DataFrame)

    real_ll: if you already have a final log likelyhood from a previous identical run
      (same features, same n_iter/lr, same dcc_topo.py code), pass it here
      to skip refitting on real features. Note this was one particular
      random init's result, not "the" answer for real features -- a fresh
      run could land a bit differently. Fine as a time-saver if nothing
      about the code/data/settings has changed since you got that number.
    """
    z_np = garch_residuals_df.values.astype('float32')
    X_np = tda_features_df.values.astype('float32')

    if real_ll is None:
        print("Fitting on REAL (correctly aligned) features...")
        t0 = time.time()
        _, _, _, _, real_ll_hist = fit_dcc_topo(
            garch_residuals_df, tda_features_df, n_iter=n_iter, lr=lr, verbose=False
        )
        real_ll = real_ll_hist[-1]
        print(f"  real_ll = {real_ll:.2f}  ({time.time() - t0:.1f}s)")
    else:
        print(f"Using cached real_ll = {real_ll:.2f} (skipping real fit)")

    if n_jobs is None:
        n_jobs = max(1, min(n_permutations, os.cpu_count() or 1))

    seeds = [seed0 + i for i in range(n_permutations)]
    rows = []

    print(f"\nFitting {n_permutations} SHUFFLED versions across {n_jobs} processes...")
    t0 = time.time()
    if n_jobs > 1:
        ctx = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            futures = {
                ex.submit(_permuted_fit_worker, s, z_np, X_np, n_iter, lr): s
                for s in seeds
            }
            for fut in concurrent.futures.as_completed(futures):
                row = fut.result()
                rows.append(row)
                print(f"    seed={row['seed']:3d}  ll_final={row['ll_final']:.2f}")
    else:
        for s in seeds:
            row = _permuted_fit_worker(s, z_np, X_np, n_iter, lr)
            rows.append(row)
            print(f"    seed={row['seed']:3d}  ll_final={row['ll_final']:.2f}")
    print(f"  done in {time.time() - t0:.1f}s")

    rows.sort(key=lambda r: r['seed'])
    permuted_lls = np.array([r['ll_final'] for r in rows])

    print("\nRESULTS")
    print("_" * 60)
    print(f"  real_ll              = {real_ll:.2f}")
    print(f"  permuted mean        = {permuted_lls.mean():.2f}")
    print(f"  permuted std         = {permuted_lls.std():.2f}")
    print(f"  permuted min / max   = {permuted_lls.min():.2f} / {permuted_lls.max():.2f}")
    print(f"  real - permuted mean = {real_ll - permuted_lls.mean():.2f}")
    n_as_good = int((permuted_lls >= real_ll).sum())
    print(f"  permuted runs >= real_ll: {n_as_good}/{n_permutations}")
    if n_as_good >= 1:
        print("  -> at least one shuffled-feature run matched or beat the real")
        print("     features. That's a warning sign the gap is more about")
        print("     parameter count than topology content -- even at 9 features.")
    else:
        print("  -> real features clearly outperformed every shuffled version.")
        print("     Consistent with the L^p-norm topology features carrying")
        print("     real signal, not just extra fitting capacity.")

    return real_ll, permuted_lls, pd.DataFrame(rows)


if __name__ == "__main__":
    from src.data.loader import load_config

    config = load_config()
    paths = config['paths']
    topo_cfg = config['models']['topo']

    garch_residuals = pd.read_parquet(paths['garch_residuals'])

    # L^p-norm reduced features (literature-matched, H1-only): 9 columns
    # total, replacing the raw 186-column landscape used in the run that
    # failed the original permutation test 10/10.
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

    real_ll, permuted_lls, results_df = run_permutation_test(
        garch_residuals,
        tda_features,
        n_permutations=topo_cfg.get('n_permutations', 10),
        n_iter=topo_cfg['n_iter'],
        lr=topo_cfg['lr'],
        # Reusing the final ll from the dcc_topo_lpnorm.py run just completed
        # (same features, same n_iter=500/lr=0.01, same code) instead of
        # refitting from scratch. Drop this argument (real_ll=None) if
        # anything about the features, n_iter, or lr changes before rerunning.
        dcc_topo_path = paths.get('dcc_topo_lpnorm', 'data/processed/dcc_topo_lpnorm_results.npy')
try:
    cached = np.load(dcc_topo_path, allow_pickle=True).item()
    cached_real_ll = cached['ll_final']
    print(f"Loaded cached real_ll = {cached_real_ll:.2f} from {dcc_topo_path}")
except (FileNotFoundError, KeyError):
    cached_real_ll = None
    print(f"No cached results at {dcc_topo_path} -- will refit real features from scratch.")

real_ll, permuted_lls, results_df = run_permutation_test(
    garch_residuals,
    tda_features,
    n_permutations=topo_cfg.get('n_permutations', 10),
    n_iter=topo_cfg['n_iter'],
    lr=topo_cfg['lr'],
    real_ll=cached_real_ll,
)
    )

    out_path = paths.get('permutation_test_lpnorm', 'data/processed/permutation_test_lpnorm_results.npy')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, {
        'real_ll':      real_ll,
        'permuted_lls': permuted_lls,
        'results':      results_df.to_dict(),
    })
    print(f"\nSaved to {out_path}")