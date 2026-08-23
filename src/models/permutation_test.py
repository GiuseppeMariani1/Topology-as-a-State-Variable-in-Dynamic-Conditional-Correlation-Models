"""
Permutation test for TopoDCC on L^p-norm (Gidea & Katz style) features.

Merges what were previously two near-identical modules:
  permutation_test_lpnorm.py      (unregularized fit_dcc_topo)
  permutation_test_lpnorm_reg.py  (Ridge-regularized fit_dcc_topo_reg)
into one, selected by the `regularized` flag. They shared ~85% of their
code -- the shuffle logic, worker pattern, multiprocessing block and
results printout were duplicated verbatim, so a fix or a speedup applied
to one silently didn't apply to the other.

THE QUESTION
  Does the L^p-norm landscape summary carry real signal, or is the
  in-sample improvement over baseline just the free parameters (9
  features -> ~20 params) fitting noise?

  This is the literature-matched follow-up to the original test, which
  found the raw 186-feature TopoDCC lost to shuffled features 10/10 --
  evidence that version was overfitting day-to-day flexibility rather
  than capturing topology. The L^p-norm reduction leaves far less spare
  capacity for shuffled features to exploit.

PROCEDURE
  1. Fit once on the real, correctly time-aligned features -> real_ll.
  2. Shuffle the *rows* of the feature matrix n_permutations times. This
     breaks the day <-> topology link (each day gets a random other
     day's features) while keeping every feature's marginal
     distribution, scale, and cross-feature correlation structure
     intact. Same parameters, same optimizer, same n_iter -- the only
     thing that changes is whether the features carry information about
     *when* things happened.
  3. Refit from scratch on each shuffled version -> permuted_ll[i].
  4. Compare real_ll against the permuted distribution.

READING THE RESULT
  - real_ll far above the permuted distribution -> the features carry
    real, time-aligned information.
  - real_ll inside or barely above it -> the model is fitting noise
    rather than topology content. At only 9 features, that would be a
    stronger negative result than the 186-feature case was.

REGULARIZED MODE (regularized=True)
  Tests fit_dcc_topo_reg at a FIXED lambda, read from the saved
  dcc_topo_reg results rather than hardcoded (so it can't drift out of
  sync if the grid search is rerun). This is arguably the more honest
  comparison, since the regularized model is the one you'd actually
  report.

PERMUTATION COUNT
  Both modes now default to 500. p can only be resolved to
  1/n_permutations, so the old defaults (10 / 100) capped the reportable
  p-value at 0.1 and 0.01 respectively regardless of how strong the
  result actually was. 500 resolves p to 0.002.

DEVICE HANDLING
  On CUDA, all permutations are fitted as ONE batched op (see
  fit_dcc_topo_batched) rather than one process per shuffle -- the
  models are tiny, so per-fit overhead dominates and batching is worth
  far more than process parallelism. On CPU, fits are split across
  worker processes as before. Both modes now get the GPU path; only the
  unregularized one did previously.

Usage:
  python src/models/permutation_test.py            # unregularized
  python src/models/permutation_test.py --reg      # regularized
"""

import argparse
import concurrent.futures
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch

from src.data.loader import load_aligned_data, load_config, standardize_features
from src.models.dcc_topo import compute_Q_bar, fit_dcc_topo, fit_dcc_topo_batched


# SINGLE-FIT HELPERS (CPU worker path)

def _fit_once(z_df, X_df, regularized, lambda_l2, n_iter, lr, patience, min_delta):
    """
    One fit on one (already shuffled or real) feature matrix, on CPU.

    Regularized mode fits on ALL rows with no train/test split, mirroring
    the "final model on full data" step in dcc_topo_reg.py -- that's the
    number that's comparable across real vs shuffled.
    """
    if not regularized:
        _, _, _, _, ll_hist = fit_dcc_topo(z_df, X_df, n_iter=n_iter, lr=lr, verbose=False)
        return ll_hist[-1]

    # Imported here rather than at module scope: dcc_topo_reg imports from
    # dcc_topo, and keeping this local avoids a circular-import surprise if
    # that ever gains an import back the other way.
    from src.models.dcc_topo_reg import fit_dcc_topo_reg

    z_t = torch.tensor(z_df.values, dtype=torch.float32)
    X_t = torch.tensor(standardize_features(X_df)[0], dtype=torch.float32)

    _, ll_hist, _, _ = fit_dcc_topo_reg(
        z_t, X_t, compute_Q_bar(z_t),
        lambda_l2=lambda_l2, n_iter=n_iter, lr=lr,
        patience=patience, min_delta=min_delta,
        device=torch.device('cpu'), verbose=False,
    )
    return ll_hist[-1]


def _permuted_fit_worker(seed, z_np, X_np, regularized, lambda_l2,
                         n_iter, lr, patience, min_delta):
    """
    Runs in a separate process: shuffles X's rows with this seed, fits,
    returns only the final ll (cheap to ship back across the boundary).
    """
    torch.set_num_threads(1)  # avoid every worker fighting for all cores

    rng = np.random.default_rng(seed)
    X_shuffled = X_np[rng.permutation(X_np.shape[0])]

    ll_final = _fit_once(
        pd.DataFrame(z_np), pd.DataFrame(X_shuffled),
        regularized, lambda_l2, n_iter, lr, patience, min_delta,
    )
    return {'seed': seed, 'll_final': ll_final}


# BATCHED GPU PATH

def _run_permutations_batched(z_np, X_np, seeds, regularized, lambda_l2,
                              n_iter, lr, device):
    """
    Build every shuffled feature matrix up front, stack into one
    (B, T, n_features) tensor, and fit all B models as a single batched
    training loop instead of one process per permutation.

    The regularized case is just the same batched fit with a constant
    per-item lambda, since fit_dcc_topo_batched already takes a (B,)
    lambda vector.
    """
    B = len(seeds)
    T, n_features = X_np.shape

    X_stack = np.empty((B, T, n_features), dtype=np.float32)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        X_shuffled = X_np[rng.permutation(T)]
        X_stack[i] = standardize_features(X_shuffled)[0]

    lam = None
    if regularized:
        lam = torch.full((B,), float(lambda_l2), dtype=torch.float32)

    print(f"  [batched] fitting all {B} shuffled versions in one pass on {device}...")
    _, ll_final, _ = fit_dcc_topo_batched(
        torch.tensor(z_np, dtype=torch.float32),
        torch.tensor(X_stack, dtype=torch.float32),
        n_iter=n_iter, lr=lr, lambda_l2=lam, device=device, verbose=False,
    )

    return [{'seed': s, 'll_final': float(ll_final[i])} for i, s in enumerate(seeds)]


# MAIN ENTRY POINT

def run_permutation_test(garch_residuals_df, tda_features_df,
                         regularized=False, lambda_l2=None,
                         n_permutations=None, n_iter=500, lr=0.01,
                         patience=30, min_delta=1e-4,
                         n_jobs=None, seed0=0, real_ll=None, device=None):
    """
    Returns: real_ll (float), permuted_lls (np.ndarray), results_df (DataFrame)

    regularized : fit the Ridge-penalised model at a fixed lambda_l2
                  instead of the unregularized model.
    lambda_l2   : required when regularized=True. Read it from the saved
                  dcc_topo_reg results rather than hardcoding.
    real_ll     : pass a cached value ONLY if it came from a run with
                  identical features, model, and hyperparameters. It was
                  one particular random init's result, not "the" answer
                  for real features -- a fresh run could land differently.
    device      : defaults to CUDA when available (batched path), else CPU
                  (process-pool path).
    """
    if regularized and lambda_l2 is None:
        raise ValueError("regularized=True requires lambda_l2 (read it from dcc_topo_reg results).")

    if n_permutations is None:
        # 500 for both modes. p can only be resolved to 1/n_permutations,
        # so the previous defaults (10 unregularized / 100 regularized)
        # could not distinguish "significant" from "not" at any useful
        # resolution -- 10 permutations bounds p at 0.1 no matter what the
        # result is. 500 gives p to 0.002, which is enough to report a
        # significance claim in the thesis without the resolution itself
        # being the binding constraint.
        n_permutations = 500

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label = f"REGULARIZED (lambda_l2={lambda_l2:.0e})" if regularized else "unregularized"

    z_np = garch_residuals_df.values.astype('float32')
    X_np = tda_features_df.values.astype('float32')

    if real_ll is None:
        print(f"Fitting {label} model on REAL (correctly aligned) features...")
        t0 = time.time()
        real_ll = _fit_once(garch_residuals_df, tda_features_df, regularized,
                            lambda_l2, n_iter, lr, patience, min_delta)
        print(f"  real_ll = {real_ll:.2f}  ({time.time() - t0:.1f}s)")
    else:
        print(f"Using cached real_ll = {real_ll:.2f} (skipping real fit)")

    seeds = [seed0 + i for i in range(n_permutations)]

    print(f"\nFitting {n_permutations} SHUFFLED versions [{label}]  (device={device})...")
    t0 = time.time()

    if device.type == 'cuda':
        rows = _run_permutations_batched(z_np, X_np, seeds, regularized,
                                         lambda_l2, n_iter, lr, device)
    else:
        if n_jobs is None:
            n_jobs = max(1, min(n_permutations, os.cpu_count() or 1))
        rows = []
        args = (regularized, lambda_l2, n_iter, lr, patience, min_delta)

        if n_jobs > 1:
            print(f"  running across {n_jobs} processes...")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_jobs, mp_context=mp.get_context('spawn')
            ) as ex:
                futures = [ex.submit(_permuted_fit_worker, s, z_np, X_np, *args) for s in seeds]
                for n_done, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                    row = fut.result()
                    rows.append(row)
                    if n_done % 10 == 0 or n_done == n_permutations:
                        print(f"    {n_done}/{n_permutations} done "
                              f"(last: seed={row['seed']:3d} ll={row['ll_final']:.2f})")
        else:
            for i, s in enumerate(seeds, start=1):
                rows.append(_permuted_fit_worker(s, z_np, X_np, *args))
                if i % 10 == 0 or i == n_permutations:
                    print(f"    {i}/{n_permutations} done "
                          f"(last: seed={rows[-1]['seed']:3d} ll={rows[-1]['ll_final']:.2f})")

    print(f"  done in {time.time() - t0:.1f}s")

    rows.sort(key=lambda r: r['seed'])
    permuted_lls = np.array([r['ll_final'] for r in rows])
    _print_results(real_ll, permuted_lls, n_permutations, regularized, lambda_l2)

    return real_ll, permuted_lls, pd.DataFrame(rows)


def _print_results(real_ll, permuted_lls, n_permutations, regularized, lambda_l2):
    print("\nRESULTS")
    print("_" * 60)
    if regularized:
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
        print("     features. That's a warning sign the gap is more about")
        print("     parameter count than topology content.")
        if regularized:
            print("     Regularization should shrink that gap relative to the")
            print("     unregularized test -- compare the two.")
    else:
        print("  -> real features outperformed every shuffled version.")
        print("     Consistent with the L^p-norm topology features carrying")
        print("     real signal, not just extra fitting capacity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--reg', action='store_true',
                        help='test the Ridge-regularized model at the best saved lambda')
    parser.add_argument('--n-permutations', type=int, default=None,
                        help='override the default (500 for both modes; p resolves to 1/n)')
    parser.add_argument('--fresh', action='store_true',
                        help='refit real features instead of reusing the cached ll')
    parser.add_argument('--features', default='lpnorm',
                        choices=['lpnorm', 'landscape', 'pi',
                                 'lpnorm_speed', 'lpnorm_levels_speed',
                                 'landscape_speed', 'landscape_levels_speed'],
                        help="which feature set to permutation-test. Non-lpnorm sets "
                             "always refit fresh (no cache lookup) and require a "
                             "--lambda-l2 for --reg since there is no per-feature-set "
                             "saved grid-search result to read from.")
    parser.add_argument('--window', type=int, default=None,
                        help="TDA window override -- rebuilds tda_pipeline (and the "
                             "derived velocity features, if --features is a _speed "
                             "variant) before running. Omit to use whatever window "
                             "the feature files on disk currently reflect.")
    parser.add_argument('--lambda-l2', type=float, default=None,
                        help="required with --reg when --features is not 'lpnorm' -- "
                             "there is no saved grid-search result to read lambda from "
                             "for other feature sets, so it must be supplied explicitly.")
    args = parser.parse_args()

    if args.window is not None:
        import subprocess
        import sys as _sys
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        print(f"Recomputing TDA features with window={args.window}...")
        result = subprocess.run(
            [_sys.executable, '-m', 'src.topology.tda_pipeline', '--window', str(args.window)],
            cwd=repo_root, capture_output=True, text=True
        )
        if result.returncode != 0:
            print("TDA recomputation failed:")
            print(result.stderr)
            raise SystemExit(1)
        print("TDA recomputation complete.")

        if args.features.endswith('_speed'):
            src = args.features.replace('_levels_speed', '').replace('_speed', '')
            mode = 'levels_speed' if args.features.endswith('_levels_speed') else 'speed'
            print(f"Rebuilding {mode} features from {src} at window={args.window}...")
            result = subprocess.run(
                [_sys.executable, '-m', 'src.topology.velocity_features',
                 '--source', src, '--mode', mode],
                cwd=repo_root, capture_output=True, text=True
            )
            if result.returncode != 0:
                print("Velocity feature rebuild failed:")
                print(result.stderr)
                raise SystemExit(1)
            print("Velocity rebuild complete.")

    config = load_config()
    garch_residuals, tda_features, paths = load_aligned_data(config, features=args.features)
    print(f"Feature columns: {list(tda_features.columns)}")

    cfg = config['models']['topo_reg' if args.reg else 'topo']

    lambda_l2 = None
    if args.reg:
        if args.features == 'lpnorm':
            # Read the validated best lambda from the saved grid-search output
            # rather than hardcoding, so this can't silently go stale.
            reg_path = paths.get('dcc_topo_reg', 'data/processed/dcc_topo_reg_results.npy')
            try:
                lambda_l2 = float(np.load(reg_path, allow_pickle=True).item()['lambda_l2'])
                print(f"Using best_lambda={lambda_l2:.0e} from {reg_path}")
            except (FileNotFoundError, KeyError) as exc:
                raise SystemExit(
                    f"Could not read the best lambda from {reg_path} ({exc}) -- run "
                    f"dcc_topo_reg.py first so this uses the validated optimum."
                )
        else:
            # There is no saved grid-search result for non-lpnorm feature sets
            # (the grid search was only ever run against the original 9-feature
            # lpnorm set at window=250). Silently reusing that lambda here
            # would be applying a value tuned for a different feature space
            # entirely -- require it explicitly instead.
            if args.lambda_l2 is None:
                raise SystemExit(
                    f"--reg with --features {args.features!r} requires --lambda-l2 "
                    f"explicitly -- there is no saved grid-search result for this "
                    f"feature set to read a validated lambda from."
                )
            lambda_l2 = args.lambda_l2
            print(f"Using --lambda-l2={lambda_l2:.0e} (explicitly supplied, not from a "
                  f"saved grid search for this feature set)")

    # Reuse the cached real-features ll only for the original lpnorm set at
    # whatever window is currently on disk -- for any other feature set (or
    # a --window override) there is no guarantee the cached file matches
    # what was just loaded, so always refit fresh in that case.
    cached_real_ll = None
    if not args.fresh and not args.reg and args.features == 'lpnorm' and args.window is None:
        topo_path = paths.get('dcc_topo_lpnorm', 'data/processed/dcc_topo_lpnorm_results.npy')
        try:
            cached_real_ll = np.load(topo_path, allow_pickle=True).item()['ll_final']
            print(f"Loaded cached real_ll = {cached_real_ll:.2f} from {topo_path}")
        except (FileNotFoundError, KeyError):
            print(f"No cached results at {topo_path} -- will refit real features.")

    real_ll, permuted_lls, results_df = run_permutation_test(
        garch_residuals, tda_features,
        regularized=args.reg,
        lambda_l2=lambda_l2,
        n_permutations=args.n_permutations,
        n_iter=cfg['n_iter'],
        lr=cfg['lr'],
        patience=cfg.get('patience', 30),
        min_delta=cfg.get('min_delta', 1e-4),
        real_ll=cached_real_ll,
    )

    key = 'permutation_test_lpnorm_reg' if args.reg else 'permutation_test_lpnorm'
    default = f"data/processed/{key}_results.npy"
    out_path = paths.get(key, default)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {
        'real_ll':      real_ll,
        'permuted_lls': permuted_lls,
        'results':      results_df.to_dict(),
        'regularized':  args.reg,
    }
    if args.reg:
        payload['lambda_l2'] = lambda_l2

    np.save(out_path, payload)
    print(f"\nSaved to {out_path}")
