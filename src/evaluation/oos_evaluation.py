# -*- coding: utf-8 -*-
"""
Out-of-sample evaluation: the honest test of whether TopoDCC's edge is real.

THE PROBLEM THIS SOLVES
  Every result so far -- the log-likelihood gain, the LR test, and the
  portfolio backtest -- is IN-SAMPLE. All models were fit on the full
  dataset and then evaluated on that same dataset. The unregularized
  TopoDCC showed a small portfolio edge over baseline DCC (8.121% vs
  8.142% realized vol, Sharpe 0.708 vs 0.695) while the regularized
  version showed essentially none. That pattern is exactly what
  overfitting looks like: the model with more effective free capacity
  scores better on the data it was fit to.

  It is also exactly what a real-but-small effect looks like, with
  regularization shrinking a genuine signal too aggressively.

  In-sample results cannot distinguish those two stories. This script
  can: it refits everything on a training window only, generates
  forecasts forward through a held-out test window the models never saw,
  and evaluates there.

WHAT IT DOES
  1. Chronological 80/20 split (no shuffling -- this is time series).
  2. Fit on TRAIN ONLY:
       - baseline DCC       (constant a, b)
       - aDCC               (constant a, b, g -- the harder benchmark)
       - TopoDCC unregularized  (lambda = 0)
       - TopoDCC regularized    (lambda from config, default 4000)
  3. Feature standardization uses TRAIN statistics only. Using full-sample
     mean/std would leak test-period information into the fit through the
     scaling, which is a subtle but real form of lookahead.
  4. Run each model's recursion forward across the full sample so the test
     period is warm-started from training history (standard practice --
     Q_{t-1} at the first test date should reflect real history, not a
     cold restart), then slice out the test period only.
  5. Portfolio backtest on the test period: min-variance weights from each
     model's covariance forecast, vs equal-weight.
  6. Diebold-Mariano tests on the test period, under QLIKE and Frobenius
     loss, for each model pair of interest.

READING THE RESULT
  - If unregularized TopoDCC still beats baseline out-of-sample, on both
    the portfolio metrics and a significant DM statistic, the edge is real.
  - If the edge shrinks toward zero or reverses, it was in-sample
    overfitting -- which is a legitimate and publishable finding, and far
    better discovered here than by an examiner.
  - The aDCC comparison is the harder test: if TopoDCC only beats vanilla
    DCC but not aDCC, then what topology is capturing may be asymmetry
    that a standard extension already handles more parsimoniously.

USAGE
  python -m src.evaluation.oos_evaluation
  python -m src.evaluation.oos_evaluation --train-frac 0.8 --n-iter 500
  python -m src.evaluation.oos_evaluation --skip-adcc     # faster, if aDCC not needed yet

NOTE ON RUNTIME
  Each fit here is a SINGLE model (no lambda grid, no permutation batch),
  which is the CPU-appropriate path -- dcc_topo_reg.py's own design notes
  say the single sequential fit has no batch width for a GPU to exploit
  and is forced to CPU deliberately. Expect this to be dominated by four
  sequential fits of ~500 Adam iterations each over ~3800 training days.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.loader import load_aligned_data, load_config
from src.models.dcc_baseline import fit_dcc_baseline, dcc_recursion_torch
from src.models.dcc_adcc import (fit_dcc_adcc, adcc_recursion,
                                  compute_N_bar, compute_delta)
from src.models.dcc_topo import TopoDCC, dcc_topo_recursion, compute_Q_bar
from src.models.dcc_topo_reg import fit_dcc_topo_reg
from src.portfolio.covariance import reconstruct_covariance
from src.portfolio.weights import min_variance_weights_seq, equal_weights_seq
from src.portfolio.backtest import (portfolio_returns, turnover,
                                     annualize_vol, annualize_sharpe)
from src.evaluation.diebold_mariano import dm_test_all_losses, dm_test, portfolio_sq_loss, qlike_loss_seq


def prepare_split(train_frac=0.8, features='lpnorm', pca_components=None, window=None,
                   split_date=None, verbose=True):
    """
    Load aligned residuals + topology features, split chronologically,
    and standardize features on training statistics only.

    split_date, if given, OVERRIDES train_frac and sets an exact cutoff
    date instead of a fraction. This matters for crisis-period analysis:
    with this project's sample (2006-2025) and a plain 80/20 split, the
    test period lands around 2022-2025 -- which excludes COVID (2020),
    the GFC (2008-09), and the EU debt crisis (2011-12) entirely, since
    all three fall inside the 80% training portion. A literature precedent
    for this kind of topology-forecasting claim (Souto 2023, "Topological
    Tail Dependence") finds its positive result specifically concentrated
    in a held-out COVID subsample -- their sample happens to end soon
    enough after COVID that an 80/20 split naturally captures it. Ours
    does not, unless the split date is moved explicitly.

    Passing split_date='2019-12-31' pushes COVID, and the 2022 rate-hike
    period, into the test set, at the cost of a smaller training sample
    (~13 years instead of ~15). This is a deliberate methodological choice
    to align with how the literature actually finds its effect, not an
    arbitrary tuning choice.

    If window is not None, recompute TDA features for that window size
    before loading (via subprocess call to tda_pipeline.py).

    If pca_components is not None, fit PCA on the training set and
    transform both train and test. PCA is fit only on train data to avoid
    lookahead (using test information to set the projection).

    Returns a dict of everything downstream steps need.
    """
    import subprocess
    
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    if window is not None:
        if verbose:
            print(f"\nRecomputing TDA features with window={window}...")
        result = subprocess.run(
            [sys.executable, '-m', 'src.topology.tda_pipeline', '--window', str(window)],
            cwd=repo_root, capture_output=True, text=True
        )
        if result.returncode != 0:
            print("TDA recomputation failed:")
            print(result.stderr)
            raise RuntimeError("TDA pipeline failed")
        if verbose:
            print("TDA recomputation complete.")

        # tda_pipeline.py only writes the raw landscape file
        # (tda_features_landscape.parquet). The 9-feature lpnorm summary is
        # a SEPARATE derived file produced by lp_norm_features.py, which
        # reads whatever landscape file is currently on disk and takes no
        # window argument of its own -- it just processes what's there.
        #
        # Omitting this step was a real bug in every --window run before
        # this fix: it silently left tda_features_lpnorm.parquet at
        # whatever window it was last built with, while printing "TDA
        # recomputation complete" as if the full chain had updated.
        # Confirmed directly: window=50/60/75 runs on 'lpnorm' and
        # 'lpnorm_speed' all produced bit-identical train/test splits,
        # identical baseline DCC fits, and (for the permutation test)
        # identical real_ll/permuted-mean/permuted-std/p-value to two
        # decimal places -- they were reading the same stale lpnorm file
        # every time, regardless of --window.
        if verbose:
            print("Deriving lpnorm features from the updated landscape file...")
        result = subprocess.run(
            [sys.executable, '-m', 'src.topology.lp_norm_features'],
            cwd=repo_root, capture_output=True, text=True
        )
        if result.returncode != 0:
            print("lp_norm_features.py failed:")
            print(result.stderr)
            raise RuntimeError("lp_norm_features.py failed")
        if verbose:
            print("lpnorm derivation complete.")

        # Velocity features are DERIVED from the TDA feature file, so a
        # window change invalidates them too. Without this they would
        # silently stay at whatever window they were last built with,
        # and the run would report results for a window it didn't use.
        if features.endswith('_speed'):
            src = features.replace('_levels_speed', '').replace('_speed', '')
            mode = 'levels_speed' if features.endswith('_levels_speed') else 'speed'
            if verbose:
                print(f"Rebuilding {mode} features from {src} at window={window}...")
            result = subprocess.run(
                [sys.executable, '-m', 'src.topology.velocity_features',
                 '--source', src, '--mode', mode],
                cwd=repo_root, capture_output=True, text=True
            )
            if result.returncode != 0:
                print("Velocity feature rebuild failed:")
                print(result.stderr)
                raise RuntimeError("velocity_features failed")
            if verbose:
                print("Velocity rebuild complete.")
    
    config = load_config()
    z_df, X_df, paths = load_aligned_data(config=config, features=features, verbose=verbose)

    # Sanity check: if a --window override was requested, confirm the file
    # that was just loaded actually reflects it, rather than silently
    # trusting the subprocess calls above. A window-W TDA computation
    # should produce a first valid date roughly W trading days after the
    # raw log-returns start; if the loaded index looks unchanged from
    # before the override, the recomputation chain didn't actually update
    # the file this run is about to use. This exact failure mode is what
    # produced bit-identical window=50/60/75 results earlier in this
    # project -- see the comments in the --window block above.
    if window is not None:
        expected_min_start_offset = window - 20  # loose tolerance for step/embed_dim effects
        log_returns_start = pd.read_parquet(config['paths']['log_returns']).index[0]
        actual_offset = (X_df.index[0] - log_returns_start).days
        if actual_offset * (252 / 365) < expected_min_start_offset * 0.5:
            raise RuntimeError(
                f"Loaded features start at {X_df.index[0].date()}, only "
                f"~{actual_offset} calendar days after the raw data starts "
                f"({log_returns_start.date()}) -- too soon for a window={window} "
                f"computation. The feature file likely was NOT actually "
                f"regenerated at this window; check the subprocess calls above "
                f"for silent failures before trusting these results."
            )

    T = len(z_df)
    if split_date is not None:
        split_ts = pd.Timestamp(split_date)
        train_size = int((z_df.index <= split_ts).sum())
        if train_size == 0 or train_size == T:
            raise ValueError(
                f"split_date={split_date} produces train_size={train_size} out of "
                f"{T} -- date is outside the sample range "
                f"({z_df.index[0].date()} to {z_df.index[-1].date()})."
            )
        split_desc = f"explicit split_date={split_date}"
    else:
        train_size = int(T * train_frac)
        split_desc = f"{train_frac:.0%} / {1-train_frac:.0%}"

    if verbose:
        print(f"\nSplit: {train_size} train / {T - train_size} test  ({split_desc})")
        print(f"  Train: {z_df.index[0].date()} -> {z_df.index[train_size-1].date()}")
        print(f"  Test:  {z_df.index[train_size].date()} -> {z_df.index[-1].date()}")

    # Standardize on TRAIN stats only -- see module docstring.
    X_train_raw = X_df.iloc[:train_size].values
    X_mean = X_train_raw.mean(axis=0)
    X_std = X_train_raw.std(axis=0) + 1e-8
    X_all_std = (X_df.values - X_mean) / X_std

    # Apply PCA if requested
    if pca_components is not None:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components)
        X_train_std = X_all_std[:train_size]
        pca.fit(X_train_std)
        X_all_std = pca.transform(X_all_std)
        if verbose:
            print(f"\nPCA: {X_df.shape[1]} -> {pca_components} components")
            print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

    z_full = torch.tensor(z_df.values, dtype=torch.float32)
    X_full = torch.tensor(X_all_std, dtype=torch.float32)

    return {
        'config': config,
        'paths': paths,
        'z_df': z_df,
        'X_df': X_df,
        'z_full': z_full,
        'X_full': X_full,
        'train_size': train_size,
        'T': T,
        'dates': z_df.index,
        'tickers': list(z_df.columns),
        'feature_columns': list(X_df.columns),
        'X_mean': X_mean,
        'X_std': X_std,
    }


def fit_all_train_only(split, n_iter=500, lr=0.01, lambda_reg=None,
                        skip_adcc=False, verbose=True):
    """
    Fit every model on the training window only, then run each recursion
    forward over the FULL sample (warm-starting the test period from
    training history) and slice out the test-period R_seq.

    Q_bar is computed from TRAIN data only -- it is a parameter of the
    model (the unconditional correlation target), so estimating it on the
    full sample would leak test information.
    """
    z_full = split['z_full']
    X_full = split['X_full']
    train_size = split['train_size']
    z_df = split['z_df']

    z_train = z_full[:train_size]
    X_train = X_full[:train_size]
    Q_bar_train = compute_Q_bar(z_train)

    if lambda_reg is None:
        lambda_grid = split['config']['models']['topo_reg']['lambda_grid']
        lambda_reg = max(lambda_grid) if False else 4000
        # 4000 is the value selected by the full-sample lambda search
        # (see dcc_topo_reg_results.npy). Reusing the same lambda here is a
        # deliberate simplification: strictly, lambda should be reselected
        # on the training split alone, since choosing it on the full sample
        # is itself a mild form of lookahead. Reselecting requires the full
        # grid search (GPU-batched, slow) and is listed as a follow-up.
        if lambda_reg not in lambda_grid and verbose:
            print(f"\n  NOTE: lambda={lambda_reg} is not in config's lambda_grid "
                  f"{lambda_grid} -- verify this is the intended value.")

    device = torch.device('cpu')
    out = {}
    timings = {}

    # --- Baseline DCC ---
    if verbose:
        print("\n[1] Fitting baseline DCC (train only)...")
    t0 = time.time()
    a_base, b_base, _, ll_hist_base = fit_dcc_baseline(
        z_df.iloc[:train_size], n_iter=n_iter, lr=lr, verbose=False
    )
    timings['baseline'] = time.time() - t0
    with torch.no_grad():
        R_base_full, _ = dcc_recursion_torch(
            z_full, torch.tensor(a_base), torch.tensor(b_base), Q_bar_train
        )
    out['Baseline DCC'] = R_base_full[train_size:].numpy()
    if verbose:
        print(f"    a={a_base:.6f} b={b_base:.6f}  ll_train={ll_hist_base[-1]:.2f}  "
              f"({timings['baseline']:.1f}s)")

    # --- aDCC ---
    if not skip_adcc:
        if verbose:
            print("\n[2] Fitting aDCC (train only)...")
        t0 = time.time()
        a_ad, b_ad, g_ad, _, ll_hist_ad, delta_ad = fit_dcc_adcc(
            z_df.iloc[:train_size], n_iter=n_iter, lr=lr, verbose=False
        )
        timings['adcc'] = time.time() - t0
        # Recompute Nbar on train only, consistent with Q_bar_train
        N_bar_train = compute_N_bar(z_train)
        with torch.no_grad():
            R_ad_full, _ = adcc_recursion(
                z_full,
                torch.tensor(a_ad, dtype=torch.float32),
                torch.tensor(b_ad, dtype=torch.float32),
                torch.tensor(g_ad, dtype=torch.float32),
                Q_bar_train, N_bar_train
            )
        out['aDCC'] = R_ad_full[train_size:].numpy()
        if verbose:
            print(f"    a={a_ad:.6f} b={b_ad:.6f} g={g_ad:.6f}  "
                  f"ll_train={ll_hist_ad[-1]:.2f}  ({timings['adcc']:.1f}s)")

    # --- TopoDCC unregularized ---
    if verbose:
        print("\n[3] Fitting TopoDCC unregularized (train only, lambda=0)...")
    t0 = time.time()
    model_unreg, ll_hist_unreg, _, _ = fit_dcc_topo_reg(
        z_train, X_train, Q_bar_train,
        lambda_l2=0.0, n_iter=n_iter, lr=lr,
        patience=30, min_delta=1e-4, device=device, verbose=False
    )
    timings['topo_unreg'] = time.time() - t0
    with torch.no_grad():
        a_seq, b_seq = model_unreg(X_full)
        R_unreg_full, _ = dcc_topo_recursion(z_full, a_seq, b_seq, Q_bar_train)
    out['TopoDCC (unreg)'] = R_unreg_full[train_size:].numpy()
    if verbose:
        print(f"    ll_train={ll_hist_unreg[-1]:.2f}  "
              f"a_t std={a_seq.std().item():.5f}  b_t std={b_seq.std().item():.5f}  "
              f"({timings['topo_unreg']:.1f}s)")

    # --- TopoDCC regularized ---
    if verbose:
        print(f"\n[4] Fitting TopoDCC regularized (train only, lambda={lambda_reg})...")
    t0 = time.time()
    model_reg, ll_hist_reg, _, _ = fit_dcc_topo_reg(
        z_train, X_train, Q_bar_train,
        lambda_l2=lambda_reg, n_iter=n_iter, lr=lr,
        patience=30, min_delta=1e-4, device=device, verbose=False
    )
    timings['topo_reg'] = time.time() - t0
    with torch.no_grad():
        a_seq_r, b_seq_r = model_reg(X_full)
        R_reg_full, _ = dcc_topo_recursion(z_full, a_seq_r, b_seq_r, Q_bar_train)
    out['TopoDCC (reg)'] = R_reg_full[train_size:].numpy()
    if verbose:
        print(f"    ll_train={ll_hist_reg[-1]:.2f}  "
              f"a_t std={a_seq_r.std().item():.5f}  b_t std={b_seq_r.std().item():.5f}  "
              f"({timings['topo_reg']:.1f}s)")
        print(f"\n  Total fitting time: {sum(timings.values()):.1f}s")

    return out, timings


def oos_portfolio_backtest(R_seq_by_model, split, sigma_df, periods_per_year=252):
    """
    Min-variance portfolio backtest restricted to the test period.

    Unlike backtest.run_backtest (which aligns a longer baseline series
    against a shorter topology one), everything here is already on
    identical test-period dates, so no alignment/slicing logic applies --
    that is why this is a separate function rather than a flag on the
    existing one.
    """
    train_size = split['train_size']
    test_dates = split['dates'][train_size:]
    tickers = split['tickers']

    log_returns = pd.read_parquet(split['paths']['log_returns'])
    returns_test = log_returns.loc[test_dates, tickers].to_numpy(dtype=np.float64)

    sigma_test = sigma_df.loc[test_dates, tickers].to_numpy(dtype=np.float64)

    N = len(tickers)
    T_test = len(test_dates)

    rows = []
    port_returns = {}
    weights_by_model = {}

    for name, R_seq in R_seq_by_model.items():
        H_seq = reconstruct_covariance(sigma_test, R_seq)
        W = min_variance_weights_seq(H_seq)
        r = portfolio_returns(W, returns_test)

        weights_by_model[name] = W
        port_returns[name] = r
        rows.append({
            'variant': name,
            'ann_return_%': np.mean(r) * periods_per_year * 100,
            'ann_vol_%': annualize_vol(r, periods_per_year) * 100,
            'sharpe': annualize_sharpe(r, periods_per_year),
            'turnover': turnover(W),
        })

    # Equal-weight benchmark
    W_eq = equal_weights_seq(T_test, N)
    r_eq = portfolio_returns(W_eq, returns_test)
    port_returns['Equal-weight (1/N)'] = r_eq
    weights_by_model['Equal-weight (1/N)'] = W_eq
    rows.append({
        'variant': 'Equal-weight (1/N)',
        'ann_return_%': np.mean(r_eq) * periods_per_year * 100,
        'ann_vol_%': annualize_vol(r_eq, periods_per_year) * 100,
        'sharpe': annualize_sharpe(r_eq, periods_per_year),
        'turnover': 0.0,
    })

    summary = pd.DataFrame(rows).set_index('variant')
    return summary, port_returns, weights_by_model


def run_dm_tests(R_seq_by_model, split, port_returns, verbose=True):
    """
    Diebold-Mariano tests on the test period.

    Pairs tested (each against the relevant benchmark rather than all
    against all -- the interesting comparisons are TopoDCC vs the two
    baselines, and reg vs unreg to quantify what regularization costs):
      - TopoDCC (unreg) vs Baseline DCC
      - TopoDCC (reg)   vs Baseline DCC
      - TopoDCC (unreg) vs aDCC          (the harder benchmark)
      - TopoDCC (reg)   vs TopoDCC (unreg)
    """
    train_size = split['train_size']
    z_test = split['z_full'][train_size:].numpy()

    pairs = [
        ('TopoDCC (unreg)', 'Baseline DCC'),
        ('TopoDCC (reg)', 'Baseline DCC'),
        ('TopoDCC (unreg)', 'aDCC'),
        ('TopoDCC (reg)', 'TopoDCC (unreg)'),
    ]

    results = {}
    for m1, m2 in pairs:
        if m1 not in R_seq_by_model or m2 not in R_seq_by_model:
            continue
        if verbose:
            print(f"\n{'='*66}")
            print(f"DM: {m1}  vs  {m2}")
            print('='*66)
        results[f"{m1} vs {m2}"] = dm_test_all_losses(
            R_seq_by_model[m1], R_seq_by_model[m2], z_test,
            name_1=m1, name_2=m2, verbose=verbose
        )

        # Economic-loss version: squared realized portfolio return
        if m1 in port_returns and m2 in port_returns:
            if verbose:
                print(f"\n[PORTFOLIO squared-return loss]")
            stat, p = dm_test(
                portfolio_sq_loss(port_returns[m1]),
                portfolio_sq_loss(port_returns[m2]),
                name_1=m1, name_2=m2, verbose=verbose
            )
            results[f"{m1} vs {m2}"]['portfolio_sq'] = {'dm_stat': stat, 'p_value': p}

    return results


# Standard crisis windows, matching src/evaluation/stats_tests.py exactly so
# a_t/b_t crisis-period behavior and OOS crisis-period forecast accuracy are
# reported on identical date ranges rather than two slightly different
# definitions drifting apart over time.
CRISIS_WINDOWS = {
    'GFC       (2008-07 to 2009-03)': ('2008-07-01', '2009-03-31'),
    'EU Debt   (2011-07 to 2012-01)': ('2011-07-01', '2012-01-31'),
    'COVID     (2020-02 to 2020-06)': ('2020-02-01', '2020-06-30'),
    'Rates     (2022-01 to 2022-12)': ('2022-01-01', '2022-12-31'),
}


def run_crisis_dm_tests(R_seq_by_model, split, verbose=True):
    """
    Repeats the QLIKE/Frobenius DM tests restricted to whichever crisis
    windows actually overlap the TEST period.

    WHY THIS EXISTS
      Souto (2023, "Topological Tail Dependence") finds persistent
      homology's forecasting advantage over baseline models is
      concentrated specifically in a held-out COVID subsample, not
      spread evenly across the full test period -- full-sample DM tests
      in that paper are often insignificant while the same comparison
      restricted to the 2020 crisis is significant. A full-sample-only
      DM test, which is all this project has run until now, could
      therefore miss a real effect that only shows up during stress
      periods, or dilute it into non-significance by averaging against
      long stretches of calm markets where topology plausibly adds
      nothing.

    ONLY windows with test-period overlap are actually tested -- with the
    default 80/20 split on this project's sample (2006-2025), the test
    period lands around 2022-2025, so GFC/EU-Debt/COVID fall entirely in
    TRAINING data and have zero test-period observations. Silently
    running a "crisis DM test" against zero dates would produce a
    meaningless or errored result; this function checks overlap first
    and reports explicitly which windows were skipped and why, rather
    than papering over the gap. Use --split-date 2019-12-31 (or similar)
    to actually place COVID inside the test period if that is the
    comparison you want.
    """
    train_size = split['train_size']
    test_dates = split['dates'][train_size:]
    z_test_full = split['z_full'][train_size:].numpy()

    results = {}

    if verbose:
        print(f"\nTest period: {test_dates[0].date()} -> {test_dates[-1].date()}")

    for label, (start, end) in CRISIS_WINDOWS.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        mask = (test_dates >= start_ts) & (test_dates <= end_ts)
        n_overlap = int(mask.sum())

        if n_overlap == 0:
            if verbose:
                print(f"\n[{label}] 0 test-period dates overlap this window -- "
                      f"entirely inside training data with the current split. Skipped.")
            continue

        if n_overlap < 20:
            if verbose:
                print(f"\n[{label}] only {n_overlap} test-period dates overlap -- "
                      f"too few for a reliable DM test (partial window at the "
                      f"train/test boundary). Skipped.")
            continue

        if verbose:
            print(f"\n{'='*66}")
            print(f"[{label}]  {n_overlap} overlapping test-period dates")
            print('='*66)

        z_crisis = z_test_full[mask]
        window_results = {}

        pairs = [
            ('TopoDCC (unreg)', 'Baseline DCC'),
            ('TopoDCC (reg)', 'Baseline DCC'),
        ]
        for m1, m2 in pairs:
            if m1 not in R_seq_by_model or m2 not in R_seq_by_model:
                continue
            R1_crisis = R_seq_by_model[m1][mask]
            R2_crisis = R_seq_by_model[m2][mask]

            if verbose:
                print(f"\nDM: {m1} vs {m2}  (crisis subperiod only)")
            l1 = qlike_loss_seq(R1_crisis, z_crisis)
            l2 = qlike_loss_seq(R2_crisis, z_crisis)
            stat, p = dm_test(l1, l2, name_1=m1, name_2=m2, verbose=verbose)
            window_results[f"{m1} vs {m2}"] = {'dm_stat': stat, 'p_value': p, 'n': n_overlap}

        results[label] = window_results

    if not results and verbose:
        print("\nNo crisis window had sufficient test-period overlap -- "
              "consider --split-date to move the test period earlier "
              "(e.g. --split-date 2019-12-31 to capture COVID).")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Out-of-sample evaluation: train/test split, portfolio backtest, DM tests"
    )
    parser.add_argument('--train-frac', type=float, default=0.8,
                         help="Fraction of the sample used for training (chronological split). "
                              "Ignored if --split-date is given.")
    parser.add_argument('--split-date', default=None,
                         help="Exact train/test cutoff date (e.g. 2019-12-31), overriding "
                              "--train-frac. Needed to place specific crisis windows (COVID, "
                              "GFC, etc.) inside the test period -- see prepare_split docstring.")
    parser.add_argument('--n-iter', type=int, default=500,
                         help="Adam iterations per fit.")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lambda-reg', type=float, default=None,
                         help="L2 penalty for the regularized TopoDCC. Defaults to 4000 "
                              "(the value selected by the full-sample lambda search).")
    parser.add_argument('--features', default='lpnorm',
                         choices=['lpnorm', 'landscape', 'pi',
                                  'lpnorm_speed', 'lpnorm_levels_speed',
                                  'landscape_speed', 'landscape_levels_speed'],
                         help="'_speed' variants use rate-of-change instead of levels "
                              "(same dimensionality as the source); '_levels_speed' "
                              "concatenates both (double dimensionality)")
    parser.add_argument('--window', type=int, default=None,
                         help='TDA window size override (triggers tda_pipeline recomputation)')
    parser.add_argument('--pca', type=int, default=None,
                         help='apply PCA to reduce features to N components (train-set-fit only, '
                              'no lookahead). Only meaningful with --features landscape or pi.')
    parser.add_argument('--skip-adcc', action='store_true',
                         help="Skip the aDCC benchmark (faster).")
    parser.add_argument('--out', default='data/processed/oos_evaluation_results.npy')
    args = parser.parse_args()

    print("="*66)
    print("OUT-OF-SAMPLE EVALUATION")
    print("="*66)

    split = prepare_split(train_frac=args.train_frac, features=args.features,
                          pca_components=args.pca, window=args.window,
                          split_date=args.split_date)

    R_seq_by_model, timings = fit_all_train_only(
        split, n_iter=args.n_iter, lr=args.lr,
        lambda_reg=args.lambda_reg, skip_adcc=args.skip_adcc
    )

    sigma_path = split['paths'].get('garch_sigma', 'data/processed/garch_sigma.parquet')
    if not os.path.exists(sigma_path):
        raise FileNotFoundError(
            f"{sigma_path} not found -- run `python -m src.volatility.garch` first. "
            "The portfolio step needs the GARCH conditional volatilities to "
            "reconstruct H_t = D_t R_t D_t."
        )
    sigma_df = pd.read_parquet(sigma_path)

    print("\n" + "="*66)
    print("OUT-OF-SAMPLE PORTFOLIO BACKTEST (test period only)")
    print("="*66 + "\n")
    summary, port_returns, weights = oos_portfolio_backtest(R_seq_by_model, split, sigma_df)
    print(summary.round(4).to_string())

    print("\n" + "="*66)
    print("DIEBOLD-MARIANO TESTS (test period only)")
    print("="*66)
    dm_results = run_dm_tests(R_seq_by_model, split, port_returns)

    print("\n" + "="*66)
    print("CRISIS-SUBPERIOD DIEBOLD-MARIANO TESTS")
    print("="*66)
    crisis_dm_results = run_crisis_dm_tests(R_seq_by_model, split)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, {
        'summary': summary,
        'dm_results': dm_results,
        'crisis_dm_results': crisis_dm_results,
        'timings': timings,
        'train_size': split['train_size'],
        'train_frac': args.train_frac,
        'split_date': args.split_date,
        'test_dates': split['dates'][split['train_size']:],
        'R_seq_test': R_seq_by_model,
        'portfolio_returns': port_returns,
    }, allow_pickle=True)
    print(f"\n\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
