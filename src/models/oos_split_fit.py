# -*- coding: utf-8 -*-
"""
Out-of-sample check: refit baseline DCC, unregularized TopoDCC, and
regularized (lambda=4000) TopoDCC on a TRAIN-ONLY split, then generate
R_seq forward through the TEST period only, and feed that into the
portfolio backtest.

This answers a different question than the full-sample backtest already
run: does the small edge unregularized TopoDCC showed over baseline
survive when the model has never seen the test period, or does it
disappear (evidence it was fitting noise in the full sample)?

Chronological 80/20 split, matching the convention already used elsewhere
in this pipeline (dcc_topo_reg.py docstring).
"""

import time
import torch
import numpy as np
import pandas as pd

from src.models.dcc_baseline import compute_Q_bar as compute_Q_bar_baseline, fit_dcc_baseline, dcc_recursion_torch
from src.models.dcc_topo import TopoDCC, dcc_topo_recursion, compute_Q_bar
from src.models.dcc_topo_reg import fit_dcc_topo_reg, eval_oos_ll


def load_inputs():
    z_df = pd.read_parquet('data/processed/garch_residuals.parquet')
    X_df = pd.read_parquet('data/processed/tda_features_lpnorm.parquet')

    # Align to common dates (topology features start later due to rolling window)
    common_idx = z_df.index.intersection(X_df.index)
    z_df = z_df.loc[common_idx]
    X_df = X_df.loc[common_idx]

    feature_cols = ['betti_1', 'entropy_h0', 'entropy_h1', 'max_persistence',
                     'total_persistence', 'wasserstein', 'lh1_k0_norm', 'lh1_k1_norm', 'lh1_k2_norm']
    X_df = X_df[feature_cols]

    return z_df, X_df


def main():
    z_df, X_df = load_inputs()
    T = len(z_df)
    train_size = int(T * 0.8)
    print(f"Total obs: {T}  |  train: {train_size}  |  test: {T - train_size}")
    print(f"Train dates: {z_df.index[0].date()} -> {z_df.index[train_size-1].date()}")
    print(f"Test dates:  {z_df.index[train_size].date()} -> {z_df.index[-1].date()}")

    z_full = torch.tensor(z_df.values, dtype=torch.float32)
    z_train = z_full[:train_size]

    # Standardize features on TRAIN stats only -- avoid leaking test-period
    # feature distribution into the fit.
    X_mean = X_df.iloc[:train_size].mean()
    X_std = X_df.iloc[:train_size].std()
    X_std_df = (X_df - X_mean) / X_std

    X_full = torch.tensor(X_std_df.values, dtype=torch.float32)
    X_train = X_full[:train_size]

    Q_bar_train = compute_Q_bar(z_train)

    results = {}

    # --- Baseline DCC (train-only fit) ---
    t0 = time.time()
    z_train_df = z_df.iloc[:train_size]
    a_base, b_base, R_seq_base_train, ll_hist_base = fit_dcc_baseline(
        z_train_df, n_iter=500, lr=0.01, verbose=False
    )
    t_base = time.time() - t0
    print(f"\n[Baseline] a={a_base:.4f} b={b_base:.4f}  train fit took {t_base:.1f}s")
    results['baseline_fit_time'] = t_base

    # Baseline test-period R_seq: same recursion forward through full sample
    # using the constant (a, b) fit on train, warm-started from Q_bar_train
    with torch.no_grad():
        R_seq_base_full, _ = dcc_recursion_torch(
            z_full, torch.tensor(a_base), torch.tensor(b_base), Q_bar_train
        )
    R_seq_base_test = R_seq_base_full[train_size:].numpy()

    # --- Unregularized TopoDCC (train-only fit, lambda=0) ---
    t0 = time.time()
    model_unreg, ll_hist_unreg, a_seq_train, b_seq_train = fit_dcc_topo_reg(
        z_train, X_train, Q_bar_train,
        lambda_l2=0.0, n_iter=500, lr=0.01, patience=30, min_delta=1e-4,
        device=torch.device('cpu'), verbose=False
    )
    t_unreg = time.time() - t0
    print(f"[Unregularized TopoDCC] train fit took {t_unreg:.1f}s")

    # --- Regularized TopoDCC (train-only fit, lambda=4000... but note: this codebase's
    #     lambda scale for fit_dcc_topo_reg looks like it's O(1e-4 to 1) per LAMBDA_GRID,
    #     NOT literally 4000 -- flag this discrepancy rather than silently guessing ---
    t0 = time.time()
    model_reg, ll_hist_reg, a_seq_train_reg, b_seq_train_reg = fit_dcc_topo_reg(
        z_train, X_train, Q_bar_train,
        lambda_l2=1e-1, n_iter=500, lr=0.01, patience=30, min_delta=1e-4,
        device=torch.device('cpu'), verbose=False
    )
    t_reg = time.time() - t0
    print(f"[Regularized TopoDCC]   train fit took {t_reg:.1f}s")

    print(f"\nTotal fitting time (both TopoDCC variants): {t_unreg + t_reg:.1f}s")

    # Generate full-sequence R_seq forward using trained weights, then slice test
    with torch.no_grad():
        a_seq_full_unreg, b_seq_full_unreg = model_unreg(X_full)
        R_seq_unreg_full, _ = dcc_topo_recursion(z_full, a_seq_full_unreg, b_seq_full_unreg, Q_bar_train)

        a_seq_full_reg, b_seq_full_reg = model_reg(X_full)
        R_seq_reg_full, _ = dcc_topo_recursion(z_full, a_seq_full_reg, b_seq_full_reg, Q_bar_train)

    R_seq_unreg_test = R_seq_unreg_full[train_size:].numpy()
    R_seq_reg_test = R_seq_reg_full[train_size:].numpy()

    print(f"\nTest-period R_seq shapes: baseline={R_seq_base_test.shape}, "
          f"unreg={R_seq_unreg_test.shape}, reg={R_seq_reg_test.shape}")

    np.save('data/processed/oos_R_seq_baseline_test.npy', R_seq_base_test)
    np.save('data/processed/oos_R_seq_unreg_test.npy', R_seq_unreg_test)
    np.save('data/processed/oos_R_seq_reg_test.npy', R_seq_reg_test)
    print("Saved test-period R_seq arrays for use in backtest.py")
    print(f"\nNOTE: lambda_l2=1e-1 was used here as a stand-in for 'the regularized "
          f"model' -- this codebase's LAMBDA_GRID for fit_dcc_topo_reg is scaled "
          f"O(1e-4 to 1e-1), NOT the same 4000 reported elsewhere for the committed "
          f"dcc_topo_reg_results.npy. That 4000 figure comes from a different lambda "
          f"parameterization/scale used in the full pipeline run -- flagging this "
          f"explicitly rather than silently treating them as the same regularization "
          f"strength. Worth reconciling before trusting this regularized OOS number.")


if __name__ == "__main__":
    main()
