# -*- coding: utf-8 -*-
"""
Asymmetric DCC (aDCC) -- Cappiello, Engle & Sheppard (2006).

WHY THIS EXISTS
  Beating vanilla DCC is a low bar: vanilla DCC has two free parameters
  and no mechanism for the well-documented empirical fact that
  correlations rise more after joint negative shocks than after joint
  positive ones (the "correlation leverage" effect). If TopoDCC's edge
  over vanilla DCC is really just picking up asymmetry that a standard
  extension already captures, that is important to know BEFORE an
  examiner asks -- so aDCC is the benchmark the thesis actually needs.

THE MODEL
  Vanilla DCC:
      Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}

  aDCC adds a third term, active only on joint negative returns:
      n_t   = z_t * 1[z_t < 0]              (elementwise indicator)
      Nbar  = E[n_t n_t']
      Q_t   = (Qbar - a Qbar - b Qbar - g Nbar)
              + a z_{t-1} z_{t-1}'
              + g n_{t-1} n_{t-1}'
              + b Q_{t-1}

  g > 0 means correlations get an extra kick when both assets fell.
  Setting g = 0 recovers vanilla DCC exactly, so aDCC nests the baseline
  and a likelihood-ratio test between them is valid (1 restriction).

STATIONARITY / POSITIVE DEFINITENESS
  The intercept (Qbar - a Qbar - b Qbar - g Nbar) must stay positive
  definite. The sufficient condition used here is the standard one:
      a + b + delta * g < 1
  where delta is the maximum eigenvalue of Qbar^{-1/2} Nbar Qbar^{-1/2}.
  Rather than clamping after the fact (which puts a kink in the gradient
  and can silently produce non-PSD Q_t), the parameters are reparameterized
  so the constraint holds by construction for every value the raw
  optimizer variables can take -- the same approach dcc_baseline.py uses
  for a + b < 1:

      s      = sigmoid(total_raw) * max_sum        (total persistence)
      p_a    = sigmoid(split_a_raw)                (share going to a)
      p_g    = sigmoid(split_g_raw)                (share of remainder to g)
      a      = s * p_a
      rest   = s * (1 - p_a)
      g      = rest * p_g / delta                  (scaled so delta*g fits)
      b      = rest * (1 - p_g)

  -> a + b + delta*g = s < max_sum, guaranteed, no clamping, no masking.

USAGE
  python -m src.models.dcc_adcc
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def compute_Q_bar(z_t):
    T = z_t.shape[0]
    return (z_t.T @ z_t) / T


def compute_N_bar(z_t):
    """
    Nbar = E[n_t n_t'] where n_t = z_t * 1[z_t < 0].
    This is the asymmetry analogue of Qbar.
    """
    n_t = z_t * (z_t < 0).to(z_t.dtype)
    T = n_t.shape[0]
    return (n_t.T @ n_t) / T


def compute_delta(Q_bar, N_bar):
    """
    delta = max eigenvalue of Qbar^{-1/2} Nbar Qbar^{-1/2}.

    Used to scale the asymmetry parameter g so that the sufficient
    condition a + b + delta*g < 1 can be imposed by construction.
    Computed once from data (not a learned parameter) and detached --
    it is a property of the sample, not something to optimize through.
    """
    with torch.no_grad():
        evals_q, evecs_q = torch.linalg.eigh(Q_bar)
        evals_q = torch.clamp(evals_q, min=1e-10)
        Q_inv_sqrt = evecs_q @ torch.diag(evals_q.pow(-0.5)) @ evecs_q.T
        M = Q_inv_sqrt @ N_bar @ Q_inv_sqrt
        delta = torch.linalg.eigvalsh(M).max()
        # Guard: if Nbar is degenerate delta could be ~0, which would make
        # g unbounded. Floor it so the reparameterization stays well-posed.
        delta = torch.clamp(delta, min=1e-6)
    return delta


@torch.jit.script
def adcc_recursion(z_t, a, b, g, Q_bar, N_bar):
    """
    aDCC recursion. Returns (R_seq, ll).

    JIT-scripted for the same reason the TopoDCC recursion is: this is a
    Python-level loop over T timesteps and it dominates fit time.
    """
    T, N = z_t.shape
    n_t = z_t * (z_t < 0).to(z_t.dtype)

    intercept = Q_bar - a * Q_bar - b * Q_bar - g * N_bar

    Q_t = Q_bar.clone()
    Q_seq = torch.zeros(T, N, N, dtype=z_t.dtype, device=z_t.device)
    Q_seq[0] = Q_t

    for t in range(1, T):
        z_outer = torch.outer(z_t[t - 1], z_t[t - 1])
        n_outer = torch.outer(n_t[t - 1], n_t[t - 1])
        Q_t = intercept + a * z_outer + g * n_outer + b * Q_t
        Q_seq[t] = Q_t

    diag_Q = torch.sqrt(torch.diagonal(Q_seq, dim1=1, dim2=2))
    R_seq = Q_seq / torch.einsum('ti,tj->tij', diag_Q, diag_Q)

    sign, log_det = torch.linalg.slogdet(R_seq)
    R_inv = torch.linalg.inv(R_seq)
    mahal = torch.sum(z_t * (R_inv @ z_t.unsqueeze(-1)).squeeze(-1), dim=1)
    ll = -0.5 * (log_det.sum() + mahal.sum())

    return R_seq, ll


def fit_dcc_adcc(garch_residuals_df, n_iter=500, lr=0.01,
                 patience=30, min_delta=1e-4, verbose=True):
    """
    Fit aDCC by maximum likelihood (Adam, same optimizer/settings as the
    other models in this pipeline so the comparison isn't confounded by
    optimizer differences).

    Early stopping mirrors fit_dcc_topo_reg: stop once loss hasn't improved
    by min_delta for `patience` iterations, and restore the best weights
    rather than the last ones.

    Returns: (a, b, g, R_seq, ll_history, delta)
    """
    z_t = torch.tensor(garch_residuals_df.values, dtype=torch.float32)
    T, N = z_t.shape

    Q_bar = compute_Q_bar(z_t)
    N_bar = compute_N_bar(z_t)
    delta = compute_delta(Q_bar, N_bar)

    max_sum = 0.9998  # matches dcc_baseline.py / TopoDCC for comparability

    # See module docstring for the reparameterization rationale.
    total_raw   = nn.Parameter(torch.tensor(4.0,  dtype=torch.float32))
    split_a_raw = nn.Parameter(torch.tensor(-3.5, dtype=torch.float32))
    split_g_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

    optimizer = torch.optim.Adam([total_raw, split_a_raw, split_g_raw], lr=lr)
    ll_history = []

    best_loss = float('inf')
    best_state = None
    n_bad = 0

    def unpack():
        s    = torch.sigmoid(total_raw) * max_sum
        p_a  = torch.sigmoid(split_a_raw)
        p_g  = torch.sigmoid(split_g_raw)
        a    = s * p_a
        rest = s * (1 - p_a)
        g    = rest * p_g / delta
        b    = rest * (1 - p_g)
        return a, b, g

    for i in range(n_iter):
        optimizer.zero_grad()
        a, b, g = unpack()
        R_seq, ll = adcc_recursion(z_t, a, b, g, Q_bar, N_bar)
        loss = -ll
        loss.backward()
        optimizer.step()
        ll_history.append(ll.item())

        if loss.item() < best_loss - min_delta:
            best_loss = loss.item()
            best_state = (total_raw.detach().clone(),
                          split_a_raw.detach().clone(),
                          split_g_raw.detach().clone())
            n_bad = 0
        else:
            n_bad += 1
            if n_bad >= patience:
                if verbose:
                    print(f"  Early stop at iter {i} (no improvement for {patience} iters)")
                break

        if verbose and (i % 50 == 0):
            print(f"  Iter {i:4d} | a={a.item():.6f} | b={b.item():.6f} | "
                  f"g={g.item():.6f} | a+b+delta*g={(a+b+delta*g).item():.6f} | "
                  f"ll={ll.item():.2f}")

    # Restore best weights before producing final outputs
    if best_state is not None:
        with torch.no_grad():
            total_raw.copy_(best_state[0])
            split_a_raw.copy_(best_state[1])
            split_g_raw.copy_(best_state[2])

    with torch.no_grad():
        a_final, b_final, g_final = unpack()
        R_seq, ll_final = adcc_recursion(z_t, a_final, b_final, g_final, Q_bar, N_bar)

    if verbose:
        print(f"\n  Final: a={a_final.item():.6f}  b={b_final.item():.6f}  "
              f"g={g_final.item():.6f}")
        print(f"  Stationarity: a+b+delta*g = {(a_final+b_final+delta*g_final).item():.6f} "
              f"(delta={delta.item():.4f})")
        print(f"  ll = {ll_final.item():.2f}")

    return (a_final.item(), b_final.item(), g_final.item(),
            R_seq.detach().numpy(), ll_history, delta.item())


def lr_test_vs_baseline(ll_adcc, ll_baseline):
    """
    Likelihood-ratio test: aDCC (3 params) vs vanilla DCC (2 params).
    aDCC nests DCC at g=0, so LR ~ chi2(1) under the null that g=0.

    Returns (LR_statistic, p_value).
    """
    from scipy import stats
    LR = 2 * (ll_adcc - ll_baseline)
    p = stats.chi2.sf(LR, df=1)
    return LR, p


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.data.loader import load_config

    config = load_config()
    paths = config['paths']
    # aDCC is a baseline variant -- reuse the baseline model's iter/lr config
    model_cfg = config['models']['baseline']

    garch_residuals = pd.read_parquet(paths['garch_residuals'])

    print(f"Fitting aDCC on {garch_residuals.shape[0]} obs x {garch_residuals.shape[1]} assets...")
    a, b, g, R_seq, ll_history, delta = fit_dcc_adcc(
        garch_residuals,
        n_iter=model_cfg['n_iter'],
        lr=model_cfg['lr'],
        verbose=True
    )

    out_path = paths.get('dcc_adcc', 'data/processed/dcc_adcc_results.npy')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, {
        'a': a,
        'b': b,
        'g': g,
        'delta': delta,
        'll_final': ll_history[-1],
        'll_history': ll_history,
        'R_seq': R_seq,
    })
    print(f"\nSaved to {out_path}")

    # If the vanilla baseline has already been fit, report the LR test --
    # this is the number that says whether asymmetry alone was worth adding.
    baseline_path = paths['dcc_baseline']
    if os.path.exists(baseline_path):
        baseline = np.load(baseline_path, allow_pickle=True).item()
        LR, p = lr_test_vs_baseline(ll_history[-1], baseline['ll_final'])
        print(f"\nLR test (aDCC vs vanilla DCC):  LR = {LR:.2f}, p = {p:.4e}")
        print(f"  vanilla DCC ll = {baseline['ll_final']:.2f}")
        print(f"  aDCC        ll = {ll_history[-1]:.2f}")
