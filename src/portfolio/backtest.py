# -*- coding: utf-8 -*-
"""
Portfolio application chapter: backtest three variants against each other.

    1. Baseline DCC minimum-variance
    2. TopoDCC (topology-driven) minimum-variance
    3. Equal-weight (1/N) benchmark

Convention: H_t (and hence w_t) is the conditional covariance forecast for
period t, formed from information available through t-1 (that's what the
DCC recursion Q_t = (1-a-b)Qbar + a*z_{t-1}z_{t-1}' + b*Q_{t-1} already
encodes). So w_t is applied to the realized return r_t at the *same* index
t — this is standard in the DCC portfolio-application literature (Engle),
not a lookahead: w_t only ever uses information dated before t.

Metrics reported, and why each one is here:
  - Realized portfolio volatility: the direct test of the thesis claim —
    did a better correlation estimate produce genuinely lower realized risk?
  - Sharpe ratio: risk-adjusted return, standard comparison point.
  - Turnover: mean absolute weight change per period. A covariance-driven
    strategy that trades constantly on every wiggle in a(t)/b(t) isn't
    actually better once transaction costs are considered — this is the
    "is it real-world usable" check, not just an academic curiosity.
"""

import numpy as np
import pandas as pd

from src.portfolio.covariance import align_sigma_to_R, reconstruct_covariance
from src.portfolio.weights import min_variance_weights_seq, equal_weights_seq


def portfolio_returns(weights, returns):
    """
    weights: (T, N), returns: (T, N) aligned by row (period t weight applied
    to period t return, per the module-level convention above).
    Returns (T,) array of realized portfolio returns.
    """
    return np.sum(weights * returns, axis=1)


def turnover(weights):
    """
    Mean L1 weight change per period: mean_t sum_i |w_{i,t} - w_{i,t-1}|.
    First period has no prior weight to compare against, so it's excluded.
    """
    diffs = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    return diffs.mean()


def annualize_vol(daily_returns, periods_per_year=252):
    return np.std(daily_returns, ddof=1) * np.sqrt(periods_per_year)


def annualize_sharpe(daily_returns, periods_per_year=252, rf=0.0):
    excess = daily_returns - rf / periods_per_year
    mu = np.mean(excess) * periods_per_year
    sigma = np.std(excess, ddof=1) * np.sqrt(periods_per_year)
    return mu / sigma if sigma > 0 else np.nan


def run_backtest(log_returns_df, sigma_df, R_seq_baseline, R_seq_topo,
                  ridge=1e-8, long_only=False, periods_per_year=252):
    """
    Runs all three variants over the topology model's date range (the
    shorter of the two, since baseline has a longer history — see the
    249-day warm-up gap from the rolling TDA window). Baseline is sliced
    down to the same dates so all three portfolios are compared on
    identical realized returns.

    Returns a dict with per-variant weights, realized returns, and a
    summary DataFrame of the headline metrics.
    """
    N = log_returns_df.shape[1]
    tickers = list(log_returns_df.columns)

    # Align baseline R_seq down to the topology model's shorter window —
    # baseline R_seq is assumed to be the trailing dates matching topo's.
    T_topo = R_seq_topo.shape[0]
    if R_seq_baseline.shape[0] < T_topo:
        raise ValueError("Baseline R_seq is shorter than topology R_seq — unexpected.")
    R_seq_baseline_aligned = R_seq_baseline[-T_topo:]

    sigma_topo, dates = align_sigma_to_R(sigma_df, R_seq_topo, tickers=tickers)
    # baseline uses the same sigma window since it's the same GARCH stage output
    sigma_baseline = sigma_topo

    H_baseline = reconstruct_covariance(sigma_baseline, R_seq_baseline_aligned)
    H_topo = reconstruct_covariance(sigma_topo, R_seq_topo)

    returns_aligned = log_returns_df.loc[dates].to_numpy(dtype=np.float64)

    w_baseline = min_variance_weights_seq(H_baseline, ridge=ridge, long_only=long_only)
    w_topo = min_variance_weights_seq(H_topo, ridge=ridge, long_only=long_only)
    w_equal = equal_weights_seq(T_topo, N)

    variants = {
        'Baseline DCC': (w_baseline, returns_aligned),
        'TopoDCC': (w_topo, returns_aligned),
        'Equal-weight (1/N)': (w_equal, returns_aligned),
    }

    summary_rows = []
    per_variant_returns = {}
    for name, (w, r) in variants.items():
        port_r = portfolio_returns(w, r)
        per_variant_returns[name] = port_r
        summary_rows.append({
            'variant': name,
            'ann_return_%': np.mean(port_r) * periods_per_year * 100,
            'ann_vol_%': annualize_vol(port_r, periods_per_year) * 100,
            'sharpe': annualize_sharpe(port_r, periods_per_year),
            'turnover': turnover(w) if name != 'Equal-weight (1/N)' else 0.0,
        })

    summary = pd.DataFrame(summary_rows).set_index('variant')

    return {
        'dates': dates,
        'weights': {'Baseline DCC': w_baseline, 'TopoDCC': w_topo, 'Equal-weight (1/N)': w_equal},
        'returns': per_variant_returns,
        'summary': summary,
        'tickers': tickers,
    }


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.data.loader import load_config

    config = load_config()
    paths = config['paths']

    log_returns = pd.read_parquet(paths['log_returns'])
    sigma_df = pd.read_parquet(paths.get('garch_sigma', 'data/processed/garch_sigma.parquet'))

    baseline_results = np.load(paths['dcc_baseline'], allow_pickle=True).item()
    topo_reg_results = np.load(paths['dcc_topo_reg'], allow_pickle=True).item()

    result = run_backtest(
        log_returns_df=log_returns,
        sigma_df=sigma_df,
        R_seq_baseline=baseline_results['R_seq'],
        R_seq_topo=topo_reg_results['R_seq'],
    )

    print("\n=== Portfolio backtest: Baseline DCC vs TopoDCC (regularized) vs Equal-weight ===\n")
    print(result['summary'].round(4).to_string())
