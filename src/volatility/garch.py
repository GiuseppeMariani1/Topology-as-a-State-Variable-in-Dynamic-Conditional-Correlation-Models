# -*- coding: utf-8 -*-
"""
GARCH(1,1) — univariate volatility filtering

Fits an independent GARCH(1,1) to each asset's log returns and saves
the standardised residuals z_t = r_t / sigma_t for use as DCC input.
"""

import numpy as np
import pandas as pd
from arch import arch_model
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.loader import load_config


def fit_garch_residuals(log_returns_df, verbose=True):
    """
    Fit GARCH(1,1) to each column of log_returns_df.
    Returns DataFrame of standardised residuals, same index.
    """
    residuals = {}

    for ticker in log_returns_df.columns:
        r = log_returns_df[ticker].dropna() * 100  # scale to % for numerical stability

        model = arch_model(r, vol='Garch', p=1, q=1, dist='normal', rescale=False)
        res = model.fit(disp='off')

        sigma = res.conditional_volatility
        z = r / sigma
        residuals[ticker] = z

        if verbose:
            params = res.params
            print(f"  {ticker}: omega={params['omega']:.4f}  "
                  f"alpha={params['alpha[1]']:.4f}  "
                  f"beta={params['beta[1]']:.4f}  "
                  f"alpha+beta={params['alpha[1]']+params['beta[1]']:.4f}")

    resid_df = pd.DataFrame(residuals).dropna()
    return resid_df


if __name__ == "__main__":
    config = load_config()
    paths  = config['paths']

    log_returns = pd.read_parquet(paths['log_returns'])

    print(f"Fitting GARCH(1,1) to {log_returns.shape[1]} assets...")
    residuals = fit_garch_residuals(log_returns, verbose=True)

    print(f"\nResiduals shape: {residuals.shape}")
    print(f"Date range: {residuals.index[0].date()} -> {residuals.index[-1].date()}")
    print(residuals.describe().round(4))

    os.makedirs(os.path.dirname(paths['garch_residuals']), exist_ok=True)
    residuals.to_parquet(paths['garch_residuals'])
    print(f"\nSaved to {paths['garch_residuals']}")