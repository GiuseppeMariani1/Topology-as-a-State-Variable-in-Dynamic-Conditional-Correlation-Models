"""
check_variance.py

Diagnostics for the topology-driven DCC-GARCH model.
Checks 1-3 are active; check 4 (out-of-sample forecast variance) is
commented out since it requires a held-out period / realized vol proxy.

Expected inputs (adjust variable names/paths to match your pipeline):
    returns        : pd.DataFrame of asset log-returns, indexed by date
    dcc_model      : fitted DCC model object with `.conditional_covariance`
    a_t_series     : np.ndarray or pd.Series, fitted a_t values over time
    X              : np.ndarray, topological feature matrix (T x 9)
    weight_model   : fitted model mapping X -> a_t (e.g. logistic regressor)
    z_t            : pd.DataFrame/np.ndarray of standardized GARCH residuals
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.loader import load_config


# ---------------------------------------------------------------------
# 1. Variance explained (R^2-style): model-implied vs realized covariance
# ---------------------------------------------------------------------
def variance_explained(realized_var: np.ndarray, model_var: np.ndarray) -> float:
    """
    R^2-style variance-explained metric comparing model-implied covariance
    to realized covariance.
    """
    realized_var = np.asarray(realized_var)
    model_var = np.asarray(model_var)

    ss_res = np.sum((realized_var - model_var) ** 2)
    ss_tot = np.sum((realized_var - realized_var.mean()) ** 2)

    return 1 - ss_res / ss_tot


# Example usage (adjust key names if your .npy dict differs):
# baseline = np.load(paths['dcc_baseline_results'], allow_pickle=True).item()
# realized_var = returns.rolling(21).cov()          # realized covariance estimator
# model_var = baseline['Q_seq']                      # or baseline['R_seq'] — check your dict keys
# r2 = variance_explained(realized_var.values, model_var)
# print(f"[1] Variance explained (R^2-style): {r2:.4%}")


# ---------------------------------------------------------------------
# 2. Percentage of total variance in a_t/b_t captured by topology (X_t)
# ---------------------------------------------------------------------
def pct_variance_from_topology(a_t_series: np.ndarray, fitted_a_t: np.ndarray) -> float:
    """
    Fraction of the variance in the fitted parameter series (e.g. a_t)
    that is explained by the topology-driven component sigma(w^T X_t),
    versus residual/noise variance.
    """
    a_t_series = np.asarray(a_t_series)
    fitted_a_t = np.asarray(fitted_a_t)

    total_var = np.var(a_t_series)
    explained_var = np.var(fitted_a_t)

    return explained_var / total_var


# Example usage — wired to your dcc_topo_results_pi.npy (has w_a, w_b, a_seq, b_seq):
# topo = np.load(paths['dcc_topo_results'], allow_pickle=True).item()
# a_t_series = topo['a_seq']
# X = pd.read_parquet(paths['tda_features']).values   # standardized if that's how you fit it
# w_a = topo['w_a']
# fitted_a_t = X @ w_a   # or run it through your model's sigmoid, if a_seq is post-activation
# pct = pct_variance_from_topology(a_t_series, fitted_a_t)
# print(f"[2] Pct of a_t variance explained by topology: {pct:.4%}")


# ---------------------------------------------------------------------
# 3. GARCH residual variance stability (standardized residual diagnostic)
# ---------------------------------------------------------------------
def residual_variance_check(z_t: pd.DataFrame, tol: float = 0.05) -> pd.Series:
    """
    Checks whether standardized GARCH residuals have (approximately)
    unit variance, per asset. Deviations beyond `tol` flag possible
    misspecification or a misalignment between the GARCH and TDA windows.
    """
    z_t = pd.DataFrame(z_t)
    var_by_asset = z_t.var()
    flagged = (var_by_asset - 1.0).abs() > tol

    result = pd.DataFrame({
        "variance": var_by_asset,
        "deviation_from_1": var_by_asset - 1.0,
        "flagged": flagged,
    })
    return result


# Example usage — wired to your garch_residuals.parquet:
# z_t = pd.read_parquet(paths['garch_residuals'])
# diag = residual_variance_check(z_t)
# print("[3] Residual variance diagnostics:")
# print(diag)


# ---------------------------------------------------------------------
# 4. (COMMENTED OUT) Out-of-sample variance / forecast accuracy
#    Requires a held-out test period and a realized volatility proxy
#    (e.g. squared returns or intraday realized vol). Uses QLIKE / MSE.
# ---------------------------------------------------------------------

# def qlike_loss(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
#     """
#     QLIKE loss - standard in the GARCH forecasting literature.
#     Lower is better.
#     """
#     realized_var = np.asarray(realized_var)
#     forecast_var = np.asarray(forecast_var)
#     return np.mean(realized_var / forecast_var - np.log(realized_var / forecast_var) - 1)
#
#
# def mse_loss(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
#     realized_var = np.asarray(realized_var)
#     forecast_var = np.asarray(forecast_var)
#     return np.mean((realized_var - forecast_var) ** 2)
#
#
# # Example usage:
# # realized_vol_proxy = returns_oos ** 2  # or intraday realized vol
# # forecast_var = dcc_model.forecast(horizon=oos_horizon)
# # print(f"[4] QLIKE: {qlike_loss(realized_vol_proxy, forecast_var):.6f}")
# # print(f"[4] MSE:   {mse_loss(realized_vol_proxy, forecast_var):.6f}")


if __name__ == "__main__":
    config = load_config()
    paths = config['paths']

    # --- Check 3: residual variance stability (fastest sanity check) ---
    z_t = pd.read_parquet(paths['garch_residuals'])
    diag = residual_variance_check(z_t)
    print("[3] Residual variance diagnostics:")
    print(diag)
    print()

    # --- Check 2: pct of a_t variance explained by topology ---
    # NOTE: rather than reconstruct sigmoid(X_std @ w_a) — which requires
    # exactly matching training-time preprocessing we don't have visibility
    # into — this regresses a_seq directly on X and reports R^2. Robust to
    # unknown scaling/link-function details.
    try:
        from sklearn.linear_model import LinearRegression

        topo = np.load(paths['dcc_topo_lpnorm'], allow_pickle=True).item()
        X = pd.read_parquet(paths['tda_features_lpnorm']).values
        a_t_series = topo['a_seq']

        print(f"[2] diagnostics: var(a_seq)={np.var(a_t_series):.8f}, "
              f"mean(a_seq)={np.mean(a_t_series):.6f}, "
              f"std(a_seq)={np.std(a_t_series):.6f}")
        if np.var(a_t_series) < 1e-4:
            print("[2] NOTE: a_t barely moves around its mean — the topology-driven "
                  "component has very little practical effect on the DCC recursion, "
                  "even if statistically detectable.")

        reg = LinearRegression().fit(X, a_t_series)
        r2 = reg.score(X, a_t_series)
        print(f"[2] R^2 of a_t regressed on topology features (X): {r2:.4%}")
    except (KeyError, FileNotFoundError) as e:
        print(f"[2] Skipped — check paths/keys in dcc_topo_lpnorm_results.npy ({e})")
    print()

    # --- Check 1: variance explained, model vs realized CORRELATION ---
    # (only R_seq — correlation matrices — is saved; no covariance/Q_seq available)
    try:
        baseline = np.load(paths['dcc_baseline'], allow_pickle=True).item()
        z_t = pd.read_parquet(paths['garch_residuals'])

        model_R = baseline['R_seq']              # shape (5006, 5, 5)
        n = model_R.shape[0]
        z_tail = z_t.iloc[-n:]                    # tail-align to R_seq length

        realized_R = z_tail.rolling(21).corr().dropna()
        # reshape realized_R (MultiIndex date x asset, asset) into (T, 5, 5)
        dates = realized_R.index.get_level_values(0).unique()
        realized_R_array = np.stack([realized_R.loc[d].values for d in dates])

        # align lengths (rolling window drops the first 20 rows)
        model_R_aligned = model_R[-len(dates):]

        r2 = variance_explained(realized_R_array, model_R_aligned)
        print(f"[1] Variance explained (R^2-style, correlation): {r2:.4%}")
    except (KeyError, FileNotFoundError) as e:
        print(f"[1] Skipped — check paths/keys in dcc_baseline_results.npy ({e})")