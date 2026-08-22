# -*- coding: utf-8 -*-
"""
Diebold-Mariano test for comparing out-of-sample forecast accuracy.

WHY THIS EXISTS
  "TopoDCC's out-of-sample log-likelihood is higher" and "TopoDCC's
  realized portfolio volatility is lower" are both point comparisons --
  they say nothing about whether the difference is larger than sampling
  noise. DM turns those into a hypothesis test with a p-value.

THE TEST
  Given two competing forecasts and a loss function L, form the loss
  differential at each out-of-sample date:
      d_t = L(model_1, t) - L(model_2, t)
  Under H0 the two models have equal expected loss, so E[d_t] = 0. The
  statistic is
      DM = mean(d) / sqrt( avar(mean(d)) )
  and is asymptotically N(0,1).

  DM < 0  ->  model_1 has LOWER loss  ->  model_1 forecasts better.
  DM > 0  ->  model_2 forecasts better.

WHY HAC (Newey-West) VARIANCE
  d_t is serially correlated in practice -- volatility and correlation
  forecast errors cluster in time (a bad week is bad on every day of it).
  Using the plain sample variance would understate the standard error and
  produce spuriously significant p-values. The long-run variance is
  estimated with a Bartlett kernel; the default lag length follows the
  usual h-1 rule for h-step forecasts, with an automatic fallback to
  floor(T^(1/3)) for the 1-step case where h-1 = 0 would leave no
  autocorrelation correction at all despite d_t plainly being persistent.

HARVEY-LEYBOURNE-NEWBOLD CORRECTION
  The asymptotic N(0,1) approximation over-rejects in finite samples.
  Harvey, Leybourne & Newbold (1997) proposed a scaling factor plus a
  t(T-1) reference distribution instead. Applied by default (`harvey=True`)
  since out-of-sample windows in this thesis are ~950 days, not asymptotic.

LOSS FUNCTIONS PROVIDED
  Two standard multivariate-volatility losses, both computed against the
  realized outer product z_t z_t' as the (noisy but unbiased) proxy for
  the true conditional correlation:

    'frobenius' -- squared Frobenius norm of the forecast error matrix.
        Symmetric, easy to interpret, but treats all entries equally and
        is not robust to the noise in the z_t z_t' proxy.

    'qlike'     -- the multivariate QLIKE / Gaussian quasi-likelihood
        loss:  log|R_t| + z_t' R_t^{-1} z_t.  This is the loss the models
        are actually fit under, and it is known to be robust to noise in
        the volatility proxy (Patton 2011), which Frobenius/MSE is not.
        QLIKE is the primary loss to report; Frobenius is a robustness check.

  A portfolio-level loss is also provided ('portfolio_sq'), using squared
  realized portfolio return as the loss -- this tests the economic claim
  (does the better covariance estimate actually produce a lower-variance
  portfolio?) rather than the statistical one.

USAGE
  from src.evaluation.diebold_mariano import dm_test, qlike_loss_seq
  l1 = qlike_loss_seq(R_seq_model1, z_test)
  l2 = qlike_loss_seq(R_seq_model2, z_test)
  stat, p = dm_test(l1, l2)
"""

import numpy as np
from scipy import stats


# LOSS FUNCTIONS

def frobenius_loss_seq(R_seq, z_t):
    """
    Squared Frobenius norm between the forecast correlation matrix and the
    realized outer product z_t z_t'.

    R_seq : (T, N, N) forecast correlation matrices
    z_t   : (T, N)    standardized residuals over the same dates

    Returns (T,) array of per-date losses.
    """
    R_seq = np.asarray(R_seq, dtype=np.float64)
    z_t = np.asarray(z_t, dtype=np.float64)
    _check_aligned(R_seq, z_t)

    realized = np.einsum('ti,tj->tij', z_t, z_t)
    diff = R_seq - realized
    return np.sum(diff ** 2, axis=(1, 2))


def qlike_loss_seq(R_seq, z_t):
    """
    Multivariate QLIKE loss:  log|R_t| + z_t' R_t^{-1} z_t.

    This is (twice) the negative Gaussian log-likelihood contribution, i.e.
    exactly the objective the DCC models are estimated under, and it is
    robust to noise in the realized proxy in the Patton (2011) sense.

    Returns (T,) array of per-date losses. Lower is better.
    """
    R_seq = np.asarray(R_seq, dtype=np.float64)
    z_t = np.asarray(z_t, dtype=np.float64)
    _check_aligned(R_seq, z_t)

    sign, logdet = np.linalg.slogdet(R_seq)
    if np.any(sign <= 0):
        n_bad = int(np.sum(sign <= 0))
        raise ValueError(
            f"{n_bad} forecast correlation matrices are not positive definite "
            "-- QLIKE is undefined for these. Check the DCC recursion's "
            "stationarity constraint rather than silently dropping them."
        )

    R_inv = np.linalg.inv(R_seq)
    mahal = np.einsum('ti,tij,tj->t', z_t, R_inv, z_t)
    return logdet + mahal


def portfolio_sq_loss(portfolio_returns):
    """
    Squared realized portfolio return, as an economic loss for the
    minimum-variance application: a covariance model that genuinely
    forecasts risk better should produce a portfolio with smaller
    squared returns on average.

    Note this tests a different (and for a thesis, complementary) claim
    than QLIKE: statistical forecast accuracy vs realized economic
    outcome. They can disagree, and if they do that is worth reporting
    rather than picking whichever is favourable.
    """
    r = np.asarray(portfolio_returns, dtype=np.float64)
    return r ** 2


def _check_aligned(R_seq, z_t):
    if R_seq.shape[0] != z_t.shape[0]:
        raise ValueError(
            f"R_seq has {R_seq.shape[0]} dates but z_t has {z_t.shape[0]} -- "
            "these must cover exactly the same out-of-sample period."
        )
    if R_seq.shape[1] != z_t.shape[1] or R_seq.shape[2] != z_t.shape[1]:
        raise ValueError(
            f"R_seq shape {R_seq.shape} inconsistent with {z_t.shape[1]} assets."
        )


# NEWEY-WEST LONG-RUN VARIANCE

def _newey_west_lrv(d, n_lags):
    """
    Bartlett-kernel HAC estimate of the long-run variance of d_t.

    Returns the long-run variance (NOT divided by T) -- the caller divides
    by T to get the variance of the sample mean.
    """
    T = len(d)
    d_centered = d - d.mean()

    gamma_0 = np.dot(d_centered, d_centered) / T
    lrv = gamma_0

    for lag in range(1, n_lags + 1):
        gamma_k = np.dot(d_centered[lag:], d_centered[:-lag]) / T
        weight = 1.0 - lag / (n_lags + 1.0)  # Bartlett
        lrv += 2.0 * weight * gamma_k

    # A Bartlett-weighted sum is guaranteed non-negative in theory but can
    # go slightly negative from floating point on near-zero variance. Floor
    # it rather than returning a NaN standard error.
    return max(lrv, 1e-12)


# THE TEST

def dm_test(loss_1, loss_2, h=1, n_lags=None, harvey=True,
            name_1="model 1", name_2="model 2", verbose=False):
    """
    Diebold-Mariano test of equal predictive accuracy.

    loss_1, loss_2 : (T,) per-date loss series for the two competing models,
                     over the SAME out-of-sample dates.
    h              : forecast horizon (1 for one-step-ahead).
    n_lags         : HAC truncation lag. Default: max(h-1, floor(T^(1/3))).
                     The floor matters -- for h=1 the textbook h-1 rule gives
                     0 lags, which assumes serially uncorrelated loss
                     differentials, and volatility forecast errors plainly
                     are not.
    harvey         : apply the Harvey-Leybourne-Newbold small-sample
                     correction and use t(T-1) rather than N(0,1).

    Returns (dm_stat, p_value).
      dm_stat < 0 -> loss_1 is lower on average -> model 1 forecasts better.
      Two-sided p-value.
    """
    loss_1 = np.asarray(loss_1, dtype=np.float64)
    loss_2 = np.asarray(loss_2, dtype=np.float64)

    if loss_1.shape != loss_2.shape:
        raise ValueError(
            f"Loss series have different lengths ({loss_1.shape} vs "
            f"{loss_2.shape}) -- both models must be evaluated on identical dates."
        )

    d = loss_1 - loss_2
    T = len(d)

    if T < 10:
        raise ValueError(f"Only {T} out-of-sample observations -- too few for a DM test.")

    if n_lags is None:
        n_lags = max(h - 1, int(np.floor(T ** (1.0 / 3.0))))

    d_bar = d.mean()
    lrv = _newey_west_lrv(d, n_lags)
    var_d_bar = lrv / T

    dm_stat = d_bar / np.sqrt(var_d_bar)

    if harvey:
        # Harvey, Leybourne & Newbold (1997) correction factor
        correction = np.sqrt(
            (T + 1 - 2 * h + h * (h - 1) / T) / T
        )
        dm_stat = dm_stat * correction
        p_value = 2 * stats.t.sf(np.abs(dm_stat), df=T - 1)
    else:
        p_value = 2 * stats.norm.sf(np.abs(dm_stat))

    if verbose:
        better = name_1 if d_bar < 0 else name_2
        print(f"  DM test: {name_1} vs {name_2}")
        print(f"    T = {T}, HAC lags = {n_lags}, "
              f"{'Harvey-corrected t' if harvey else 'asymptotic normal'}")
        print(f"    mean loss diff = {d_bar:.6f}")
        print(f"    mean loss: {name_1} = {loss_1.mean():.6f}, "
              f"{name_2} = {loss_2.mean():.6f}")
        print(f"    DM statistic = {dm_stat:.4f}")
        print(f"    p-value      = {p_value:.4f}")
        if p_value < 0.05:
            print(f"    -> {better} forecasts significantly better at 5%.")
        else:
            print(f"    -> no significant difference at 5% "
                  f"({better} is nominally better but within noise).")

    return dm_stat, p_value


def dm_test_all_losses(R_seq_1, R_seq_2, z_test, h=1, harvey=True,
                        name_1="model 1", name_2="model 2", verbose=True):
    """
    Convenience wrapper: run the DM test under both QLIKE and Frobenius
    loss for the same pair of models, and return a tidy dict.

    Reporting both matters: QLIKE is the primary (it's the estimation
    objective and is proxy-noise robust), Frobenius is the robustness
    check. If they disagree, that disagreement belongs in the write-up.
    """
    results = {}

    for loss_name, loss_fn in (('qlike', qlike_loss_seq),
                                ('frobenius', frobenius_loss_seq)):
        l1 = loss_fn(R_seq_1, z_test)
        l2 = loss_fn(R_seq_2, z_test)
        if verbose:
            print(f"\n[{loss_name.upper()} loss]")
        stat, p = dm_test(l1, l2, h=h, harvey=harvey,
                          name_1=name_1, name_2=name_2, verbose=verbose)
        results[loss_name] = {
            'dm_stat': stat,
            'p_value': p,
            f'mean_loss_{name_1}': float(l1.mean()),
            f'mean_loss_{name_2}': float(l2.mean()),
        }

    return results
