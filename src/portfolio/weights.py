# -*- coding: utf-8 -*-
"""
Portfolio weight construction from a covariance sequence H_seq.

Minimum-variance is used because it directly tests the thesis's actual
claim: does a better covariance/correlation estimate (topology-driven DCC)
translate into genuinely lower realized portfolio risk? It isolates the
covariance estimate as the only input — no expected-return assumption is
needed, unlike mean-variance — so any difference in realized outcomes is
attributable to the covariance model, not to return forecasting skill.
"""

import numpy as np


def min_variance_weights(H_t, ridge=1e-8):
    """
    Closed-form minimum-variance weights for a single covariance matrix:
        w = H^-1 1 / (1' H^-1 1)

    ridge: small diagonal loading added before inversion. H_t is
    reconstructed from fitted GARCH/DCC estimates rather than observed
    directly, so it can be near-singular on short/quiet windows; this
    guards against inversion blowing up without materially changing
    well-conditioned cases.
    """
    N = H_t.shape[0]
    H_reg = H_t + ridge * np.eye(N)

    ones = np.ones(N)
    H_inv_ones = np.linalg.solve(H_reg, ones)
    w = H_inv_ones / (ones @ H_inv_ones)
    return w


def min_variance_weights_seq(H_seq, ridge=1e-8, long_only=False):
    """
    Apply min_variance_weights across a full (T, N, N) sequence.

    long_only: if True, clip negative weights to zero and renormalise to
    sum to 1 at each t. Off by default — unconstrained min-variance is the
    textbook DCC-application comparison (Engle-style), and clipping
    introduces its own bias worth reporting as a separate robustness
    variant rather than silently baking in.
    """
    T, N, _ = H_seq.shape
    W = np.zeros((T, N))

    for t in range(T):
        w = min_variance_weights(H_seq[t], ridge=ridge)
        if long_only:
            w = np.clip(w, 0, None)
            s = w.sum()
            w = w / s if s > 0 else np.ones(N) / N
        W[t] = w

    return W


def equal_weights_seq(T, N):
    """1/N benchmark weights, constant over time."""
    return np.tile(np.ones(N) / N, (T, 1))
