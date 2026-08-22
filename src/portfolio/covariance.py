# -*- coding: utf-8 -*-
"""
Covariance reconstruction for the portfolio application chapter.

Reconstructs the full time-varying covariance matrix
    H_t = D_t R_t D_t
where D_t = diag(sigma_1t, ..., sigma_Nt) is the per-asset GARCH conditional
volatility and R_t is the DCC correlation matrix (baseline or topology-driven).

R_seq is produced by the DCC stages (dcc_baseline.py / dcc_topo.py /
dcc_topo_reg.py) and saved with shape (T, N, N).
sigma_t is produced by src/volatility/garch.py (fit_garch_residuals) and saved
alongside the standardised residuals z_t.

Because the topology-driven variants only start once a full rolling TDA
window is available (see config['topology']['window']), their R_seq is
shorter than the full sample and needs to be aligned against sigma_t by
slicing sigma_t to the same tail window — this module handles that
alignment explicitly rather than assuming shapes already match.
"""

import numpy as np
import pandas as pd


def align_sigma_to_R(sigma_df, R_seq, tickers=None):
    """
    Slice sigma_df (T_full x N) down to the trailing window matching R_seq's
    length (T_model x N x N), assuming R_seq's observations are the most
    recent T_model dates in sigma_df (true for baseline vs topology variants
    in this pipeline, where topology needs a warm-up window before the first
    estimate).

    Returns (sigma_aligned, dates_aligned) where sigma_aligned is an
    (T_model, N) numpy array in the same column order as `tickers`
    (or sigma_df's existing column order if tickers is None).
    """
    T_model = R_seq.shape[0]
    T_full = sigma_df.shape[0]

    if T_model > T_full:
        raise ValueError(
            f"R_seq has more observations ({T_model}) than sigma_df ({T_full}); "
            "cannot align — check you're comparing outputs from the same pipeline run."
        )

    if tickers is not None:
        sigma_df = sigma_df[tickers]

    sigma_aligned_df = sigma_df.iloc[-T_model:]
    return sigma_aligned_df.to_numpy(dtype=np.float64), sigma_aligned_df.index


def reconstruct_covariance(sigma_seq, R_seq):
    """
    H_t = D_t R_t D_t for every t.

    sigma_seq : (T, N) array of GARCH conditional volatilities, same scale
                used when GARCH was fit (percent returns in this pipeline —
                keep consistent, since portfolio weights only depend on
                *relative* scale within H_t, but mixing scales across models
                would silently break comparisons).
    R_seq     : (T, N, N) array of DCC correlation matrices.

    Returns H_seq : (T, N, N) array of covariance matrices.
    """
    if sigma_seq.shape[0] != R_seq.shape[0]:
        raise ValueError(
            f"sigma_seq has {sigma_seq.shape[0]} rows but R_seq has {R_seq.shape[0]} "
            "— call align_sigma_to_R first."
        )
    T, N = sigma_seq.shape
    if R_seq.shape[1:] != (N, N):
        raise ValueError(f"R_seq shape {R_seq.shape} inconsistent with N={N} assets.")

    D_seq = np.zeros((T, N, N), dtype=np.float64)
    idx = np.arange(N)
    D_seq[:, idx, idx] = sigma_seq

    # H_t = D_t @ R_t @ D_t, vectorised over t via einsum
    H_seq = np.einsum('tij,tjk,tkl->til', D_seq, R_seq.astype(np.float64), D_seq)
    return H_seq


def load_and_reconstruct(sigma_path, R_seq_path, tickers=None):
    """
    Convenience wrapper: load sigma parquet + R_seq .npy results dict,
    align, and reconstruct H_seq in one call.

    R_seq_path results file is expected to be a dict with at least an
    'R_seq' key (matches dcc_baseline / dcc_topo_lpnorm / dcc_topo_reg
    output format).
    """
    sigma_df = pd.read_parquet(sigma_path)
    results = np.load(R_seq_path, allow_pickle=True).item()
    R_seq = results['R_seq']

    sigma_seq, dates = align_sigma_to_R(sigma_df, R_seq, tickers=tickers)
    H_seq = reconstruct_covariance(sigma_seq, R_seq)
    return H_seq, dates, results
