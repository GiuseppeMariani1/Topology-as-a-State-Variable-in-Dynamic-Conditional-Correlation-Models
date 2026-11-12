# -*- coding: utf-8 -*-
"""
Loader for the gk_faithful feature sets. Mirrors src.data.loader.
load_aligned_data (same return signature) so the original model,
evaluation and permutation code can be reused unchanged.

FEATURE SETS
  gk_l2     : [var_l2, lfsd_l2]                       2 features
  gk        : [var_l2, lfsd_l2, var_l1, lfsd_l1]      4 features
  levels_gk : thesis's 9 lpnorm features + gk_l2      11 features
              (the lpnorm file is read as-is from the thesis folder, so it
              reflects whatever thesis window is on disk, normally 250)

TIMING
  Features are shifted forward by timing.feature_lag trading days before
  alignment, so the row used for R_t was computed from returns up to t-lag.
  See config_gk.yaml for why the default is 1.
"""

import pandas as pd

from gk_faithful.common import load_gk_config, load_thesis_paths, spectral_path

FEATURE_SETS = ('gk_l2', 'gk', 'levels_gk')


def load_gk_features(features, w, m, gk_cfg=None):
    gk_cfg = gk_cfg or load_gk_config()
    spec = pd.read_parquet(spectral_path(w, m, gk_cfg))

    if features == 'gk_l2':
        return spec[['var_l2', 'lfsd_l2']]
    if features == 'gk':
        cols = [c for c in ('var_l2', 'lfsd_l2', 'var_l1', 'lfsd_l1') if c in spec.columns]
        return spec[cols]
    if features == 'levels_gk':
        levels = pd.read_parquet(load_thesis_paths()['tda_features_lpnorm'])
        levels = levels.drop(columns=[c for c in ('betti_0',) if c in levels.columns])
        both = levels.join(spec[['var_l2', 'lfsd_l2']], how='inner')
        return both
    raise ValueError(f"features must be one of {FEATURE_SETS}, got {features!r}")


def load_aligned_gk(features, w, m, lag=None, gk_cfg=None, verbose=True):
    """
    Returns (garch_residuals, tda_features, thesis_paths), aligned to shared
    dates, exactly like src.data.loader.load_aligned_data.
    """
    gk_cfg = gk_cfg or load_gk_config()
    lag = gk_cfg['timing']['feature_lag'] if lag is None else lag
    paths = load_thesis_paths()

    X = load_gk_features(features, w, m, gk_cfg)
    if lag:
        # shift along the features' own trading-day index, then drop the
        # first `lag` rows that no longer have a value
        X = X.shift(lag).iloc[lag:]

    z = pd.read_parquet(paths['garch_residuals'])
    common = z.index.intersection(X.index)
    if len(common) == 0:
        raise ValueError("No overlapping dates between GARCH residuals and G&K features.")
    z, X = z.loc[common], X.loc[common]

    if verbose:
        print(f"Residuals: {z.shape}")
        print(f"Features:  {X.shape}  ({features}, w={w}, m={m}, lag={lag})")
        print(f"  columns: {list(X.columns)}")
        print(f"  aligned: {common[0].date()} -> {common[-1].date()} ({len(common)} days)")
    return z, X, paths
