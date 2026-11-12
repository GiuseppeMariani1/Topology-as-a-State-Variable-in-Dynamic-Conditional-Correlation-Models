# -*- coding: utf-8 -*-
"""
Early-warning indicators on the landscape-norm series, as in Gidea & Katz
(2018), following Guttal et al. (2016): rolling variance and average
spectral density at low frequencies, over a trailing window of m days.

STRICTLY TRAILING
  The row dated t uses norm values on days t-m+1 .. t only. A centred
  window (the default in many early-warning toolkits) would put future
  values into the feature and leak test-period information.

DEFINITIONS (per trailing window x_1..x_m of one norm series)
  1. Remove a linear trend (config: spectral.detrend), giving e_1..e_m.
  2. var  = sample variance of e.
  3. Periodogram I(f_j) = |sum_s e_s exp(-2 pi i f_j s)|^2 / m at the
     positive Fourier frequencies f_j = j/m, j = 1..floor(m/2).
  4. lfsd = mean of I(f_j) over the lowest fraction of those frequencies
     (config: spectral.low_freq_fraction).
  G&K do not state the low-frequency cut-off or the detrending; both are
  exposed as parameters and reported in every output file name / README.

NO STANDARDISATION HERE
  As in src/topology/velocity_features.py: prepare_split standardises on
  training statistics only, so scaling stays downstream.

USAGE
  python -m gk_faithful.spectral_gk                  # all (w, m) in config
  python -m gk_faithful.spectral_gk --window 50 --m 500
"""

import argparse

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from gk_faithful.common import load_gk_config, norms_path, spectral_path


def _detrend(windows, how):
    """windows: (n, m). Returns residuals after removing mean or linear trend."""
    if how in (None, 'none', 'constant'):
        return windows - windows.mean(axis=1, keepdims=True)
    if how != 'linear':
        raise ValueError(f"detrend must be 'linear', 'constant' or 'none', got {how!r}")
    m = windows.shape[1]
    t = np.arange(m, dtype=float)
    t = t - t.mean()
    slope = (windows - windows.mean(axis=1, keepdims=True)) @ t / (t @ t)
    return windows - windows.mean(axis=1, keepdims=True) - slope[:, None] * t[None, :]


def rolling_indicators(series, m, low_freq_fraction=0.125, detrend='linear'):
    """
    series : pd.Series of one norm, DatetimeIndex.
    Returns a DataFrame with columns ['var', 'lfsd'], dated on each trailing
    window's last day; the first m-1 days are dropped (no full window).
    """
    x = series.to_numpy(dtype=float)
    if len(x) < m:
        raise ValueError(f"series has {len(x)} obs, fewer than the window m={m}")
    if np.isnan(x).any():
        raise ValueError("norm series contains NaN; check the tda_gk output")

    W = sliding_window_view(x, m)                    # (n, m), row i ends at i+m-1
    e = _detrend(W, detrend)
    var = e.var(axis=1, ddof=1)

    spec = np.abs(np.fft.rfft(e, axis=1)) ** 2 / m   # (n, m//2 + 1)
    pos = spec[:, 1:m // 2 + 1]                      # drop zero frequency
    n_low = max(1, int(np.floor(low_freq_fraction * pos.shape[1])))
    lfsd = pos[:, :n_low].mean(axis=1)

    return pd.DataFrame({'var': var, 'lfsd': lfsd}, index=series.index[m - 1:])


def compute_spectral_features(norms_df, m, low_freq_fraction=0.125, detrend='linear'):
    """One (var, lfsd) pair per norm column: var_l2, lfsd_l2, var_l1, lfsd_l1, ..."""
    parts = []
    for col in [c for c in norms_df.columns if c.startswith('norm_')]:
        tag = col[len('norm_'):]
        ind = rolling_indicators(norms_df[col], m, low_freq_fraction, detrend)
        ind.columns = [f'var_{tag}', f'lfsd_{tag}']
        parts.append(ind)
    return pd.concat(parts, axis=1)


def main():
    gk = load_gk_config()
    parser = argparse.ArgumentParser(description="G&K variance / low-frequency spectral density")
    parser.add_argument('--window', type=int, default=None, help="point-cloud window w")
    parser.add_argument('--m', type=int, default=None, help="spectral (trailing) window")
    args = parser.parse_args()

    sp = gk['spectral']
    windows = [args.window] if args.window else gk['point_cloud']['windows']
    ms = [args.m] if args.m else sp['windows']

    for w in windows:
        norms = pd.read_parquet(norms_path(w, gk))
        for m in ms:
            df = compute_spectral_features(norms, m, sp['low_freq_fraction'], sp['detrend'])
            path = spectral_path(w, m, gk)
            df.to_parquet(path)
            print(f"w={w} m={m}: {df.shape[0]} obs x {df.shape[1]} features "
                  f"({df.index[0].date()} -> {df.index[-1].date()})  -> {path}")


if __name__ == "__main__":
    main()
