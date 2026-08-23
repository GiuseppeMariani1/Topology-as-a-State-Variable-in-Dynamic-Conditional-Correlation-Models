# -*- coding: utf-8 -*-
"""
Velocity ("speed") features: the RATE of change in topology, not its level.

WHY THIS EXISTS
  The in-sample observation motivating this thesis was that the geometry of
  the return cloud gets *destroyed* ahead of stress -- clusters fragment,
  loops collapse. Destruction is a derivative: it is a change in the
  topology, not a property of any single snapshot.

  The existing TopoDCC feeds LEVELS into the recursion:
      a_t = sigmoid(w_a . phi_t)
  So it can only learn "high entropy -> fast adaptation". It cannot express
  "entropy is *collapsing right now* -> fast adaptation", because phi_t
  carries no memory of phi_{t-k}.

  These features make the derivative explicit:
      dphi_t = phi_t - phi_{t-k}

WHY k > 1
  phi_t is itself computed from a rolling window of W days. A one-day
  difference moves only 1/W of the underlying point cloud, so dphi_t with
  k=1 on a 250-day window is nearly zero and dominated by numerical noise.
  k should be large enough that a meaningful fraction of the window has
  turned over, but short enough to still be timely. Default k=5 (one
  trading week).

WHY 'speed' MODE IS THE HONEST FIRST TEST
  The unregularized permutation test on the 9 level features came back at
  p ~= 0.060, with the explicit warning that the likelihood gap might
  reflect parameter count rather than topology content. Concatenating
  levels and speed would DOUBLE the feature count (9 -> 18) and make that
  objection worse, not better.

  'speed' mode keeps the dimensionality identical to the current model, so
  the two are directly comparable and any improvement cannot be dismissed
  as extra capacity. 'levels_speed' is available as a follow-up once the
  pure-derivative hypothesis has been tested on its own terms.

NO STANDARDIZATION HERE
  oos_evaluation.prepare_split standardizes on TRAIN statistics only, to
  avoid leaking test-period scale into the fit. Standardizing here as well
  would apply full-sample statistics and silently reintroduce exactly that
  leakage. Raw differences are written; scaling stays downstream.

USAGE
  python -m src.topology.velocity_features --source lpnorm --mode speed --k 5
  python -m src.topology.velocity_features --source lpnorm --mode levels_speed
  python -m src.topology.velocity_features --source landscape --mode speed
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.loader import load_config


SOURCE_KEYS = {
    'lpnorm':    'tda_features_lpnorm',
    'landscape': 'tda_features_landscape',
    'pi':        'tda_features_pi',
}

# Where each (source, mode) combination gets written. These keys are also
# registered in config.yaml and in loader.load_aligned_data so the new sets
# can be selected with --features exactly like the originals.
OUTPUT_KEYS = {
    ('lpnorm',    'speed'):        'tda_features_lpnorm_speed',
    ('lpnorm',    'levels_speed'): 'tda_features_lpnorm_levels_speed',
    ('landscape', 'speed'):        'tda_features_landscape_speed',
    ('landscape', 'levels_speed'): 'tda_features_landscape_levels_speed',
}


def compute_velocity(features_df, k=5, mode='speed'):
    """
    features_df : (T, F) DataFrame of TDA features, DatetimeIndex.
    k           : lookback in trading days for the difference.
    mode        : 'speed'        -> dphi_t only              (F columns)
                  'levels_speed' -> [phi_t ; dphi_t]         (2F columns)

    The first k rows have no valid difference and are dropped rather than
    zero-filled. Zero-filling would assert "no geometric change occurred"
    for the first week of the sample, which is a fabricated observation.
    Dropping shortens the sample by k days; load_aligned_data intersects
    indices, so downstream alignment handles this automatically.

    Returns a DataFrame with the same DatetimeIndex minus the first k rows.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k >= len(features_df):
        raise ValueError(f"k={k} exceeds the number of observations ({len(features_df)}).")

    delta = features_df.diff(periods=k)
    delta.columns = [f"d{k}_{c}" for c in features_df.columns]

    if mode == 'speed':
        out = delta
    elif mode == 'levels_speed':
        out = pd.concat([features_df, delta], axis=1)
    else:
        raise ValueError(f"mode must be 'speed' or 'levels_speed', got {mode!r}")

    out = out.iloc[k:]

    if out.isna().any().any():
        n_bad = int(out.isna().any(axis=1).sum())
        raise ValueError(
            f"{n_bad} rows still contain NaN after dropping the first {k}. "
            "The source feature file likely has internal gaps -- check the "
            "TDA pipeline output before using these features."
        )

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build velocity (rate-of-change) features from TDA feature levels."
    )
    parser.add_argument('--source', default='lpnorm', choices=list(SOURCE_KEYS),
                        help="which TDA feature file to differentiate")
    parser.add_argument('--mode', default='speed', choices=['speed', 'levels_speed'],
                        help="'speed' keeps dimensionality identical to the source "
                             "(directly comparable, no extra capacity); "
                             "'levels_speed' concatenates and doubles it")
    parser.add_argument('--k', type=int, default=5,
                        help="lookback in trading days for the difference (default 5)")
    args = parser.parse_args()

    config = load_config()
    paths = config['paths']

    src_path = paths[SOURCE_KEYS[args.source]]
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"{src_path} not found -- run the TDA pipeline for '{args.source}' first."
        )

    features = pd.read_parquet(src_path)
    print(f"Source: {src_path}")
    print(f"  {features.shape[0]} obs x {features.shape[1]} features")
    print(f"  {features.index[0].date()} -> {features.index[-1].date()}")

    out = compute_velocity(features, k=args.k, mode=args.mode)

    print(f"\nMode: {args.mode}  (k={args.k})")
    print(f"  {out.shape[0]} obs x {out.shape[1]} features")
    print(f"  {out.index[0].date()} -> {out.index[-1].date()}")
    print(f"  (dropped first {args.k} rows with no valid difference)")

    out_key = OUTPUT_KEYS[(args.source, args.mode)]
    out_path = paths.get(out_key, f"data/processed/{out_key}.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path)
    print(f"\nSaved to {out_path}")

    # A quick scale report. If the differences are numerically tiny relative
    # to the levels, k is too short for the underlying rolling window and the
    # model will be fitting noise -- better to see that here than to discover
    # it after a full OOS run.
    if args.mode == 'speed':
        rel = (out.abs().mean() / features.abs().mean().values).mean()
        print(f"\nMean |delta| / mean |level| = {rel:.4f}")
        if rel < 0.01:
            print("  WARNING: differences are <1% of level magnitude. k is probably")
            print("  too short relative to the TDA rolling window -- consider a larger")
            print("  --k, or regenerate the source features with a shorter --window.")


if __name__ == "__main__":
    main()
