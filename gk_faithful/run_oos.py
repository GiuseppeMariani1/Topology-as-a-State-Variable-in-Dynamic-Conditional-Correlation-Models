# -*- coding: utf-8 -*-
"""
Out-of-sample evaluation of TopoDCC driven by the G&K features.

Reuses src/evaluation/oos_evaluation.py unchanged: prepare_split,
fit_all_train_only, oos_portfolio_backtest, run_dm_tests and
run_crisis_dm_tests are imported and called as they are. The only
intervention is that, for the duration of the call, prepare_split's
load_aligned_data is pointed at gk_faithful.data.load_aligned_gk. This is
in-process (prepare_split is called with window=None, so it never spawns
the subprocesses that would rebuild, and overwrite, thesis feature files)
and the original binding is restored afterwards.

Baseline DCC, aDCC, equal weight and both TopoDCC variants are all refit on
the SAME aligned sample and split, so every comparison is like for like.
They will not match the thesis numbers exactly, because the sample starts
later (it has to wait for the w-day point cloud plus the m-day spectral
window) and the features are lagged by one day.

USAGE
  python -m gk_faithful.run_oos --features gk_l2 --window 50 --m 500
  python -m gk_faithful.run_oos --features gk --window 50 --m 250 --split-date 2019-12-31
"""

import argparse
import contextlib
import os

import numpy as np
import pandas as pd

from gk_faithful.common import load_gk_config, result_path, run_tag
from gk_faithful.data import FEATURE_SETS, load_aligned_gk
import src.evaluation.oos_evaluation as oos  # original, read-only use


@contextlib.contextmanager
def _gk_loader(features, w, m, lag):
    original = oos.load_aligned_data

    def _load(config=None, features=None, verbose=True):
        # prepare_split passes its own `features` name; it is ignored here
        # because the G&K set is fixed by the enclosing arguments.
        return load_aligned_gk(features_gk, w, m, lag=lag, verbose=verbose)

    features_gk = features
    oos.load_aligned_data = _load
    try:
        yield
    finally:
        oos.load_aligned_data = original


def run(features, w, m, split_name='frac80', train_frac=0.8, split_date=None,
        n_iter=500, lr=0.01, lambda_reg=None, skip_adcc=False, lag=None, out=None):
    print("=" * 66)
    print(f"G&K OOS EVALUATION  features={features} w={w} m={m} split={split_name}")
    print("=" * 66)

    with _gk_loader(features, w, m, lag):
        split = oos.prepare_split(train_frac=train_frac, features='lpnorm',
                                  pca_components=None, window=None,
                                  split_date=split_date)

    R_seq_by_model, timings = oos.fit_all_train_only(
        split, n_iter=n_iter, lr=lr, lambda_reg=lambda_reg, skip_adcc=skip_adcc)

    sigma_df = pd.read_parquet(split['paths']['garch_sigma'])

    print("\n" + "=" * 66 + "\nOUT-OF-SAMPLE PORTFOLIO BACKTEST (test period only)\n" + "=" * 66)
    summary, port_returns, _ = oos.oos_portfolio_backtest(R_seq_by_model, split, sigma_df)
    print(summary.round(4).to_string())

    print("\n" + "=" * 66 + "\nDIEBOLD-MARIANO TESTS (test period only)\n" + "=" * 66)
    dm_results = oos.run_dm_tests(R_seq_by_model, split, port_returns)

    print("\n" + "=" * 66 + "\nCRISIS-SUBPERIOD DIEBOLD-MARIANO TESTS\n" + "=" * 66)
    crisis_dm_results = oos.run_crisis_dm_tests(R_seq_by_model, split)

    out = out or result_path('oos', run_tag(features, w, m, split_name))
    np.save(out, {
        'features': features, 'window': w, 'spectral_window': m,
        'feature_columns': split['feature_columns'],
        'feature_lag': load_gk_config()['timing']['feature_lag'] if lag is None else lag,
        'summary': summary,
        'dm_results': dm_results,
        'crisis_dm_results': crisis_dm_results,
        'timings': timings,
        'train_size': split['train_size'],
        'train_frac': train_frac,
        'split_date': split_date,
        'sample_dates': (split['dates'][0], split['dates'][-1]),
        'test_dates': split['dates'][split['train_size']:],
        'R_seq_test': R_seq_by_model,
        'portfolio_returns': port_returns,
    }, allow_pickle=True)
    print(f"\nSaved results to {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="OOS evaluation of TopoDCC on G&K features")
    parser.add_argument('--features', default='gk_l2', choices=FEATURE_SETS)
    parser.add_argument('--window', type=int, default=50, help="point-cloud window w")
    parser.add_argument('--m', type=int, default=500, help="spectral window")
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--split-date', default=None)
    parser.add_argument('--n-iter', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lambda-reg', type=float, default=None,
                        help="defaults to oos_evaluation's 4000 (tuned on the thesis's "
                             "9 features, not on these)")
    parser.add_argument('--lag', type=int, default=None,
                        help="override timing.feature_lag (0 = thesis convention)")
    parser.add_argument('--skip-adcc', action='store_true')
    parser.add_argument('--split-name', default=None,
                        help="label used in the output file name (run_all passes its own)")
    args = parser.parse_args()

    split_name = args.split_name or (
        f"split{args.split_date}" if args.split_date else f"frac{int(args.train_frac*100)}")
    run(args.features, args.window, args.m, split_name=split_name,
        train_frac=args.train_frac, split_date=args.split_date,
        n_iter=args.n_iter, lr=args.lr, lambda_reg=args.lambda_reg,
        skip_adcc=args.skip_adcc, lag=args.lag)


if __name__ == "__main__":
    main()
