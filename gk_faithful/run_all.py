# -*- coding: utf-8 -*-
"""
Run the whole G&K-faithful experiment grid and collect one summary table.

Stages (each skips work whose output already exists unless --rebuild):
  1. tda_gk         landscape norms for every point-cloud window w
  2. spectral_gk    variance / low-frequency spectral density for every (w, m)
  3. diagnostic_gk  Kendall-tau pre-event trends + plots (no models)
  4. run_oos        every feature set x w x m x split in config_gk.yaml
  5. run_permutation (only with --permutations; 500 shuffles each, slow on CPU)
  6. summary        data/processed_gk/summary_oos.csv (+ summary_perm.csv)

Default grid: 3 feature sets x 2 windows (50, 100) x 2 spectral windows
(250, 500) x 2 splits (80/20, COVID) = 24 OOS runs.

USAGE
  python -m gk_faithful.run_all                     # stages 1-4, 6
  python -m gk_faithful.run_all --jobs 12           # 12 OOS runs at once
  python -m gk_faithful.run_all --permutations      # adds stage 5
  python -m gk_faithful.run_all --only diagnostic   # stages 1-3 only
  python -m gk_faithful.run_all --summary-only
"""

import argparse
import concurrent.futures
import glob
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from gk_faithful import diagnostic_gk, run_permutation, spectral_gk, tda_gk
from gk_faithful.common import (REPO_ROOT, _guard_output, load_gk_config, load_thesis_paths,
                                norms_path, out_dir, result_path, run_tag, spectral_path)

TOPO = ('TopoDCC (unreg)', 'TopoDCC (reg)')


def _stamp(msg):
    print(f"\n{'#' * 70}\n[{time.strftime('%H:%M:%S')}] {msg}\n{'#' * 70}")


def build_features(gk, rebuild):
    pc, sp = gk['point_cloud'], gk['spectral']
    log_returns = None
    for w in pc['windows']:
        if rebuild or not os.path.exists(norms_path(w, gk)):
            if log_returns is None:
                log_returns = pd.read_parquet(load_thesis_paths()['log_returns'])
            tda_gk.compute_norm_series(
                log_returns, w, ps=gk['norms']['p'], step=pc['step'],
                standardise=pc['standardise_window'], n_jobs=pc['n_jobs']
            ).to_parquet(norms_path(w, gk))
        norms = pd.read_parquet(norms_path(w, gk))
        for m in sp['windows']:
            if rebuild or not os.path.exists(spectral_path(w, m, gk)):
                spectral_gk.compute_spectral_features(
                    norms, m, sp['low_freq_fraction'], sp['detrend']
                ).to_parquet(spectral_path(w, m, gk))
            print(f"features ready: w={w} m={m}")


def _split_kwargs(spec):
    return {'train_frac': spec.get('train_frac', 0.8), 'split_date': spec.get('split_date')}


def _oos_command(feats, w, m, split_name, spec, ex, skip_adcc):
    cmd = [sys.executable, '-m', 'gk_faithful.run_oos',
           '--features', feats, '--window', str(w), '--m', str(m),
           '--split-name', split_name,
           '--n-iter', str(ex['n_iter']), '--lr', str(ex['lr'])]
    kw = _split_kwargs(spec)
    if kw['split_date']:
        cmd += ['--split-date', str(kw['split_date'])]
    else:
        cmd += ['--train-frac', str(kw['train_frac'])]
    if skip_adcc:
        cmd.append('--skip-adcc')
    return cmd


def _run_one(tag, cmd, log_path, env):
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as log:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env).returncode
    return tag, rc, time.time() - t0


def run_oos_grid(gk, rebuild, skip_adcc, jobs):
    """
    Each OOS run is independent, so they run as separate processes, `jobs`
    at a time. The DCC recursion is a sequential day-by-day loop, so one
    process cannot use more than about one core; parallelism has to come
    from running several fits at once. Each process is pinned to one
    thread (OMP/MKL) so `jobs` processes do not fight over the cores.
    Output of each run goes to data/processed_gk/logs/<tag>.log.
    """
    ex = gk['experiments']
    todo = []
    for feats in ex['feature_sets']:
        for w in gk['point_cloud']['windows']:
            for m in gk['spectral']['windows']:
                for split_name, spec in ex['splits'].items():
                    tag = run_tag(feats, w, m, split_name)
                    if not rebuild and os.path.exists(result_path('oos', tag)):
                        print(f"skip (exists): {tag}")
                        continue
                    todo.append((tag, _oos_command(feats, w, m, split_name, spec, ex, skip_adcc)))

    if not todo:
        return []

    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    log_dir = os.path.join(out_dir(gk), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    jobs = max(1, min(jobs, len(todo)))
    print(f"{len(todo)} OOS runs, {jobs} in parallel. Logs: {log_dir}")

    failures, done = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_run_one, tag, cmd,
                               _guard_output(os.path.join(log_dir, f'{tag}.log')), env)
                   for tag, cmd in todo]
        for fut in concurrent.futures.as_completed(futures):
            tag, rc, secs = fut.result()
            done += 1
            status = 'ok' if rc == 0 else f'FAILED (exit {rc}, see log)'
            print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(todo)} {tag}: "
                  f"{status}  {secs / 60:.1f} min")
            if rc != 0:
                failures.append(tag)
    return failures


def run_perm_grid(gk, rebuild):
    failures = []
    for feats in [f for f in gk['experiments']['feature_sets'] if f != 'levels_gk']:
        for w in gk['point_cloud']['windows']:
            for m in gk['spectral']['windows']:
                tag = run_tag(feats, w, m, 'full')
                if not rebuild and os.path.exists(result_path('perm', tag)):
                    print(f"skip (exists): perm {tag}")
                    continue
                _stamp(f"PERMUTATION {tag}")
                try:
                    run_permutation.run(feats, w, m)
                except Exception:
                    traceback.print_exc()
                    failures.append(tag)
    return failures


def summarise(gk):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir(gk), 'oos', '*.npy'))):
        r = np.load(path, allow_pickle=True).item()
        s = r['summary']
        base = s.loc['Baseline DCC']
        for model in TOPO:
            if model not in s.index:
                continue
            key = f"{model} vs Baseline DCC"
            dm = r['dm_results'].get(key, {})
            rows.append({
                'features': r['features'], 'w': r['window'], 'm': r['spectral_window'],
                'split': os.path.basename(path).rsplit('_', 1)[-1][:-4],
                'test_start': pd.Timestamp(r['test_dates'][0]).date(),
                'model': model,
                'n_features': len(r['feature_columns']),
                'vol_%': s.loc[model, 'ann_vol_%'], 'vol_base_%': base['ann_vol_%'],
                'sharpe': s.loc[model, 'sharpe'], 'sharpe_base': base['sharpe'],
                # DM sign: < 0 means TopoDCC has the lower loss (better)
                'dm_qlike': dm.get('qlike', {}).get('dm_stat'),
                'p_qlike': dm.get('qlike', {}).get('p_value'),
                'dm_frob': dm.get('frobenius', {}).get('dm_stat'),
                'p_frob': dm.get('frobenius', {}).get('p_value'),
                'p_portfolio': dm.get('portfolio_sq', {}).get('p_value'),
            })
    if rows:
        df = pd.DataFrame(rows)
        path = _guard_output(os.path.join(out_dir(gk), 'summary_oos.csv'))
        df.to_csv(path, index=False)
        print(df.round(4).to_string(index=False))
        print(f"\nSaved {path}")
        wins = df[(df['dm_qlike'] < 0) & (df['p_qlike'] < 0.05)]
        print(f"\nRuns where TopoDCC beats baseline on QLIKE at 5%: {len(wins)} / {len(df)}")

    perm = []
    for kind in ('perm', 'perm_reg'):
        for path in sorted(glob.glob(os.path.join(out_dir(gk), kind, '*.npy'))):
            r = np.load(path, allow_pickle=True).item()
            perm.append({'features': r['features'], 'w': r['window'], 'm': r['spectral_window'],
                         'regularised': r['regularized'], 'real_ll': r['real_ll'],
                         'beat': r['n_beat_real'], 'n': len(r['permuted_lls']),
                         'p': r['p_value']})
    if perm:
        df = pd.DataFrame(perm)
        path = _guard_output(os.path.join(out_dir(gk), 'summary_perm.csv'))
        df.to_csv(path, index=False)
        print("\n" + df.round(4).to_string(index=False))
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Run the full G&K-faithful grid")
    parser.add_argument('--rebuild', action='store_true', help="recompute even if outputs exist")
    parser.add_argument('--permutations', action='store_true', help="also run permutation tests")
    parser.add_argument('--only', choices=['diagnostic'], default=None,
                        help="'diagnostic': build features and run the diagnostic, no models")
    parser.add_argument('--skip-adcc', action='store_true')
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="OOS runs in parallel (default: physical cores, "
                             "i.e. half the logical CPU count)")
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()

    gk = load_gk_config()
    t0 = time.time()
    failures = []

    if not args.summary_only:
        _stamp("Stages 1-2: G&K landscape norms and spectral indicators")
        build_features(gk, args.rebuild)

        _stamp("Stage 3: pre-event diagnostic")
        diagnostic_gk.main()

        if args.only != 'diagnostic':
            _stamp("Stage 4: OOS grid")
            failures += run_oos_grid(gk, args.rebuild, args.skip_adcc, args.jobs)
            if args.permutations:
                _stamp("Stage 5: permutation tests")
                failures += run_perm_grid(gk, args.rebuild)

    _stamp("Stage 6: summary")
    summarise(gk)

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    if failures:
        print(f"FAILED runs ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
