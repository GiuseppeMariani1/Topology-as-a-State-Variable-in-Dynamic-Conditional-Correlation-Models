# -*- coding: utf-8 -*-
"""
Replication check before any model fitting: does the G&K early-warning
pattern appear on the thesis's assets at all?

For each (w, m) and each event in config_gk.yaml that the sample covers,
this computes the Kendall tau of each indicator against time over the
pre_event_days before the event (the Mann-Kendall trend statistic G&K
report). A positive, significant tau is the paper's signature. It also
plots the norm and indicator series with the events marked.

If the indicators show no rising trend before any event here, there is
little reason to expect them to help the DCC, and that is worth knowing
before spending a night of permutation tests on them.

Purely descriptive: no lag is applied and nothing feeds a model.

USAGE
  python -m gk_faithful.diagnostic_gk
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from gk_faithful.common import (load_gk_config, norms_path, out_dir,  # noqa: E402
                                result_path, spectral_path, _guard_output)


def pre_event_trends(spec, events, horizon):
    rows = []
    for name, date in events.items():
        d = pd.Timestamp(date)
        pre = spec.loc[spec.index < d].tail(horizon)   # strictly before the event
        if len(pre) < horizon:
            rows.append({'event': name, 'date': date, 'n': len(pre), 'covered': False})
            continue
        row = {'event': name, 'date': date, 'n': len(pre), 'covered': True}
        t = np.arange(len(pre))
        for col in spec.columns:
            tau, p = kendalltau(t, pre[col].to_numpy())
            row[f'tau_{col}'] = tau
            row[f'p_{col}'] = p
        rows.append(row)
    return pd.DataFrame(rows)


def plot(norms, spec, events, title, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(norms.index, norms['norm_l2'], lw=0.6, label='L2 norm')
    if 'norm_l1' in norms:
        axes[0].plot(norms.index, norms['norm_l1'], lw=0.6, alpha=0.6, label='L1 norm')
    axes[1].plot(spec.index, spec['var_l2'], lw=0.8, label='variance (L2 norm)')
    axes[2].plot(spec.index, spec['lfsd_l2'], lw=0.8, label='low-freq. spectral density (L2)')
    for ax in axes:
        for d in events.values():
            ax.axvline(pd.Timestamp(d), color='grey', ls='--', lw=0.7)
        ax.legend(loc='upper left', fontsize=8)
        ax.set_yscale('log')
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    gk = load_gk_config()
    events = gk['diagnostic']['events']
    horizon = gk['diagnostic']['pre_event_days']
    tables = []

    for w in gk['point_cloud']['windows']:
        norms = pd.read_parquet(norms_path(w, gk))
        for m in gk['spectral']['windows']:
            spec = pd.read_parquet(spectral_path(w, m, gk))
            tab = pre_event_trends(spec, events, horizon)
            tab.insert(0, 'm', m)
            tab.insert(0, 'w', w)
            tables.append(tab)
            plot(norms, spec, events, f'G&K indicators, w={w}, m={m}',
                 result_path('diagnostic', f'gk_w{w}_m{m}'))

    out = pd.concat(tables, ignore_index=True)
    csv = _guard_output(os.path.join(out_dir(gk), 'diagnostic', 'pre_event_kendall.csv'))
    out.to_csv(csv, index=False)

    show = ['w', 'm', 'event', 'covered'] + [c for c in out.columns if c.startswith('tau_') and 'l2' in c]
    print(out[show].round(3).to_string(index=False))
    print(f"\nFull table (with p-values and L1): {csv}")


if __name__ == "__main__":
    main()
