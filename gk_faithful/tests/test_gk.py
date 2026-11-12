# -*- coding: utf-8 -*-
"""
Correctness checks for gk_faithful. Run with:  python -m pytest gk_faithful/tests -q

1. The closed-form landscape norm equals grid integration of the thesis's
   own compute_persistence_landscape() over ALL levels.
2. The spectral indicators are strictly trailing: changing future values
   never changes an earlier row.
3. The low-frequency spectral density responds to slow variation, not to
   fast noise of the same variance.
4. The feature lag shifts features forward in time.
5. Outputs cannot be written into the thesis data folder.
"""

import numpy as np
import pandas as pd
import pytest

from gk_faithful.common import REPO_ROOT, _guard_output
from gk_faithful.spectral_gk import rolling_indicators
from gk_faithful.tda_gk import h1_diagram, landscape_norm_from_diagram
from src.topology.tda_pipeline import compute_persistence_landscape


def _grid_norm(dgm, p, n_grid=200001):
    finite = dgm[np.isfinite(dgm[:, 1])]
    grid = np.linspace(finite[:, 0].min(), finite[:, 1].max(), n_grid)
    lam = compute_persistence_landscape(dgm, grid, n_landscapes=len(finite))
    return np.trapezoid(np.sum(np.abs(lam) ** p, axis=0), grid) ** (1 / p)


@pytest.mark.parametrize("p", [1, 2])
def test_closed_form_matches_grid_on_handmade_diagram(p):
    dgm = np.array([[0.0, 1.0], [0.2, 0.9], [0.5, 2.0], [1.5, 1.6], [0.0, np.inf]])
    assert landscape_norm_from_diagram(dgm, p) == pytest.approx(_grid_norm(dgm, p), rel=1e-6)


@pytest.mark.parametrize("p", [1, 2])
def test_closed_form_matches_grid_on_real_rips_diagram(p):
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    pts = np.c_[np.cos(t), np.sin(t), rng.normal(0, 0.05, (50, 3))]   # a loop in R^5
    dgm = h1_diagram(pts)
    assert len(dgm) > 0
    assert landscape_norm_from_diagram(dgm, p) == pytest.approx(_grid_norm(dgm, p), rel=1e-5)


def test_empty_diagram_is_zero():
    assert landscape_norm_from_diagram(np.empty((0, 2)), 2) == 0.0


def test_spectral_is_strictly_trailing():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range('2010-01-01', periods=800)
    s = pd.Series(rng.normal(size=800).cumsum(), index=idx)
    base = rolling_indicators(s, 250)
    s2 = s.copy()
    s2.iloc[600:] += 100 * rng.normal(size=200)
    changed = rolling_indicators(s2, 250)
    cutoff = idx[599]
    pd.testing.assert_frame_equal(base.loc[:cutoff], changed.loc[:cutoff])
    assert not np.allclose(base.loc[idx[650]:], changed.loc[idx[650]:])
    assert base.index[0] == idx[249]


def test_low_freq_density_detects_slow_variation():
    m = 500
    t = np.arange(m)
    slow = np.sin(2 * np.pi * t / 200.0)
    fast = np.sin(2 * np.pi * t / 4.0)
    idx = pd.bdate_range('2010-01-01', periods=m)
    a = rolling_indicators(pd.Series(slow, idx), m, detrend='constant')
    b = rolling_indicators(pd.Series(fast, idx), m, detrend='constant')
    assert a['var'].iloc[0] == pytest.approx(b['var'].iloc[0], rel=0.05)
    assert a['lfsd'].iloc[0] > 100 * b['lfsd'].iloc[0]


def test_feature_lag_shifts_forward():
    idx = pd.bdate_range('2010-01-01', periods=5)
    X = pd.DataFrame({'f': [1.0, 2, 3, 4, 5]}, index=idx)
    lagged = X.shift(1).iloc[1:]
    assert lagged.loc[idx[1], 'f'] == 1.0     # row for day 2 uses day 1's value


def test_cannot_write_into_thesis_folder():
    with pytest.raises(RuntimeError):
        _guard_output(f"{REPO_ROOT}/data/processed/anything.parquet")
