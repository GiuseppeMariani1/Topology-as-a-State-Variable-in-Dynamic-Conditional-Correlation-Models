# Topological Dynamic Conditional Correlations: A Persistence Landscape Approach to Time-Varying Dependence in Financial Markets

**MSc Asset Pricing Thesis, King's College London**

---

## Overview

This repository implements and rigorously tests **TopoDCC**, an extension of the Dynamic Conditional Correlation (DCC) framework in which topological features derived from persistent homology drive the DCC shock-sensitivity and persistence parameters (`a_t`, `b_t`), rather than fixing them as constants.

The core idea, motivated by Gidea & Katz (2018), is that the geometric structure of the multivariate return point cloud carries information about market regime changes that is not explained by volatility, autocorrelation, or realized correlation alone. Unlike prior work that uses topology as an external, descriptive feature (a crash-detection index, an MST-based diagnostic), this project treats it as an **endogenous state variable** inside the model's own recursion.

**Headline finding:** across every configuration tested — linear encoder, 8+ rolling window sizes (21–400 days), levels and rate-of-change (velocity) features, regularized and unregularized fits, standard and crisis-restricted out-of-sample splits — TopoDCC does not forecast correlation dynamics better than a two-parameter baseline DCC model out-of-sample. The in-sample improvement reported early in this project (+420 log-likelihood nats) reflects overfitting, confirmed directly via a proper train/test split. This is a rigorously evidenced negative result; see `RESULTS.md` for the full chronology and every test performed.

---

## Methodology

### Pipeline

```
Raw Prices
    └── Log Returns  (src/data/loader.py)
        └── GARCH(1,1) Residuals + Volatility  (src/volatility/garch.py)
            └── TDA Pipeline  (src/topology/tda_pipeline.py)
                │   - Delay embedding (embed_dim=5)
                │   - PCA reduction (pca_dim=10)
                │   - Vietoris-Rips persistence via Ripser
                │   - Persistence landscapes on fixed global grid
                └── L² Norm Features  (src/topology/lp_norm_features.py)
                    │   - 9 features: betti_1, entropy_h0/h1, max/total
                    │     persistence, wasserstein, 3 landscape-level norms
                    ├── Velocity Features  (src/topology/velocity_features.py)
                    │   - Δφ_t = φ_t - φ_{t-k}: rate-of-change, not level
                    ├── TopoDCC  (src/models/dcc_topo.py, dcc_topo_reg.py)
                    ├── Baseline DCC  (src/models/dcc_baseline.py)
                    ├── aDCC  (src/models/dcc_adcc.py)  -- asymmetric benchmark
                    ├── Permutation Test  (src/models/permutation_test.py)
                    ├── Out-of-Sample Evaluation  (src/evaluation/oos_evaluation.py)
                    │   - train/test split, forward recursion, DM tests,
                    │     crisis-subperiod tests
                    └── Portfolio Application  (src/portfolio/)
                        - covariance.py, weights.py, backtest.py
```

### Key Design Choices

- **Assets:** SPY, EEM, GLD, TLT, DBC — chosen for cross-asset geometric diversity (equity, EM, gold, rates, commodities). Sample: 2006-02 to 2025-12 (DBC's Feb 2006 inception is the binding start-date constraint).
- **Homology degree:** H1 only (loop structure), following Gidea & Katz (2018).
- **TopoDCC parameterization:** `a_t = σ(w_a·φ_t + bias_a)`, `b_t = σ(w_b·φ_t + bias_b)`, a single linear layer — deliberately kept simple; an MLP encoder was considered and rejected (see `RESULTS.md`, July 2026) since added capacity would worsen the exact overfitting risk under investigation.
- **Regularization:** Ridge (L2) penalty on `w_a`/`w_b`, fit via Adam with early stopping.
- **aDCC baseline:** Cappiello-Engle-Sheppard (2006) asymmetric extension — the harder benchmark. Result: underperforms vanilla DCC on this dataset (LR=-18.73, p=1.0).
- **Diebold-Mariano test:** HAC/Newey-West variance (Bartlett kernel), Harvey-Leybourne-Newbold small-sample correction. Three losses: QLIKE (primary), Frobenius (robustness), portfolio squared-return (economic).
- **Out-of-sample evaluation:** chronological train/test split (80/20 by default, or an explicit `--split-date`), features standardized on train statistics only, models fit on train and forward-recursed into the held-out period.

### Windows and features actually tested

| Feature set | Windows tested |
|---|---|
| `lpnorm` (9 features, levels) | 21, 50, 60, 75, 100, 150, 200, 250, 300, 400 |
| `landscape` (27 features, levels) | 100, 150, 200, 300, 400 |
| `landscape` + PCA-5/PCA-10 | 100, 150, 200, 300, 400 |
| `lpnorm_speed` (9 features, Δφ_t, k=5) | 50, 60, 75, 250 |
| `lpnorm_levels_speed` (18 features) | 60, 250 |

None beat baseline DCC out-of-sample on QLIKE with a robust, replicated result. See `RESULTS.md` for every number, including the mid-project bug (window overrides silently not propagating to the derived `lpnorm` feature file) that initially produced a false "confirmation" at windows 60/75 — caught, fixed, and re-run before being trusted.

### Crisis-subperiod testing

Motivated by Souto (2023, *Topological Tail Dependence*), which finds persistent-homology forecasting gains concentrated in a held-out COVID subsample for flexible/neural models (but weaker-to-absent for linear ones). Replicated the design (`--split-date 2019-12-31` places COVID and the 2022 rate-hike period inside the test set; `--window 21` matches Souto's window). No crisis-concentrated advantage was found for TopoDCC's linear specification — consistent with Souto's own finding that linear models don't reliably show this effect.

---

## Repository Structure

```
MSCTHESIS/
├── config/
│   └── config.yaml                    # all parameters in one place
├── data/
│   ├── raw/                           # gitignored
│   └── processed/                     # tracked (force-added; see .gitignore)
├── src/
│   ├── data/
│   │   └── loader.py                  # prices, log returns, aligned data helpers
│   ├── volatility/
│   │   └── garch.py                   # GARCH(1,1) -> z_t (residuals) + sigma_t
│   ├── topology/
│   │   ├── tda_pipeline.py            # persistence landscapes, --window override
│   │   ├── lp_norm_features.py        # L² norm reduction -> 9 features
│   │   └── velocity_features.py       # Δφ_t rate-of-change features
│   ├── models/
│   │   ├── dcc_baseline.py            # constant-parameter DCC
│   │   ├── dcc_adcc.py                # asymmetric DCC (harder benchmark)
│   │   ├── dcc_topo.py                # TopoDCC model class + recursion
│   │   ├── dcc_topo_reg.py            # TopoDCC with L1/L2 regularization
│   │   └── permutation_test.py        # 500-run feature-shuffle significance test
│   ├── portfolio/
│   │   ├── covariance.py              # H_t = D_t R_t D_t reconstruction
│   │   ├── weights.py                 # minimum-variance portfolio weights
│   │   └── backtest.py                # 3-way backtest vs equal-weight
│   └── evaluation/
│       ├── diebold_mariano.py         # DM test, HAC variance, QLIKE/Frobenius
│       ├── oos_evaluation.py          # full OOS pipeline + crisis subperiod tests
│       └── stats_tests.py             # stationarity, crisis-window diagnostics
├── overnight_run.py                   # batch runner: all feature x window combos
├── RESULTS.md                         # full chronological result log (see below)
└── README.md
```

---

## Reproducing Results

```bash
# 1. Install dependencies
pip install numpy pandas torch ripser persim arch statsmodels scikit-learn matplotlib joblib yfinance pyarrow

# 2. Download data
python -m src.data.loader

# 3. GARCH filtering (produces z_t and sigma_t)
python -m src.volatility.garch

# 4. TDA pipeline (default window=250; add --window N to override)
python -m src.topology.tda_pipeline

# 5. L² norm reduction
python -m src.topology.lp_norm_features

# 6. (Optional) velocity/speed features
python -m src.topology.velocity_features --source lpnorm --mode speed --k 5

# 7. Fit models
python -m src.models.dcc_baseline
python -m src.models.dcc_adcc
python -m src.models.dcc_topo_reg          # regularized TopoDCC + lambda grid

# 8. Significance testing (500 permutations, GPU-batched)
python -m src.models.permutation_test
python -m src.models.permutation_test --reg

# 9. Out-of-sample evaluation (the main result)
python -m src.evaluation.oos_evaluation --features lpnorm --window 250
python -m src.evaluation.oos_evaluation --features lpnorm_speed --window 60
python -m src.evaluation.oos_evaluation --features lpnorm --window 21 --split-date 2019-12-31

# 10. Portfolio application
python -m src.portfolio.backtest --variant both

# 11. Diagnostics
python -m src.evaluation.stats_tests
```

For a full window x feature sweep in one run: `python overnight_run.py`.

---

## Dependencies

- Python 3.14
- PyTorch (CUDA optional; permutation tests batch on GPU when available, single fits run on CPU by design)
- Ripser / Persim (TDA)
- arch (GARCH)
- statsmodels, scikit-learn, scipy
- pandas, numpy, matplotlib, joblib, yfinance, pyarrow

---

## References

- Engle, R. (2002). Dynamic Conditional Correlation. *Journal of Business & Economic Statistics.*
- Cappiello, L., Engle, R., & Sheppard, K. (2006). Asymmetric Dynamics in the Correlations of Global Equity and Bond Returns. *Journal of Financial Econometrics.*
- Gidea, M. & Katz, Y. (2018). Topological Data Analysis of Financial Time Series: Landscapes of Crashes. *Physica A.*
- Bubenik, P. (2015). Statistical Topological Data Analysis using Persistence Landscapes. *JMLR.*
- Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics.*
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the Equality of Prediction Mean Squared Errors. *International Journal of Forecasting.*
- Souto, H.G. (2023). Topological Tail Dependence: Evidence from Forecasting Realized Volatility. *Journal of Finance and Data Science.*
- De Leon Miranda, J., Dolfin, M., Kapetanios, G., & Leonida, L. (2026). Detecting Network Instability via Multiscale Detrended Cross-Correlations and MST Topology. arXiv:2602.10174.

---

*King's College London MSc Asset Pricing 2026*
