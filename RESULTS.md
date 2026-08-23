# TopoDCC Project Summary — Chronological Record

## Feb 2026 — Origins: MSc thesis timing and PhD applications
Started thinking about PhD applications while the MSc thesis was still unfinished (due August). Discussed how to handle the timing gap — using a thesis proposal/chapter draft as a writing sample, leaning on references and a strong research proposal to carry the application in place of a finished thesis. Explored PhD programs (LSE realistic on grades, Oxford better fit but higher bar; TDA-in-economics is a niche intersection requiring the right supervisor more than the right department).

## May 2026 — Building the pipeline from scratch
**May 22:** Laid out the full architecture: GARCH(1,1) per asset → TDA pipeline (delay embedding, PCA reduction, Vietoris-Rips via Ripser, persistence landscapes) → L² norm feature extraction → TopoDCC (`a_t = σ(w_a·X_t)`, `b_t = σ(w_b·X_t)`) → permutation testing. Assets settled: SPY, EEM, GLD, TLT, DBC. Positioned against Gidea & Katz (2018) — moving from "scalar crash-detection index" to "vector state variable driving correlation forecasting."

**May 21:** Built the first geometric state variable, G_t (entropy of the return embedding). Found it dropped *during* crises (COVID min 2.04 vs mean ~3.9), not before — a contemporaneous signal, not a leading one. Explicitly flagged this as a limitation: "this clearly isn't a predictor." Decided to try shortening the window and using ΔG_t (rate of change) instead of levels — the same idea that resurfaced in August as the "speed features" breakthrough attempt.

## June 2026 — First real results
**~June 25:** First full TopoDCC fit completed.
- Baseline DCC (constant a,b): ll = **-8,247.96**
- TopoDCC (topology-driven a_t, b_t): ll = **-7,827.58**
- Improvement: **+420.38 nats**
- First permutation test (10 runs): real features beat permuted mean by +11.21 nats, only 1/10 shuffled runs beat real — looked clean at the time, but n=10 could only resolve p to 0.1.
- ADF test confirmed the three landscape L² norm series were stationary.

**July 1:** Regularization sweep. Unregularized weights had `|w|_2 = 1.31`; at λ=1000, this shrank to `0.042` (~30x tighter). Crucially, `ll_test` (out-of-sample within a train/test split) at λ=1000 (**-1822.30**) *beat* the unregularized model (**-1829.54**) — genuine evidence of overfitting in the unregularized version, confirmed properly rather than assumed. Final full-data refit at λ=1000: ll=-7849.53, still +398.42 over baseline. Flagged that `ll_test` was still improving at the top of the grid (λ=1000) — the true optimum hadn't been found yet, motivating a wider grid later.

**July 1 (same session):** Discussed adding MLP layers to the encoder. Explicitly decided against it — reasoned that more parameters would worsen the exact problem the permutation test was trying to solve (more capacity = easier to fit shuffled noise, as already seen when the raw 186-feature landscape model failed 8/10). Concluded: linear + strong regularization is the right MSc-scope story; MLP encoder was earmarked as a **PhD-scope extension**, not an MSc addition.

## August 2026 — Literature check, then heavy empirical work

**Aug 19:** Checked novelty against recent literature, including a live search for Marina Dolfin's newest paper. Found: **"Detecting Network Instability via Multiscale Detrended Cross-Correlations and MST Topology"** (De Leon Miranda, Dolfin, Kapetanios, Leonida — arXiv:2602.10174, Feb 2026). Their Elastic DCCR measure is network/graph-topological (MST length sensitivity to scale), used as a **derived diagnostic**, not a state variable driving a model's evolution — confirmed this still leaves TopoDCC's core claim (topology as endogenous state variable) distinct. Also checked and dismissed a Pasupuleti "quantum TDA" paper as templated, non-serious prior art.

**Aug 22–23 (this session, most of the heavy lifting):**

1. Built the portfolio application module (`covariance.py`, `weights.py`, `backtest.py`) — reconstructed `H_t = D_t R_t D_t`, min-variance weights, three-way backtest vs equal-weight. Found the `garch.py` gap (σ_t was never saved) and fixed it.

2. Built the aDCC baseline (Cappiello-Engle-Sheppard asymmetric extension) as a harder benchmark. **Result: aDCC underperforms vanilla DCC** (LR = -18.73, p=1.0) — asymmetry doesn't help this dataset.

3. Built the Diebold-Mariano test module (HAC/Newey-West variance, Harvey-Leybourne-Newbold small-sample correction, QLIKE/Frobenius/portfolio losses).

4. Built the full out-of-sample evaluation runner (`oos_evaluation.py`) — chronological 80/20 split, train-only fits, forward recursion into held-out data.

5. **First OOS run (window=250, the original design):** unregularized TopoDCC lost to baseline on QLIKE (DM p=0.044, baseline better) and Frobenius (p=0.017, baseline better). This directly answered the standing question "is the levels-based model overfit?" — **yes**, the in-sample edge did not survive out-of-sample testing.

6. Scaled permutation tests from 10/100 runs to 500 runs (statistically meaningful resolution). At window=250: unregularized landed at p≈0.060 (borderline, explicit warning about parameter count vs topology content); regularized (λ=4000, later corrected to λ=20000 in a subsequent run) landed at p≈0.000 (clean).

7. Ran an overnight sweep: 4 feature representations (lpnorm, landscape-27, landscape+PCA-10, landscape+PCA-5) × 5 window sizes (100, 150, 200, 300, 400) = 20 OOS evaluations. **Universal negative result** — baseline DCC beat every TopoDCC variant across every window and feature combination tested that night.

8. Motivated by the original May observation that geometry "gets destroyed" pre-crisis (a rate, not a level), built velocity/speed features (`velocity_features.py`): `Δφ_t = φ_t − φ_{t-5}`, same dimensionality as the levels version to avoid the parameter-count confound.

9. **First speed-feature result (window=60):** TopoDCC(reg) beat baseline on QLIKE (DM p=0.0445) — the first OOS win in the entire project. Levels+speed combined (18 features) reverted to baseline winning — confirmed the signal was specifically in the *rate*, not the *level*, and that adding levels back in reintroduced overfitting.

10. Permutation test on window=60 speed came back **marginal**: p≈0.052 — right at the boundary, not a clean confirmation.

11. Ran adjacent windows (50, 75) to check whether the window=60 result was part of a genuine neighborhood effect. Windows 60 and 75 both showed TopoDCC(reg) beating baseline (p=0.031–0.045); window 50 was untested at that point. **This looked like real, confirmed, three-window evidence.**

12. **Critical bug discovered and fixed:** the `--window` override in both `oos_evaluation.py` and `permutation_test.py` only called `tda_pipeline.py`, which writes the raw landscape file — but the 9-feature `lpnorm` file used by all the "speed" experiments is a **separate derived file** produced by `lp_norm_features.py`, which was never being re-invoked. Every "different window" run for `lpnorm`/`lpnorm_speed` had silently been reading the same stale feature file. Confirmed directly: window=50/60/75 runs had produced bit-identical train/test splits and bit-identical baseline DCC fits, and the two "different window" permutation tests returned identical results to two decimal places across 500 stochastic fits — not statistically plausible unless the underlying data was unchanged.

13. **Re-ran windows 50, 60, 75 with the bug fixed.** The apparent "three-window confirmation" did not survive:
    - Window 50: QLIKE p=0.264 (baseline nominally better)
    - Window 60: QLIKE p=0.0512 (TopoDCC nominally better, barely)
    - Window 75: QLIKE p=0.427 (TopoDCC nominally better, not significant)

    No consistent neighborhood pattern. The earlier "confirmation" was an artifact of the bug — all three "different" windows had actually been testing the same underlying window=250 data the whole time.

---

## Where the project honestly stands right now

**Confirmed, robust findings:**
- Baseline DCC beats levels-based TopoDCC out-of-sample, across window sizes 100–400 and four feature representations (lpnorm, landscape, landscape+PCA at two compression levels). This is a genuine, thoroughly-tested negative result.
- aDCC (asymmetric DCC) underperforms vanilla DCC on this dataset — asymmetry isn't the missing ingredient.
- The original in-sample result (June: +420 nats, 1/10 permutation) reflected genuine overfitting once tested properly out-of-sample — this was directly diagnosed, not just suspected.

**Unresolved / negative after correction:**
- The "speed" (rate-of-change) feature hypothesis, motivated by the original May 2026 observation about geometric destruction preceding crises, showed one promising result at window=60 (QLIKE p=0.0445) that appeared to replicate at window=75 — but this replication was later found to be a data pipeline bug (stale features), not real. Once fixed, the window=50/60/75 sweep shows no consistent pattern: one borderline result (p=0.051) flanked by two null results. This does not currently constitute evidence that speed features improve out-of-sample forecasting.

**Honest overall conclusion for the thesis:** TopoDCC's topology-driven correlation dynamics, in every configuration rigorously tested to date (levels or speed, five-plus window sizes, four feature representations, regularized or not), have not demonstrated a robust out-of-sample forecasting advantage over a simple two-parameter baseline DCC model. This is a defensible, well-evidenced negative result, arrived at through a genuine and traceable process of hypothesis generation, testing, bug discovery, and honest re-testing.

---

## Addendum: literature-motivated final checks (Aug 23, continued)

After the window=50/60/75 correction above, two further checks were run, directly motivated by a literature search for comparable published work.

**Literature check:** Souto (2023, "Topological Tail Dependence: Evidence from Forecasting Realized Volatility," *J. Finance and Data Science*) is the closest published positive result to this project's design — persistent homology driving a volatility/correlation-adjacent forecast, tested with formal DM tests. Key differences from TopoDCC, identified before running anything further:
- His feature is a single scalar (2D Wasserstein distance between consecutive persistence diagrams), not a 9-dimensional vector — already a pure "rate of change" measure by construction.
- His window is 21 trading days, shorter than anything tested in this project until this point.
- His positive result is **concentrated in a held-out COVID subsample** — full-sample DM tests in his paper are often null; the COVID-restricted ones are significant.
- His effect is present for flexible/neural models (NBEATSx) but weaker-to-absent for his own **linear** models — a direct match to TopoDCC's linear specification.

**21-day window test** (`lpnorm`, `--window 21`, default 80/20 split): TopoDCC(unreg) vs baseline QLIKE p=0.041, baseline wins. TopoDCC(reg) vs baseline: p=0.725, no difference. Matches the pattern from every other window tested — shortening to match Souto's exact window does not flip the result.

**Crisis-subperiod test** (`--split-date 2019-12-31`, `--window 21`): a structural problem was identified and fixed before this could even run — with the project's default 80/20 split, the test period lands around 2022-2025, meaning COVID, the GFC, and the EU debt crisis all fall entirely inside *training* data, not the held-out test set. `--split-date 2019-12-31` was added to `oos_evaluation.py` specifically to move the test period earlier, placing COVID (104 test-period dates) and the 2022 rate-hike period (251 dates) genuinely in-sample-for-testing — verified directly before trusting any result from it.

Result: **no crisis-concentrated advantage found.** Baseline DCC was nominally better than both TopoDCC variants in every crisis-restricted comparison (COVID: p=0.107 unreg, p=0.168 reg; Rates 2022: p=0.070 unreg, p=0.587 reg). This is the opposite of what Souto's theory would predict if the effect were present and merely diluted by full-sample averaging.

**Interpretation:** the crisis-concentration mechanism that produces Souto's positive result does not replicate for TopoDCC. This is consistent with — not contradictory to — Souto's own reported finding that linear models do not reliably show the same crisis-period benefit his neural models do. TopoDCC's linear encoder (`a_t = σ(w_a·φ_t)`) was a deliberate design choice made in July 2026, for good reasons at the time (avoiding overfitting risk that additional capacity would worsen). In hindsight, that same choice may be precisely what excludes the kind of effect a nonlinear encoder could capture. This is a genuine, citable tension for the thesis discussion section, not a resolved question — an MLP or other nonlinear encoder is the natural next step, explicitly informed by this contrast, and is noted as future work rather than pursued within the MSc timeline.

**Where this leaves the project:** a complete, thorough, multi-angle negative result — levels (10 window/feature combinations), speed (4 combinations), aDCC, and now the specific literature-motivated crisis-subperiod check — all converging on the same conclusion. TopoDCC's linear specification does not forecast correlation dynamics better than baseline DCC out-of-sample, under any configuration tested.
