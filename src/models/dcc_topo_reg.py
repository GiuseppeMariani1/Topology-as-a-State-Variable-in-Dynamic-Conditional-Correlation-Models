"""
Regularised Topology-DCC with Ridge (L2) penalty.

Why Ridge here:
  Landscape features are highly correlated by construction: adjacent grid
  points and adjacent landscape levels move together. Ridge handles collinear
  predictors well: it shrinks all weights proportionally rather than picking
  one arbitrarily (Lasso) or zeroing groups (Group-Lasso). Elastic Net would
  also work and is tried here as a comparison.

Workflow:
  1. Standardise features.
  2. 80/20 chronological train/test split.
  3. Grid-search over lambda_reg values.
  4. For each lambda: fit on train, evaluate log-likelihood on test.
  5. Save best model and results.

Performance notes (see PR discussion):
  - Early stopping with patience cuts iterations once train loss plateaus,
    instead of always running the full n_iter steps.
  - GPU is used automatically if available (torch.cuda.is_available()).
  - The 7 lambda fits in lambda_search are independent of each other, so
    they're run in parallel worker processes when on CPU. (On a single GPU
    we keep it sequential — multiple processes contending for one GPU
    context tends to be slower and flakier than just running in series.)
  - lambda_search no longer recomputes a_seq/b_seq after training; it
    reuses the pass already done at the end of fit_dcc_topo_reg.

Usage:
  python src/models/dcc_topo_reg.py
"""

import os
import sys
import concurrent.futures
import multiprocessing as mp

import torch
import numpy as np
import pandas as pd


from src.models.dcc_topo import TopoDCC, dcc_topo_recursion, compute_Q_bar, fit_dcc_topo_batched, dcc_topo_recursion_batched


# REGULARISED TRAINING

def fit_dcc_topo_reg(z_train, X_train, Q_bar,
                     lambda_l2=1e-2,
                     lambda_l1=0.0,
                     n_iter=500,
                     lr=0.01,
                     patience=30,
                     min_delta=1e-4,
                     device=None,
                     verbose=False):
    """
    Fit TopoDCC with elastic-net regularisation on training data.

    Loss = -ll(train) + lambda_l2 * ||w||_2^2 + lambda_l1 * ||w||_1

    Setting lambda_l1=0 gives pure Ridge.
    Setting lambda_l2=0 gives pure Lasso.

    Early stopping: training stops once `loss` hasn't improved by at least
    `min_delta` for `patience` consecutive iterations. The weights from the
    best iteration are restored at the end (not necessarily the last ones
    run), so a stall right after a good step doesn't lose progress.

    Returns: fitted model, ll_history (train), a_seq, b_seq
      a_seq/b_seq are the model's outputs on X_train under the *final
      restored* weights, computed once — callers should reuse these rather
      than calling model(X_train) again.
    """
    device = device or torch.device('cpu')
    z_train = z_train.to(device)
    X_train = X_train.to(device)
    Q_bar   = Q_bar.to(device)

    n_features = X_train.shape[1]
    model      = TopoDCC(n_features).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    ll_history = []

    best_loss  = float('inf')
    best_state = None
    stall      = 0

    for i in range(n_iter):
        optimizer.zero_grad()
        a_seq, b_seq = model(X_train)
        _, ll        = dcc_topo_recursion(z_train, a_seq, b_seq, Q_bar)

        l2_pen = model.w_a.pow(2).sum() + model.w_b.pow(2).sum()
        l1_pen = model.w_a.abs().sum()  + model.w_b.abs().sum()
        loss   = -ll + lambda_l2 * l2_pen + lambda_l1 * l1_pen

        loss.backward()
        optimizer.step()
        ll_history.append(ll.item())

        loss_val = loss.item()
        if loss_val < best_loss - min_delta:
            best_loss  = loss_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1

        if verbose and (i % 100 == 0 or i == n_iter - 1):
            a_seq_d, b_seq_d = a_seq.detach(), b_seq.detach()
            print(f"  iter {i:4d} | ll={ll.item():.2f} | "
                  f"a={a_seq_d.mean():.4f}±{a_seq_d.std():.4f} | "
                  f"b={b_seq_d.mean():.4f}±{b_seq_d.std():.4f}")

        if stall >= patience:
            if verbose:
                print(f"  early stop at iter {i} (no improvement for {patience} iters)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Single forward pass under final weights reused by callers instead
    # of being recomputed.
    with torch.no_grad():
        a_seq_final, b_seq_final = model(X_train)

    return model, ll_history, a_seq_final, b_seq_final

def eval_oos_ll(model, z_t, X_t, Q_bar, train_size, device=None):
    """
    Run the full DCC recursion over all data (so test period is warm-started
    from training history), then return the log-likelihood on the test portion.
    """
    device = device or torch.device('cpu')
    z_t   = z_t.to(device)
    X_t   = X_t.to(device)
    Q_bar = Q_bar.to(device)

    with torch.no_grad():
        a_seq, b_seq = model(X_t)
        R_seq, _     = dcc_topo_recursion(z_t, a_seq, b_seq, Q_bar)

        z_test = z_t[train_size:]
        R_test = R_seq[train_size:]

        sign, log_det = torch.linalg.slogdet(R_test)
        R_inv  = torch.linalg.inv(R_test)
        mahal  = torch.sum(
            z_test * (R_inv @ z_test.unsqueeze(-1)).squeeze(-1), dim=1
        )
        ll_test = -0.5 * (log_det.sum() + mahal.sum())

    return ll_test.item()


def eval_oos_ll_batched(model, z_t, X_t_batch, Q_bar, train_size, device=None):
    """
    Batched counterpart to eval_oos_ll: model is a TopoDCCBatched (B models),
    X_t_batch is (B, T, n_features) -- e.g. the same features repeated B
    times for a lambda search, or B different feature sets. z_t / Q_bar
    are shared across the batch, matching dcc_topo_recursion_batched.

    Returns: (B,) numpy array of out-of-sample log-likelihoods.
    """
    device = device or torch.device('cpu')
    z_t = z_t.to(device)
    X_t_batch = X_t_batch.to(device)
    Q_bar = Q_bar.to(device)

    with torch.no_grad():
        a_seq, b_seq = model(X_t_batch)                          # (B, T)
        R_seq, _ = dcc_topo_recursion_batched(z_t, a_seq, b_seq, Q_bar)  # (B, T, N, N)

        z_test = z_t[train_size:]                                # (T_test, N)
        R_test = R_seq[:, train_size:]                            # (B, T_test, N, N)
        B = R_test.shape[0]

        sign, log_det = torch.linalg.slogdet(R_test)              # (B, T_test)
        R_inv = torch.linalg.inv(R_test)                          # (B, T_test, N, N)

        z_test_exp = z_test.unsqueeze(0).expand(B, *z_test.shape)  # (B, T_test, N)
        mahal = torch.sum(
            z_test_exp * torch.einsum('btij,btj->bti', R_inv, z_test_exp), dim=2
        )  # (B, T_test)

        ll_test = -0.5 * (log_det.sum(dim=1) + mahal.sum(dim=1))  # (B,)

    return ll_test.detach().cpu().numpy()


# LAMBDA GRID SEARCH

LAMBDA_GRID = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1]


def _lambda_worker(lam, z_train_np, X_train_np, Qbar_np, z_t_np, X_t_np,
                    train_size, n_iter, lr, patience, min_delta):
    """
    Runs in a separate process. Pure-numpy in, plain dict out — avoids
    shipping torch tensors / CUDA state across the process boundary.
    """
    import torch as _torch  # local import: this runs in a fresh process
    _torch.set_num_threads(1)  # avoid every worker fighting for all cores

    from src.models.dcc_topo import dcc_topo_recursion as _  # noqa: F401 (sanity import)

    device = _torch.device('cpu')
    z_train = _torch.tensor(z_train_np, dtype=_torch.float32)
    X_train = _torch.tensor(X_train_np, dtype=_torch.float32)
    Q_bar   = _torch.tensor(Qbar_np,    dtype=_torch.float32)
    z_t     = _torch.tensor(z_t_np,     dtype=_torch.float32)
    X_t     = _torch.tensor(X_t_np,     dtype=_torch.float32)

    model, ll_hist, a_seq, b_seq = fit_dcc_topo_reg(
        z_train, X_train, Q_bar,
        lambda_l2=lam, n_iter=n_iter, lr=lr,
        patience=patience, min_delta=min_delta,
        device=device, verbose=False
    )
    ll_train = ll_hist[-1]
    ll_test  = eval_oos_ll(model, z_t, X_t, Q_bar, train_size, device=device)

    with _torch.no_grad():
        w_norm = (model.w_a.pow(2).sum() + model.w_b.pow(2).sum()).sqrt().item()
        a_std  = a_seq.std().item()
        b_std  = b_seq.std().item()

    return {
        'lambda_l2':   lam,
        'll_train':    round(ll_train, 2),
        'll_test':     round(ll_test, 2),
        '|w|_2':       round(w_norm, 4),
        'a_std':       round(a_std, 4),
        'b_std':       round(b_std, 4),
        'n_iters_run': len(ll_hist),
    }

def _lambda_search_gpu(z_t, X_train, X_t, Q_bar, train_size, lambda_grid,
                        n_iter, lr, device):
    """
    GPU path: fits one model per lambda value, all simultaneously as a
    single batched training run instead of sequential fits. Replaces the
    old "single GPU, sequential lambdas" fallback -- now the whole grid
    trains as one batched op, same principle as the permutation test's
    GPU path.
    """
    B = len(lambda_grid)
    n_features = X_train.shape[1]

    X_train_batch = X_train.unsqueeze(0).expand(B, *X_train.shape).contiguous()
    lambda_tensor = torch.tensor(lambda_grid, dtype=z_t.dtype)

    print(f"\n  [GPU batched] fitting all {B} lambda values as one batch on {device}...")
    model, ll_train_final, ll_hist = fit_dcc_topo_batched(
        z_t[:train_size], X_train_batch, n_iter=n_iter, lr=lr,
        lambda_l2=lambda_tensor, device=device, verbose=False
    )

    # fit_dcc_topo_batched moves its OWN local copies of these tensors to
    # `device` internally -- that doesn't affect these caller-side
    # variables, so they're still wherever they started (CPU). Move them
    # explicitly before reusing them against the now-GPU-resident model.
    X_train_batch = X_train_batch.to(device)
    X_t_batch = X_t.unsqueeze(0).expand(B, *X_t.shape).contiguous().to(device)

    ll_test = eval_oos_ll_batched(model, z_t, X_t_batch, Q_bar, train_size, device=device)

    with torch.no_grad():
        w_norm = (model.w_a.pow(2).sum(dim=1) + model.w_b.pow(2).sum(dim=1)).sqrt().cpu().numpy()
        a_seq, b_seq = model(X_train_batch)
        a_std = a_seq.std(dim=1).cpu().numpy()
        b_std = b_seq.std(dim=1).cpu().numpy()

    rows = []
    for i, lam in enumerate(lambda_grid):
        rows.append({
            'lambda_l2':   lam,
            'll_train':    round(float(ll_train_final[i]), 2),
            'll_test':     round(float(ll_test[i]), 2),
            '|w|_2':       round(float(w_norm[i]), 4),
            'a_std':       round(float(a_std[i]), 4),
            'b_std':       round(float(b_std[i]), 4),
            'n_iters_run': n_iter,  # batched path runs the full n_iter, no per-item early stop
        })
        print(f"    lambda_l2={lam:.0e}  ll_train={rows[-1]['ll_train']:.2f}  "
              f"ll_test={rows[-1]['ll_test']:.2f}  |w|={rows[-1]['|w|_2']:.4f}")

    return rows


def lambda_search(z_t, X_t, Q_bar, train_size,
                  lambda_grid=LAMBDA_GRID,
                  n_iter=500, lr=0.01,
                  patience=30, min_delta=1e-4,
                  n_jobs=None, device=None):
    """
    Fit one model per lambda value, report train and test log-likelihoods.
    Returns a DataFrame of results sorted by test ll descending.

    - On CUDA: all lambda fits run as one batched op (see _lambda_search_gpu).
      Early stopping/patience isn't applied per-item in the batched path
      (all items train for the full n_iter) -- batching trades that off
      for the much larger throughput win of one GPU op instead of B.
    - On CPU: lambda fits are independent, so they're run in parallel
      worker processes (n_jobs of them, default = min(len(grid), cpu_count)),
      each with its own early stopping via patience/min_delta.
    """
    device = device or torch.device('cpu')
    z_train = z_t[:train_size]
    X_train = X_t[:train_size]

    if device.type == 'cuda':
        rows = _lambda_search_gpu(z_t, X_train, X_t, Q_bar, train_size,
                                   lambda_grid, n_iter, lr, device)
        results_df = pd.DataFrame(rows).sort_values('ll_test', ascending=False)
        return results_df

    if n_jobs is None:
        n_jobs = max(1, min(len(lambda_grid), os.cpu_count() or 1))

    rows = []

    if n_jobs > 1:
        ctx = mp.get_context('spawn')
        z_train_np = z_train.cpu().numpy()
        X_train_np = X_train.cpu().numpy()
        Qbar_np    = Q_bar.cpu().numpy()
        z_t_np     = z_t.cpu().numpy()
        X_t_np     = X_t.cpu().numpy()

        print(f"\n  Running {len(lambda_grid)} lambda fits across {n_jobs} processes...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            futures = {
                ex.submit(_lambda_worker, lam, z_train_np, X_train_np, Qbar_np,
                          z_t_np, X_t_np, train_size, n_iter, lr, patience, min_delta): lam
                for lam in lambda_grid
            }
            for fut in concurrent.futures.as_completed(futures):
                row = fut.result()
                rows.append(row)
                print(f"    lambda_l2={row['lambda_l2']:.0e}  "
                      f"ll_train={row['ll_train']:.2f}  ll_test={row['ll_test']:.2f}  "
                      f"|w|={row['|w|_2']:.4f}  (stopped at iter {row['n_iters_run']})")
    else:
        for lam in lambda_grid:
            print(f"\n  lambda_l2={lam:.0e}  fitting...")
            model, ll_hist, a_seq, b_seq = fit_dcc_topo_reg(
                z_train, X_train, Q_bar,
                lambda_l2=lam, n_iter=n_iter, lr=lr,
                patience=patience, min_delta=min_delta,
                device=device, verbose=False
            )
            ll_train = ll_hist[-1]
            ll_test  = eval_oos_ll(model, z_t, X_t, Q_bar, train_size, device=device)

            with torch.no_grad():
                w_norm = (model.w_a.pow(2).sum() + model.w_b.pow(2).sum()).sqrt().item()
                a_std  = a_seq.std().item()
                b_std  = b_seq.std().item()

            rows.append({
                'lambda_l2':   lam,
                'll_train':    round(ll_train, 2),
                'll_test':     round(ll_test,  2),
                '|w|_2':       round(w_norm,   4),
                'a_std':       round(a_std,    4),
                'b_std':       round(b_std,    4),
                'n_iters_run': len(ll_hist),
            })
            print(f"    ll_train={ll_train:.2f}  ll_test={ll_test:.2f}  |w|={w_norm:.4f}  "
                  f"(stopped at iter {len(ll_hist)})")

    results_df = pd.DataFrame(rows).sort_values('ll_test', ascending=False)
    return results_df

# MAIN

if __name__ == "__main__":
    from src.data.loader import load_aligned_data, load_config, standardize_features

    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    garch_residuals, tda_features, paths = load_aligned_data(config)

    z_t = torch.tensor(garch_residuals.values, dtype=torch.float32)
    X_np, X_mean, X_std = standardize_features(tda_features)
    X_t = torch.tensor(X_np, dtype=torch.float32)

    topo_reg_cfg = config['models']['topo_reg']
    eval_cfg     = config['evaluation']

    patience  = topo_reg_cfg.get('patience', 30)
    min_delta = topo_reg_cfg.get('min_delta', 1e-4)
    n_jobs    = topo_reg_cfg.get('n_jobs', None)

    T = z_t.shape[0]
    train_size = int(T * (1.0 - eval_cfg['test_size']))
    print(f"\nTrain: {train_size} days  |  Test: {T - train_size} days")

    Q_bar = compute_Q_bar(z_t[:train_size])

    # Lambda grid search
    print("LAMBDA GRID SEARCH (Ridge regularisation)")

    results = lambda_search(
        z_t,
        X_t,
        Q_bar,
        train_size,
        lambda_grid=topo_reg_cfg['lambda_grid'],
        n_iter=topo_reg_cfg['n_iter'],
        lr=topo_reg_cfg['lr'],
        patience=patience,
        min_delta=min_delta,
        n_jobs=n_jobs,
        device=device,
    )

    print("RESULTS (sorted by out-of-sample ll)")
    print("_" * 60)
    print(results.to_string(index=False))

    # Fit best model
    best_lambda = results.iloc[0]['lambda_l2']
    print(f"\nBest lambda_l2 = {best_lambda:.0e}")
    print("Fitting final model on full data...")

    model, ll_hist, a_seq, b_seq = fit_dcc_topo_reg(
        z_t,
        X_t,
        compute_Q_bar(z_t),
        lambda_l2=best_lambda,
        n_iter=topo_reg_cfg['n_iter'],
        lr=topo_reg_cfg['lr'],
        patience=patience,
        min_delta=min_delta,
        device=device,
        verbose=True
    )

    with torch.no_grad():
        R_seq, _ = dcc_topo_recursion(z_t.to(device), a_seq, b_seq, compute_Q_bar(z_t).to(device))

    ll_final = ll_hist[-1]

    # CHANGED: baseline comparison now loaded from the actual current
    # dcc_baseline_results.npy instead of a hardcoded stale value
    # (-4786.44, left over from the old 5060-row/186-feature run).
    try:
        baseline_results = np.load(paths['dcc_baseline'], allow_pickle=True).item()
        baseline_ll = baseline_results['ll_final']
    except (FileNotFoundError, KeyError):
        baseline_ll = None
        print("\n  WARNING: could not load dcc_baseline_results.npy — "
              "run stage 4 first to get a baseline comparison.")

    print(f"\nFinal ll (full data): {ll_final:.2f}")
    if baseline_ll is not None:
        print(f"Baseline ll (from dcc_baseline_results.npy): {baseline_ll:.2f}")
        print(f"Improvement over baseline: {ll_final - baseline_ll:.2f}")

    # Save
    os.makedirs(os.path.dirname(paths['dcc_topo_reg']), exist_ok=True)
    np.save(paths['dcc_topo_reg'], {
        'a_seq':       a_seq.detach().cpu().numpy(),
        'b_seq':       b_seq.detach().cpu().numpy(),
        'R_seq':       R_seq.detach().cpu().numpy(),
        'll_history':  ll_hist,
        'll_final':    ll_final,
        'lambda_l2':   best_lambda,
        'w_a':         model.w_a.detach().cpu().numpy(),
        'w_b':         model.w_b.detach().cpu().numpy(),
        'lambda_results': results.to_dict(),
        'X_mean':      X_mean,
        'X_std':       X_std,
        'feature_columns': list(tda_features.columns),
    })
    print(f"Saved to {paths['dcc_topo_reg']}")