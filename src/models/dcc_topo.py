import torch
import torch.nn as nn
import numpy as np
import pandas as pd


def compute_Q_bar(z_t):
    T = z_t.shape[0]
    return (z_t.T @ z_t) / T


def dcc_topo_recursion(z_t, a_seq, b_seq, Q_bar):
    T, N = z_t.shape
    Q_t = Q_bar.clone()
    Q_seq = torch.zeros(T, N, N, dtype=z_t.dtype)
    Q_seq[0] = Q_t

    for t in range(1, T):
        a_t = a_seq[t]
        b_t = b_seq[t]
        z_outer = torch.outer(z_t[t-1], z_t[t-1])
        Q_t = (1 - a_t - b_t) * Q_bar + a_t * z_outer + b_t * Q_t
        Q_seq[t] = Q_t

    diag_Q = torch.sqrt(torch.diagonal(Q_seq, dim1=1, dim2=2))
    R_seq = Q_seq / torch.einsum('ti,tj->tij', diag_Q, diag_Q)

    sign, log_det = torch.linalg.slogdet(R_seq)
    R_inv = torch.linalg.inv(R_seq)
    mahal = torch.sum(z_t * (R_inv @ z_t.unsqueeze(-1)).squeeze(-1), dim=1)
    ll = -0.5 * (log_det.sum() + mahal.sum())

    return R_seq, ll


class TopoDCC(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.w_a = nn.Parameter(torch.randn(n_features) * 0.01)
        self.w_b = nn.Parameter(torch.randn(n_features) * 0.01)
        # Biases initialised to reproduce baseline DCC values
        # sigmoid(-3.5) ≈ 0.03  ≈ baseline a
        # sigmoid(2.9)  ≈ 0.95  ≈ baseline b
        self.bias_a = nn.Parameter(torch.tensor(-3.5))
        self.bias_b = nn.Parameter(torch.tensor(2.9))

    def forward(self, X_t):
        # Independent sigmoids — a and b no longer compete via softmax
        a_raw = torch.sigmoid(X_t @ self.w_a + self.bias_a)
        b_raw = torch.sigmoid(X_t @ self.w_b + self.bias_b)

        # Soft constraint: rescale only when a+b would exceed 0.9998
        total = a_raw + b_raw + 1e-6
        exceed = (total > 0.9998).float()
        a_t = a_raw * (1 - exceed) + a_raw * (0.9998 / total) * exceed
        b_t = b_raw * (1 - exceed) + b_raw * (0.9998 / total) * exceed

        return a_t, b_t


def fit_dcc_topo(garch_residuals_df, tda_features_df,
                 n_iter=500, lr=0.01, verbose=True,):

    z_t = torch.tensor(garch_residuals_df.values, dtype=torch.float32)

    X_raw = tda_features_df.values
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0) + 1e-8
    X_t = torch.tensor((X_raw - X_mean) / X_std, dtype=torch.float32)

    T, N = z_t.shape
    n_features = X_t.shape[1]

    Q_bar = compute_Q_bar(z_t)
    model = TopoDCC(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ll_history = []

    for i in range(n_iter):
        optimizer.zero_grad()
        a_seq, b_seq = model(X_t)
        R_seq, ll = dcc_topo_recursion(z_t, a_seq, b_seq, Q_bar)
        loss = -ll
        loss.backward()
        optimizer.step()
        ll_history.append(ll.item())

        if verbose and (i % 50 == 0 or i == n_iter - 1):
            a_std = a_seq.std().item()
            b_std = b_seq.std().item()
            print(
                f"  Iter {i:4d} | "
                f"a_t mean={a_seq.mean().item():.4f} std={a_std:.4f} | "
                f"b_t mean={b_seq.mean().item():.4f} std={b_std:.4f} | "
                f"a+b mean={(a_seq+b_seq).mean().item():.4f} | "
                f"ll={ll.item():.2f}"
            )

    with torch.no_grad():
        a_seq, b_seq = model(X_t)
        R_seq, _ = dcc_topo_recursion(z_t, a_seq, b_seq, Q_bar)

    return model, a_seq, b_seq, R_seq, ll_history

def compute_r2(a_seq, b_seq, X_t_np):
    """
    R² of topology features explaining variation in a_t and b_t.
    Uses simple OLS on the normalised features.
    """
    from sklearn.linear_model import LinearRegression

    a_np = a_seq.detach().numpy()
    b_np = b_seq.detach().numpy()

    reg_a = LinearRegression().fit(X_t_np, a_np)
    reg_b = LinearRegression().fit(X_t_np, b_np)

    r2_a = reg_a.score(X_t_np, a_np)
    r2_b = reg_b.score(X_t_np, b_np)

    print(f"  R² (X_t → a_t): {r2_a:.4f} ({r2_a*100:.1f}%)")
    print(f"  R² (X_t → b_t): {r2_b:.4f} ({r2_b*100:.1f}%)")

    return r2_a, r2_b

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.data.loader import load_config

    config = load_config()
    model_cfg = config['models']['topo']
    paths = config['paths']

    
    garch_residuals = pd.read_parquet(paths['garch_residuals'])

    # L^p-norm reduced features (literature-matched, H1-only)
    # the raw 186-column landscape file used in the run that failed the
    # permutation test. 9 columns: lh1_k0_norm, lh1_k1_norm, lh1_k2_norm
    # + the 6 scalar features (betti_1, entropy_h0, entropy_h1,
    # max_persistence, total_persistence, wasserstein).
    tda_features = pd.read_parquet(
        paths.get('tda_features_lpnorm', 'data/processed/tda_features_lpnorm.parquet')
    )

    common = garch_residuals.index.intersection(tda_features.index)
    garch_residuals = garch_residuals.loc[common]
    tda_features = tda_features.loc[common]

    assert (garch_residuals.index == tda_features.index).all(), \
        "Index mismatch between residuals and TDA features"

    print(f"Residuals shape: {garch_residuals.shape}")
    print(f"Features shape:  {tda_features.shape}")
    print(f"Feature columns: {list(tda_features.columns)}")

    model, a_seq, b_seq, R_seq, ll_history = fit_dcc_topo(
        garch_residuals,
        tda_features,
        n_iter=model_cfg['n_iter'],
        lr=model_cfg['lr'],
        verbose=True
    )

    print(f"\nFinal ll: {ll_history[-1]:.2f}")
    print(f"Improvement over baseline (-8247.96): {ll_history[-1] - (-8247.96):.2f}")

    # R² diagnostics
    X_raw = tda_features.values
    X_t_np = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-8)
    print("\nR² diagnostics:")
    r2_a, r2_b = compute_r2(a_seq, b_seq, X_t_np)

    out_path = paths.get('dcc_topo_lpnorm', 'data/processed/dcc_topo_lpnorm_results.npy')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, {
        'a_seq':      a_seq.detach().numpy(),
        'b_seq':      b_seq.detach().numpy(),
        'R_seq':      R_seq.detach().numpy(),
        'll_history': ll_history,
        'll_final':   ll_history[-1],
        'w_a':        model.w_a.detach().numpy(),
        'w_b':        model.w_b.detach().numpy(),
        'feature_columns': list(tda_features.columns),
    })
    print(f"Saved to {out_path}")