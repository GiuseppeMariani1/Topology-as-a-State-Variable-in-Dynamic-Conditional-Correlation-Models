import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.loader import load_config
import numpy as np
import pandas as pd

config = load_config()
paths = config['paths']

topo = np.load(paths['dcc_topo_lpnorm'], allow_pickle=True).item()
w_a = topo['w_a']
w_b = topo['w_b']
X = pd.read_parquet(paths['tda_features_lpnorm']).values

print("=== Weight magnitude ===")
print(f"w_a: {w_a}")
print(f"||w_a|| = {np.linalg.norm(w_a):.6f}")
print(f"w_b: {w_b}")
print(f"||w_b|| = {np.linalg.norm(w_b):.6f}")
print()
print("(init was randn * 0.01, so ||w_a_init|| ~= 0.01 * sqrt(9) ~= 0.03)")
print()

print("=== Feature scale (unstandardized X) ===")
print(pd.DataFrame(X, columns=topo['feature_columns']).describe().T[['mean', 'std', 'min', 'max']])
print()

print("=== Raw linear signal into sigmoid: X @ w_a ===")
raw = X @ w_a
print(f"mean(X @ w_a) = {raw.mean():.6f}, std(X @ w_a) = {raw.std():.6f}")
print(f"For comparison, bias_a ~= -3.5 dominates unless X @ w_a swings by a similar magnitude")
print()

print("=== Log-likelihood training curve (last 20 of 500 iters) ===")
ll_hist = topo['ll_history']
print(ll_hist[-20:])
print(f"ll improvement over training: {ll_hist[-1] - ll_hist[0]:.4f}")
print(f"ll change in last 50 iters: {ll_hist[-1] - ll_hist[-50]:.6f}  (near zero = converged/stuck)")

config_models = config.get('models', {})
print()
print("=== Training config used ===")
print("topo:", config_models.get('topo'))
