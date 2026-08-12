import pandas as pd
import numpy as np
import yaml
import os

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def download_prices(tickers, start_date, end_date):
    """Download adjusted closing prices from yfinance."""
    # Imported lazily: yfinance is only needed for this one function, but
    # every downstream module imports this file for load_config /
    # load_aligned_data. A module-scope import would make yfinance a hard
    # dependency of the entire pipeline for scripts that never download
    # anything.
    import yfinance as yf

    print(f"Downloading: {tickers}")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    prices = raw["Close"]
    prices.dropna(how="all", inplace=True)
    print(f"Downloaded {prices.shape[0]} days x {prices.shape[1]} assets")
    return prices

def compute_log_returns(prices):
    """Compute log returns from price series."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"Log returns shape: {log_returns.shape}")
    return log_returns

def save_data(df, path):
    """Save dataframe to parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    print(f"Saved to {path}")


# SHARED LOADING / PREP HELPERS
#
# These replace blocks that were previously copy-pasted verbatim across
# dcc_topo.py, dcc_topo_reg.py, permutation_test.py and several scripts.

def load_aligned_data(config=None, features='lpnorm', verbose=True):
    """
    Load GARCH residuals + TDA features and align them to shared dates.
    """
    if config is None:
        config = load_config()
    paths = config['paths']

    key = {
        'lpnorm':    'tda_features_lpnorm',
        'landscape': 'tda_features_landscape',
        'pi':        'tda_features_pi',
    }[features]

    garch_residuals = pd.read_parquet(paths['garch_residuals'])
    tda_features    = pd.read_parquet(paths[key])

    common = garch_residuals.index.intersection(tda_features.index)
    garch_residuals = garch_residuals.loc[common]
    tda_features    = tda_features.loc[common]

    if len(common) == 0:
        raise ValueError(
            "No overlapping dates between GARCH residuals and TDA features -- "
            "re-run the TDA pipeline and lp_norm_features.py."
        )

    if 'betti_0' in tda_features.columns:
        tda_features = tda_features.drop(columns=['betti_0'])

    if verbose:
        print(f"Residuals: {garch_residuals.shape}")
        print(f"Features:  {tda_features.shape}  ({features})")
        print(f"(aligned to {len(common)} shared dates)")

    return garch_residuals, tda_features, paths


def standardize_features(tda_features):
    """
    Zero-mean/unit-variance the feature matrix.
    Returns (X_std_array, X_mean, X_std).
    """
    X_raw  = tda_features.values if hasattr(tda_features, 'values') else tda_features
    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0) + 1e-8
    return (X_raw - X_mean) / X_std, X_mean, X_std

def run(config_path="config/config.yaml"):
    # Load config
    config = load_config(config_path)
    tickers    = config["data"]["tickers"]
    start_date = config["data"]["start_date"]
    end_date   = config["data"]["end_date"]

    # Download
    prices = download_prices(tickers, start_date, end_date)

    # Log returns
    log_returns = compute_log_returns(prices)

    # Save
    paths = config['paths']
    save_data(prices,      paths['prices'])
    save_data(log_returns, paths['log_returns'])

    return prices, log_returns

if __name__ == "__main__":
    run()