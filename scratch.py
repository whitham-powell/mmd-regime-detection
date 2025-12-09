# scratch.py

# %%

import time
from sympy import plot
import torch
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# test_data = yf.download("SPY", start="2020-01-01", end="2024-12-31", auto_adjust=False)
# print(test_data.head())
# %%
df = yf.download(
    "SPY", period="5y", start="2020-01-01", auto_adjust=True, multi_level_index=False
)


# %% Compute features
# Base logs
log_open = np.log(df["Open"])
log_high = np.log(df["High"])
log_low = np.log(df["Low"])
log_close = np.log(df["Close"])
log_vol = np.log(df["Volume"])

# Primary time diffs
ret = log_close.diff()  # close-to-close return
d_log_vol = log_vol.diff()  # volume change

# Intraday shape
body = log_close - log_open  # C - O
range_ = log_high - log_low  # H - L
upper = log_high - log_close  # H - C
lower = log_close - log_low  # C - L

# Shape ratio
CLV = (log_close - log_low) / (log_high - log_low)  # close location in range

# Cross-day relationships
overnight = log_open - log_close.shift(1)  # O_t - C_{t-1}

true_range = pd.concat(
    [
        (log_high - log_low),
        (log_high - log_close.shift(1)).abs(),
        (log_low - log_close.shift(1)).abs(),
    ],
    axis=1,
).max(axis=1)

C_minus_yH = log_close - log_high.shift(1)
C_minus_yL = log_close - log_low.shift(1)
O_minus_yH = log_open - log_high.shift(1)
O_minus_yL = log_open - log_low.shift(1)

# Shape dynamics
d_body = body.diff()
d_range = range_.diff()
d_upper = upper.diff()
d_lower = lower.diff()
d_CLV = CLV.diff()

# Realized vol at multiple horizons
rv_10 = ret.rolling(10).std()
rv_30 = ret.rolling(30).std()
rv_60 = ret.rolling(60).std()
rv_90 = ret.rolling(90).std()

# Moving averages
sma_20 = log_close.rolling(20).mean()
sma_50 = log_close.rolling(50).mean()
sma_200 = log_close.rolling(200).mean()

# Moving True Range
mtr_20 = true_range.rolling(20).mean()
mtr_50 = true_range.rolling(50).mean()
mtr_200 = true_range.rolling(200).mean()

# %% Assemble feature dataframes

features_base = pd.DataFrame(
    {
        "log_open": log_open,
        "log_high": log_high,
        "log_low": log_low,
        "log_close": log_close,
        "log_vol": log_vol,
    }
).dropna()

features_intraday_shape = pd.DataFrame(
    {
        "body": body,
        "range": range_,
        "upper_wick": upper,  # calling this and the following "wick" is not quite accurate with respect to candle terminology, but it's close enough
        "lower_wick": lower,
        "CLV": CLV,
    }
).dropna()

features_cross_day = pd.DataFrame(
    {
        "overnight": overnight,
        "true_range": true_range,
        "C_minus_yH": C_minus_yH,
        "C_minus_yL": C_minus_yL,
        "O_minus_yH": O_minus_yH,
        "O_minus_yL": O_minus_yL,
    }
).dropna()

features_shape_dynamics = pd.DataFrame(
    {
        "d_body": d_body,
        "d_range": d_range,
        "d_upper_wick": d_upper,
        "d_lower_wick": d_lower,
        "d_CLV": d_CLV,
        "d_log_vol": d_log_vol,
    }
).dropna()


features_vol_structure = pd.DataFrame(
    {
        "rv_10": rv_10,
        "rv_30": rv_30,
        "rv_60": rv_60,
        "rv_90": rv_90,
    }
).dropna()

features_simple_moving_averages = pd.DataFrame(
    {
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,
    }
).dropna()

features_moving_true_range = pd.DataFrame(
    {
        "mtr_20": mtr_20,
        "mtr_50": mtr_50,
        "mtr_200": mtr_200,
    }
).dropna()

features_all = pd.DataFrame(
    {
        "log_open": log_open,
        "log_high": log_high,
        "log_low": log_low,
        "log_close": log_close,
        "log_vol": log_vol,
        "ret": ret,
        "d_log_vol": d_log_vol,
        "body": body,
        "range": range_,
        "upper_wick": upper,
        "lower_wick": lower,
        "CLV": CLV,
        "overnight": overnight,
        "true_range": true_range,
        "C_minus_yH": C_minus_yH,
        "C_minus_yL": C_minus_yL,
        "O_minus_yH": O_minus_yH,
        "O_minus_yL": O_minus_yL,
        "d_body": d_body,
        "d_range": d_range,
        "d_upper_wick": d_upper,
        "d_lower_wick": d_lower,
        "d_CLV": d_CLV,
        "rv_10": rv_10,
        "rv_30": rv_30,
        "rv_60": rv_60,
        "rv_90": rv_90,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "mtr_20": mtr_20,
        "mtr_50": mtr_50,
        "mtr_200": mtr_200,
    }
).dropna()

# %%

from scipy.spatial.distance import cdist


def rbf_kernel(X, Y=None, sigma=1.0):
    if Y is None:
        Y = X
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    D2 = cdist(X, Y, metric="sqeuclidean")
    gamma = 1 / (2 * sigma**2)
    return np.exp(-gamma * D2)
    # return np.exp(-D2 / (2 * sigma**2))


def mmd_squared(X, Y, gamma=1.0):
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    Kxx = rbf(X, X, gamma=gamma)
    Kyy = rbf(Y, Y, gamma=gamma)
    Kxy = rbf(X, Y, gamma=gamma)

    # print(f"Kxx shape: {Kxx.shape}, Kyy shape: {Kyy.shape}, Kxy shape: {Kxy.shape}")
    # print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    m, n = len(X), len(Y)

    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    return (
        Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2 * Kxy.sum() / (m * n)
    )


def mmd_permutation_test(X, Y, sigma=1.0, n_permutations=500):
    observed = mmd_squared(X, Y, gamma=sigma)
    pooled = np.vstack([X, Y])
    m = len(X)

    null_dist = []

    for _ in range(n_permutations):
        perm = np.random.permutation(len(pooled))
        null_dist.append(mmd_squared(pooled[perm[:m]], pooled[perm[m:]], gamma=sigma))

    null_dist = np.array(null_dist)
    p_value = (null_dist >= observed).mean()

    return observed, p_value, null_dist.mean(), null_dist.std()


def sliding_window_mmd(returns, window=60, step=5, sigma=1.0, n_perms=500):

    print(f"returns shape: {returns.shape}")
    returns = np.array(returns)  # .reshape(-1, 1)

    print(f"returns shape: {returns.shape}")
    n = len(returns)

    results = []

    for t in range(window, n - window, step):
        before = returns[t - window : t]
        after = returns[t : t + window]

        mmd, p_val, null_mean, null_std = mmd_permutation_test(
            before, after, sigma=sigma, n_permutations=n_perms
        )
        std_from_null = (mmd - null_mean) / null_std if null_std > 0 else 0

        results.append(
            {"t": t, "mmd": mmd, "p_val": p_val, "std_from_null": std_from_null}
        )

    return results


# %%
# returns = df["Close"].pct_change().dropna().values
returns = features_base.values
# returns = torch.Tensor(features_all.values)
import time

# returns = log_close.diff().dropna().values

sigma = np.median(np.abs(returns - np.median(returns)))

start_time = time.time()
results = sliding_window_mmd(returns, window=30, step=1, sigma=sigma, n_perms=1000)
end_time = time.time()
for r in results:
    if r["std_from_null"] > 2.0:
        print(
            f"Potential regime change at index {r['t']}: {r['std_from_null']:.2f} std from null"
        )

    # print(
    #     f"Time: {r['t']}, MMD: {r['mmd']:.4f}, p-value: {r['p_val']:.4f}, Std from null: {r['std_from_null']:.4f}"
    # )
print(f"Time taken: {end_time - start_time:.2f} seconds")

# %% Plot results

import matplotlib.pyplot as plt

std_from_nulls = [r["std_from_null"] for r in results]
times = [r["t"] for r in results]
plt.figure(figsize=(12, 6))
plt.plot(times, std_from_nulls, label="Std from Null")
plt.axhline(y=2.0, color="r", linestyle="--", label="2 Std Threshold")
plt.xlabel("Time")
plt.ylabel("Std from Null")
plt.title("Sliding Window MMD Analysis")
plt.legend()
plt.show()

# %% Plot mmds

mmds = [r["mmd"] for r in results]
plt.figure(figsize=(12, 6))
plt.plot(times, mmds, label="MMD")
plt.xlabel("Time")
plt.ylabel("MMD")
plt.title("Sliding Window MMD Analysis")
plt.legend()
plt.show()

# %% Plot p-values
p_values = [r["p_val"] for r in results]
plt.figure(figsize=(12, 6))
plt.plot(times, p_values, label="p-value")
plt.axhline(y=0.05, color="r", linestyle="--", label="0.05 Significance Level")
plt.xlabel("Time")
plt.ylabel("p-value")
plt.title("Sliding Window MMD Analysis")
plt.legend()
plt.show()

# %%
import pandas as pd

# assuming these DataFrames are already defined:
# features_base
# features_intraday_shape
# features_cross_day
# features_shape_dynamics
# features_vol_structure
# features_simple_moving_averages
# features_moving_true_range
# features_all  # optional convenience

FEATURE_GROUPS = {
    "base": features_base,
    "intraday_shape": features_intraday_shape,
    "cross_day": features_cross_day,
    "shape_dynamics": features_shape_dynamics,
    "vol_structure": features_vol_structure,
    "sma": features_simple_moving_averages,
    "moving_true_range": features_moving_true_range,
    "all": features_all,  # full library shortcut
}


def make_features(group_names, dropna=True):
    """
    Build a feature DataFrame from one or more named feature groups.

    Parameters
    ----------
    group_names : list[str] or str
        Name or list of names from FEATURE_GROUPS keys.
        e.g. ["base", "intraday_shape"] or "base".
    dropna : bool, default True
        If True, drop any rows with NaNs after concatenation
        (inner sample intersection).

    Returns
    -------
    pd.DataFrame
        Concatenated feature matrix with aligned index.
    """
    if isinstance(group_names, str):
        group_names = [group_names]

    dfs = []
    for name in group_names:
        if name not in FEATURE_GROUPS:
            raise ValueError(
                f"Unknown feature group: {name!r}. "
                f"Available: {list(FEATURE_GROUPS.keys())}"
            )
        dfs.append(FEATURE_GROUPS[name])

    # Align on index; inner join keeps only timestamps present in all groups
    features = pd.concat(dfs, axis=1, join="inner")

    if dropna:
        features = features.dropna()

    return features


# %%
# Example usage:
# 1) Raw-only (Tier 0)
X_base = make_features("base")

# 2) Base + minimal dynamics (base + shape dynamics)
X_dyn = make_features(["base", "shape_dynamics"])

# 3) Shape-only representation
X_shape = make_features("intraday_shape")

# 4) Big structural set: base + intraday shape + cross-day + vol structure
X_big = make_features(["base", "intraday_shape", "cross_day", "vol_structure"])

# 5) Full library (equivalent to features_all)
X_all = make_features("all")

# %%
import time
from kta import kta, kta_torch, alignment, alignment_torch, rbf, rbf_torch
import numpy as np
from src.mmd import sliding_window_mmd
from src.features import features_base
from src.plots import (
    plot_sliding_window_kta,
    plot_sliding_window_mmds,
    plot_sliding_window_pvals,
    plot_sliding_window_std_from_null,
)


signal = features_base.values
# signal = df["Close"].pct_change().dropna().values.reshape(-1, 1)
sigma = np.median(np.abs(signal - np.median(signal)))

kernel_params = {"gamma": 1.0 / (2 * sigma**2)}
kernel_fn = rbf

start_time = time.time()
results = sliding_window_mmd(
    data=signal,
    kernel_fn=kernel_fn,
    kernel_params=kernel_params,
    window=30,
    step=5,
    n_permutations=1000,
)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
# %%
plot_sliding_window_mmds(results)
plot_sliding_window_pvals(results)
plot_sliding_window_std_from_null(results)
plot_sliding_window_kta(results)

# %% plot kta values over time
kta_values = [r["kta_val"] for r in results]
times = [r["t"] for r in results]
plt.figure(figsize=(12, 6))
plt.plot(times, kta_values, label="KTA Value")
plt.xlabel("Time")
plt.ylabel("KTA Value")
plt.title("Sliding Window KTA Analysis")
plt.legend()
plt.show()

# %%
import src.features
