# scratch.py

# %%

from sympy import O
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

    return np.exp(-D2 / (2 * sigma**2))


def mmd_squared(X, Y, sigma=1.0):
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    Kxx = rbf_kernel(X, X, sigma=sigma)
    Kyy = rbf_kernel(Y, Y, sigma=sigma)
    Kxy = rbf_kernel(X, Y, sigma=sigma)

    print(f"Kxx shape: {Kxx.shape}, Kyy shape: {Kyy.shape}, Kxy shape: {Kxy.shape}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    m, n = len(X), len(Y)

    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    return (
        Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2 * Kxy.sum() / (m * n)
    )


def mmd_permutation_test(X, Y, sigma=1.0, n_permutations=500):
    observed = mmd_squared(X, Y, sigma=sigma)
    pooled = np.vstack([X, Y])
    m = len(X)

    null_dist = []

    for _ in range(n_permutations):
        perm = np.random.permutation(len(pooled))
        null_dist.append(mmd_squared(pooled[perm[:m]], pooled[perm[m:]], sigma=sigma))

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
# returns = features_base.values
returns = features_all.values
import time

# returns = log_close.diff().dropna().values

sigma = np.median(np.abs(returns - np.median(returns)))

start_time = time.time()
results = sliding_window_mmd(returns, window=10, step=1, sigma=sigma, n_perms=1000)
end_time = time.time()
for r in results:
    if r["std_from_null"] > 2.0:
        print(
            f"Potential regime change at index {r['t']}: {r['std_from_null']:.2f} std from null"
        )

    print(
        f"Time: {r['t']}, MMD: {r['mmd']:.4f}, p-value: {r['p_val']:.4f}, Std from null: {r['std_from_null']:.4f}"
    )
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

# log_close = np.log(df["Close"])
# log_open = np.log(df["Open"])
# log_high = np.log(df["High"])
# log_low = np.log(df["Low"])
# log_vol = np.log(df["Volume"])

# features_rawish = pd.DataFrame(
#     {
#         "log_open": log_open,
#         "log_high": log_high,
#         "log_low": log_low,
#         "log_close": log_close,
#         "d_log_vol": log_vol.diff(),
#         "ret": log_close.diff(),
#     }
# ).dropna()

# # Engineered features
# body = log_close - log_open
# range_ = log_high - log_low
# upper = log_high - log_close
# lower = log_close - log_low
# CLV = (log_close - log_low) / (log_high - log_low)

# overnight = log_open - log_close.shift(1)
# intraday = log_close - log_open
# gap = overnight
# ret = log_close.diff()
# oo_ret = log_open.diff()

# true_range = pd.concat(
#     [
#         (log_high - log_low),
#         (log_high - log_close.shift(1)).abs(),
#         (log_low - log_close.shift(1)).abs(),
#     ],
#     axis=1,
# ).max(axis=1)

# C_minus_yH = log_close - log_high.shift(1)
# C_minus_yL = log_close - log_low.shift(1)

# d_body = body.diff()
# d_range = range_.diff()
# d_upper = upper.diff()
# d_lower = lower.diff()
# d_CLV = CLV.diff()

# d_log_vol = log_vol.diff()
# rv_20 = log_close.diff().rolling(20).std()
# rv_60 = log_close.diff().rolling(60).std()
# rv_120 = log_close.diff().rolling(120).std()

# features_minimal = pd.DataFrame(
#     {
#         "body": log_close - log_open,
#         "range": log_high - log_low,
#         "upper_wick": log_high - log_close,
#         "lower_wick": log_open - log_low,
#         "ret": log_close.diff(),
#         "d_log_vol": log_vol.diff(),
#     }
# ).dropna()

# features_maximal = pd.DataFrame(
#     {
#         "body": body,
#         "range": range_,
#         "upper_wick": upper,
#         "lower_wick": lower,
#         "CLV": CLV,
#         "overnight": overnight,
#         "intraday": intraday,
#         "ret": ret,
#         "oo_ret": oo_ret,
#         "gap": gap,
#         "true_range": true_range,
#         "C_minus_yH": log_close - log_high.shift(1),
#         "C_minus_yL": log_close - log_low.shift(1),
#         "d_body": body.diff(),
#         "d_range": range_.diff(),
#         "d_upper_wick": upper.diff(),
#         "d_lower_wick": lower.diff(),
#         "d_CLV": CLV.diff(),
#         "d_log_vol": log_vol.diff(),
#         "realized_vol20": log_close.diff().rolling(20).std(),
#     }
# ).dropna()

# # pd.DataFrame(
# #     {
# #         "ret": log_close.diff(),
# #         "d_log_vol": log_vol.diff(),
# #         "open": log_open.diff(),
# #         "high": log_high.diff(),
# #         "low": log_low.diff(),
# #         # "close": log_close.diff(),
# #         "volatility": log_close.diff().rolling(window=20).std(),
# #         "log_HO": log_high - log_open,
# #         "log_LO": log_low - log_open,
# #         # "log_CO": log_close - log_open,
# #         "log_HL": log_high - log_low,
# #     },
# # ).dropna()
# #

# %%
