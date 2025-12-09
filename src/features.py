# features.py

import numpy as np
import pandas as pd
import yfinance as yf

# This moves all of the data and feature generation to its own module for easier importing elsewhere,
# should consider a smarter way to organize this later

# TODO: refactor data download and feature generation into functions or classes
df = yf.download(
    "SPY",
    period="5y",
    start="2020-01-01",
    auto_adjust=True,
    multi_level_index=False,
)
# Calculate features

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

# Assemble feature DataFrames
features_base = pd.DataFrame(
    {
        "log_open": log_open,
        "log_high": log_high,
        "log_low": log_low,
        "log_close": log_close,
        "log_vol": log_vol,
    },
).dropna()

features_intraday_shape = pd.DataFrame(
    {
        "body": body,
        "range": range_,
        "upper_wick": upper,  # calling this and the following "wick" is not quite accurate with respect to candle terminology, but it's close enough
        "lower_wick": lower,
        "CLV": CLV,
    },
).dropna()

features_cross_day = pd.DataFrame(
    {
        "overnight": overnight,
        "true_range": true_range,
        "C_minus_yH": C_minus_yH,
        "C_minus_yL": C_minus_yL,
        "O_minus_yH": O_minus_yH,
        "O_minus_yL": O_minus_yL,
    },
).dropna()

features_shape_dynamics = pd.DataFrame(
    {
        "d_body": d_body,
        "d_range": d_range,
        "d_upper_wick": d_upper,
        "d_lower_wick": d_lower,
        "d_CLV": d_CLV,
        "d_log_vol": d_log_vol,
    },
).dropna()

features_vol_structure = pd.DataFrame(
    {
        "rv_10": rv_10,
        "rv_30": rv_30,
        "rv_60": rv_60,
        "rv_90": rv_90,
    },
).dropna()

features_simple_moving_averages = pd.DataFrame(
    {
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,
    },
).dropna()

features_moving_true_range = pd.DataFrame(
    {
        "mtr_20": mtr_20,
        "mtr_50": mtr_50,
        "mtr_200": mtr_200,
    },
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
    },
).dropna()


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
                f"Available: {list(FEATURE_GROUPS.keys())}",
            )
        dfs.append(FEATURE_GROUPS[name])

    # Align on index; inner join keeps only timestamps present in all groups
    features = pd.concat(dfs, axis=1, join="inner")

    if dropna:
        features = features.dropna()

    return features


if __name__ == "__main__":

    # Example usage:
    # 1) Raw-only (Tier 0)
    X_base = make_features("base")
    print(X_base.head())
    # 2) Base + minimal dynamics (base + shape dynamics)
    X_dyn = make_features(["base", "shape_dynamics"])
    print(X_dyn.head())
    # 3) Shape-only representation
    X_shape = make_features("intraday_shape")
    print(X_shape.head())
    # 4) Big structural set: base + intraday shape + cross-day + vol structure
    X_big = make_features(["base", "intraday_shape", "cross_day", "vol_structure"])
    print(X_big.head())
    # 5) Full library (equivalent to features_all)
    X_all = make_features("all")
    print(X_all.head())
