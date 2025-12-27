# features.py
"""
Feature engineering for regime detection.

Provides functions to compute various technical features from OHLCV data.
Features are organized into groups that can be combined as needed.
"""

import warnings
from typing import List, Union

import numpy as np
import pandas as pd


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute base log-transformed features from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with Open, High, Low, Close, Volume columns

    Returns
    -------
    pd.DataFrame
        Features: log_open, log_high, log_low, log_close, log_vol
    """
    return pd.DataFrame(
        {
            "log_open": np.log(df["Open"]),
            "log_high": np.log(df["High"]),
            "log_low": np.log(df["Low"]),
            "log_close": np.log(df["Close"]),
            "log_vol": np.log(df["Volume"]),
        },
        index=df.index,
    ).dropna()


def compute_intraday_shape(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute intraday candle shape features.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame

    Returns
    -------
    pd.DataFrame
        Features: body, range, upper_wick, lower_wick, CLV
    """
    log_open = np.log(df["Open"])
    log_high = np.log(df["High"])
    log_low = np.log(df["Low"])
    log_close = np.log(df["Close"])

    body = log_close - log_open
    range_ = log_high - log_low
    upper = log_high - log_close
    lower = log_close - log_low
    CLV = (log_close - log_low) / (log_high - log_low)

    return pd.DataFrame(
        {
            "body": body,
            "range": range_,
            "upper_wick": upper,
            "lower_wick": lower,
            "CLV": CLV,
        },
        index=df.index,
    ).dropna()


def compute_cross_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-day relationship features.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame

    Returns
    -------
    pd.DataFrame
        Features: overnight, true_range, C_minus_yH, C_minus_yL, O_minus_yH, O_minus_yL
    """
    log_open = np.log(df["Open"])
    log_high = np.log(df["High"])
    log_low = np.log(df["Low"])
    log_close = np.log(df["Close"])

    overnight = log_open - log_close.shift(1)

    true_range = pd.concat(
        [
            (log_high - log_low),
            (log_high - log_close.shift(1)).abs(),
            (log_low - log_close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return pd.DataFrame(
        {
            "overnight": overnight,
            "true_range": true_range,
            "C_minus_yH": log_close - log_high.shift(1),
            "C_minus_yL": log_close - log_low.shift(1),
            "O_minus_yH": log_open - log_high.shift(1),
            "O_minus_yL": log_open - log_low.shift(1),
        },
        index=df.index,
    ).dropna()


def compute_shape_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute shape dynamics (first differences of shape features).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame

    Returns
    -------
    pd.DataFrame
        Features: d_body, d_range, d_upper_wick, d_lower_wick, d_CLV, d_log_vol
    """
    shape = compute_intraday_shape(df)
    log_vol = np.log(df["Volume"])

    return pd.DataFrame(
        {
            "d_body": shape["body"].diff(),
            "d_range": shape["range"].diff(),
            "d_upper_wick": shape["upper_wick"].diff(),
            "d_lower_wick": shape["lower_wick"].diff(),
            "d_CLV": shape["CLV"].diff(),
            "d_log_vol": log_vol.diff(),
        },
        index=df.index,
    ).dropna()


def compute_vol_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute realized volatility at multiple horizons.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame

    Returns
    -------
    pd.DataFrame
        Features: rv_10, rv_30, rv_60, rv_90
    """
    log_close = np.log(df["Close"])
    ret = log_close.diff()

    return pd.DataFrame(
        {
            "rv_10": ret.rolling(10).std(),
            "rv_30": ret.rolling(30).std(),
            "rv_60": ret.rolling(60).std(),
            "rv_90": ret.rolling(90).std(),
        },
        index=df.index,
    ).dropna()


def compute_sma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple moving averages of log price.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame

    Returns
    -------
    pd.DataFrame
        Features: sma_20, sma_50, sma_200
    """
    log_close = np.log(df["Close"])

    return pd.DataFrame(
        {
            "sma_20": log_close.rolling(20).mean(),
            "sma_50": log_close.rolling(50).mean(),
            "sma_200": log_close.rolling(200).mean(),
        },
        index=df.index,
    ).dropna()


def compute_moving_true_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute moving averages of true range.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame

    Returns
    -------
    pd.DataFrame
        Features: mtr_20, mtr_50, mtr_200
    """
    cross_day = compute_cross_day(df)
    true_range = cross_day["true_range"]

    return pd.DataFrame(
        {
            "mtr_20": true_range.rolling(20).mean(),
            "mtr_50": true_range.rolling(50).mean(),
            "mtr_200": true_range.rolling(200).mean(),
        },
        index=df.index,
    ).dropna()


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all available features.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame

    Returns
    -------
    pd.DataFrame
        All 33 features combined
    """
    log_open = np.log(df["Open"])
    log_high = np.log(df["High"])
    log_low = np.log(df["Low"])
    log_close = np.log(df["Close"])
    log_vol = np.log(df["Volume"])

    ret = log_close.diff()
    d_log_vol = log_vol.diff()

    body = log_close - log_open
    range_ = log_high - log_low
    upper = log_high - log_close
    lower = log_close - log_low
    CLV = (log_close - log_low) / (log_high - log_low)

    overnight = log_open - log_close.shift(1)
    true_range = pd.concat(
        [
            (log_high - log_low),
            (log_high - log_close.shift(1)).abs(),
            (log_low - log_close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return pd.DataFrame(
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
            "C_minus_yH": log_close - log_high.shift(1),
            "C_minus_yL": log_close - log_low.shift(1),
            "O_minus_yH": log_open - log_high.shift(1),
            "O_minus_yL": log_open - log_low.shift(1),
            "d_body": body.diff(),
            "d_range": range_.diff(),
            "d_upper_wick": upper.diff(),
            "d_lower_wick": lower.diff(),
            "d_CLV": CLV.diff(),
            "rv_10": ret.rolling(10).std(),
            "rv_30": ret.rolling(30).std(),
            "rv_60": ret.rolling(60).std(),
            "rv_90": ret.rolling(90).std(),
            "sma_20": log_close.rolling(20).mean(),
            "sma_50": log_close.rolling(50).mean(),
            "sma_200": log_close.rolling(200).mean(),
            "mtr_20": true_range.rolling(20).mean(),
            "mtr_50": true_range.rolling(50).mean(),
            "mtr_200": true_range.rolling(200).mean(),
        },
        index=df.index,
    ).dropna()


# Mapping from group names to compute functions
FEATURE_GROUP_FUNCTIONS = {
    "base": compute_base_features,
    "intraday_shape": compute_intraday_shape,
    "cross_day": compute_cross_day,
    "shape_dynamics": compute_shape_dynamics,
    "vol_structure": compute_vol_structure,
    "sma": compute_sma,
    "moving_true_range": compute_moving_true_range,
    "all": compute_all_features,
}

# Public list of available groups
FEATURE_GROUPS = list(FEATURE_GROUP_FUNCTIONS.keys())


def make_features(
    df: pd.DataFrame,
    groups: Union[str, List[str]] = "base",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Build a feature DataFrame from one or more named feature groups.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with Open, High, Low, Close, Volume columns
    groups : str or list[str]
        Name or list of names from FEATURE_GROUPS.
        e.g. ["base", "intraday_shape"] or "base".
    dropna : bool, default True
        If True, drop any rows with NaNs after concatenation.

    Returns
    -------
    pd.DataFrame
        Concatenated feature matrix with aligned index.

    Examples
    --------
    >>> from regime_detection import load_ohlcv, make_features
    >>> df = load_ohlcv("AAPL", period="2y")
    >>> X = make_features(df, "base")
    >>> X = make_features(df, ["base", "vol_structure"])
    """
    if isinstance(groups, str):
        groups = [groups]

    dfs = []
    for name in groups:
        if name not in FEATURE_GROUP_FUNCTIONS:
            raise ValueError(
                f"Unknown feature group: {name!r}. " f"Available: {FEATURE_GROUPS}",
            )
        compute_fn = FEATURE_GROUP_FUNCTIONS[name]
        dfs.append(compute_fn(df))

    # Align on index; inner join keeps only timestamps present in all groups
    features = pd.concat(dfs, axis=1, join="inner")

    if dropna:
        features = features.dropna()

    return features


# =============================================================================
# Backwards Compatibility (Deprecated)
# =============================================================================

_legacy_df = None
_legacy_feature_groups = None


def _load_legacy_data():
    """Lazy load legacy SPY data for backwards compatibility."""
    global _legacy_df, _legacy_feature_groups

    if _legacy_df is None:
        warnings.warn(
            "Accessing module-level 'df' or legacy FEATURE_GROUPS dicts is deprecated. "
            "Use load_ohlcv() and make_features(df, group) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        import yfinance as yf

        _legacy_df = yf.download(
            "SPY",
            period="5y",
            start="2020-01-01",
            auto_adjust=True,
            multi_level_index=False,
            progress=False,
        )
        # Pre-compute legacy feature DataFrames
        _legacy_feature_groups = {
            name: fn(_legacy_df) for name, fn in FEATURE_GROUP_FUNCTIONS.items()
        }

    return _legacy_df, _legacy_feature_groups


def __getattr__(name: str):
    """
    Lazy loading for backwards compatibility.

    Allows old code like `from regime_detection.features import df` to still work,
    but with a deprecation warning.
    """
    if name == "df":
        df, _ = _load_legacy_data()
        return df
    elif name == "features_base":
        _, groups = _load_legacy_data()
        return groups["base"]
    elif name == "features_intraday_shape":
        _, groups = _load_legacy_data()
        return groups["intraday_shape"]
    elif name == "features_cross_day":
        _, groups = _load_legacy_data()
        return groups["cross_day"]
    elif name == "features_shape_dynamics":
        _, groups = _load_legacy_data()
        return groups["shape_dynamics"]
    elif name == "features_vol_structure":
        _, groups = _load_legacy_data()
        return groups["vol_structure"]
    elif name == "features_simple_moving_averages":
        _, groups = _load_legacy_data()
        return groups["sma"]
    elif name == "features_moving_true_range":
        _, groups = _load_legacy_data()
        return groups["moving_true_range"]
    elif name == "features_all":
        _, groups = _load_legacy_data()
        return groups["all"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    # Example usage with new API
    from regime_detection.data import load_ohlcv

    # Load any ticker
    df = load_ohlcv("AAPL", period="2y")
    print(f"Loaded {len(df)} rows for AAPL")

    # 1) Base features
    X_base = make_features(df, "base")
    print(f"\nBase features: {list(X_base.columns)}")
    print(X_base.head())

    # 2) Multiple groups
    X_combo = make_features(df, ["base", "vol_structure"])
    print(f"\nBase + vol_structure: {list(X_combo.columns)}")
    print(X_combo.head())

    # 3) All features
    X_all = make_features(df, "all")
    print(f"\nAll features ({len(X_all.columns)} total): {list(X_all.columns)}")
