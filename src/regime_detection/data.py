# data.py
"""
Data loading utilities for regime detection.

Provides a clean interface for loading OHLCV data from yfinance
without hardcoding tickers or date ranges.
"""

from typing import Optional

import pandas as pd
import yfinance as yf


def load_ohlcv(
    ticker: str = "SPY",
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "5y",
) -> pd.DataFrame:
    """
    Load OHLCV data from yfinance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (default: "SPY")
    start : str, optional
        Start date in 'YYYY-MM-DD' format. If provided, `period` is ignored.
    end : str, optional
        End date in 'YYYY-MM-DD' format. Defaults to today if start is provided.
    period : str, optional
        Period to download (e.g., '1y', '5y', 'max'). Used only if start is None.
        Default: '5y'

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with columns: Open, High, Low, Close, Volume
        Index is DatetimeIndex.

    Examples
    --------
    >>> df = load_ohlcv("AAPL", period="2y")
    >>> df = load_ohlcv("MSFT", start="2020-01-01", end="2024-01-01")
    >>> df = load_ohlcv()  # SPY, last 5 years
    """
    if start is not None:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            multi_level_index=False,
            progress=False,
        )
    else:
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            multi_level_index=False,
            progress=False,
        )

    validate_ohlcv(df, ticker)
    return df


def validate_ohlcv(df: pd.DataFrame, ticker: str = "unknown") -> None:
    """
    Validate that DataFrame has required OHLCV columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    ticker : str
        Ticker symbol for error messages

    Raises
    ------
    ValueError
        If required columns are missing or DataFrame is empty
    """
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {missing}. "
            f"Got: {list(df.columns)}",
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"DataFrame index must be DatetimeIndex, got {type(df.index).__name__}",
        )
