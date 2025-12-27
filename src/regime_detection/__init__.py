# src/regime_detection/__init__.py
"""
Regime Detection via Maximum Mean Discrepancy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Market regime detection using kernel two-sample tests.

Quick Start
-----------
>>> from regime_detection import load_ohlcv, make_features, sliding_window_mmd
>>> from kta import rbf
>>> import numpy as np
>>>
>>> # Load data for any ticker
>>> df = load_ohlcv("AAPL", period="2y")
>>>
>>> # Generate features
>>> features = make_features(df, "base")
>>>
>>> # Run sliding window MMD
>>> sigma = np.median(np.abs(features.values))
>>> gamma = 1.0 / (2 * sigma**2)
>>> results = sliding_window_mmd(
...     data=features.values,
...     kernel_fn=rbf,
...     kernel_params={"gamma": gamma},
... )
"""

# Data loading
from .data import load_ohlcv, validate_ohlcv

# Feature engineering
from .features import (
    FEATURE_GROUPS,
    compute_all_features,
    compute_base_features,
    compute_cross_day,
    compute_intraday_shape,
    compute_moving_true_range,
    compute_shape_dynamics,
    compute_sma,
    compute_vol_structure,
    make_features,
)

# MMD computation
from .mmd import compute_kta, mmd_permutation_test, mmd_squared, sliding_window_mmd

# Visualization
from .plots import (
    find_regime_boundaries,
    plot_regime_boundaries_summary,
    plot_regime_detection_panel,
    plot_sliding_window_kta,
    plot_sliding_window_mmds,
    plot_sliding_window_pvals,
    plot_sliding_window_std_from_null,
    results_to_dataframe,
)

__all__ = [
    # data
    "load_ohlcv",
    "validate_ohlcv",
    # features
    "make_features",
    "FEATURE_GROUPS",
    "compute_base_features",
    "compute_intraday_shape",
    "compute_cross_day",
    "compute_shape_dynamics",
    "compute_vol_structure",
    "compute_sma",
    "compute_moving_true_range",
    "compute_all_features",
    # mmd
    "compute_kta",
    "mmd_permutation_test",
    "mmd_squared",
    "sliding_window_mmd",
    # plots
    "find_regime_boundaries",
    "plot_regime_boundaries_summary",
    "plot_regime_detection_panel",
    "plot_sliding_window_kta",
    "plot_sliding_window_mmds",
    "plot_sliding_window_pvals",
    "plot_sliding_window_std_from_null",
    "results_to_dataframe",
]

__version__ = "0.2.0"
