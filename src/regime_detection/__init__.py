# src/regime_detection/__init__.py
"""
Regime Detection via Maximum Mean Discrepancy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Market regime detection using kernel two-sample tests.

Convenience re-exports so users can do:

    from regime_detection import sliding_window_mmd, mmd_squared
    from regime_detection import make_features, prepare_signal
    from regime_detection import plot_regime_detection_panel
"""

from .features import FEATURE_GROUPS, df, make_features
from .mmd import compute_kta, mmd_permutation_test, mmd_squared, sliding_window_mmd
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
    # mmd
    "compute_kta",
    "mmd_permutation_test",
    "mmd_squared",
    "sliding_window_mmd",
    # features
    "df",
    "make_features",
    "FEATURE_GROUPS",
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
