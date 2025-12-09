# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
# ---

# %% [markdown]
# # Market Regime Detection via Maximum Mean Discrepancy (MMD)
#
# This notebook applies a sliding window MMD two-sample test to detect
# distributional regime changes in SPY returns. Detected boundaries
# are validated against known market events (e.g., COVID-19 crash).

# %%
# =============================================================================
# Imports
# =============================================================================

import time
from typing import Dict, List

import numpy as np
import pandas as pd
from kta import rbf
from sklearn.preprocessing import StandardScaler

from src.features import df, make_features
from src.mmd import sliding_window_mmd
from src.plots import (
    find_regime_boundaries,
    plot_regime_boundaries_summary,
    plot_regime_detection_panel,
    results_to_dataframe,
)

# %%
# =============================================================================
# Configuration
# =============================================================================

# Sliding window parameters
WINDOW = 30  # days in each window (compare before vs after)
STEP = 5  # days between successive windows
N_PERMUTATIONS = 1000  # permutations for null distribution (reduce for speed)

# Boundary detection parameters
METRIC = "std_from_null"  # metric to threshold: 'std_from_null', 'mmd', 'kta_val'
THRESHOLD = 10.0  # values above this are flagged as boundaries
MIN_GAP_DAYS = 20  # merge detections within this many days

# Feature selection
FEATURE_GROUP = "base"  # options: 'base', 'intraday_shape', 'vol_structure', 'all'

# Standardization
STANDARDIZE = True  # Use StandardScaler to standardize features


# %%
# =============================================================================
# Data Preparation
# =============================================================================


def prepare_signal(
    feature_group: str = "base",
    standardize: bool = True,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load and prepare feature matrix for MMD analysis.

    Parameters
    ----------
    feature_group : str
        Name of feature group to load
    standardize : bool
        If True, standardize features to zero mean and unit variance.
        Recommended for kernel methods to prevent scale-dominated distances.

    Returns
    -------
    signal : np.ndarray
        Feature matrix (n_samples, n_features)
    index : pd.DatetimeIndex
        Corresponding dates
    """
    features = make_features(feature_group)
    print(f"Feature group: {feature_group}")
    print(f"Features: {list(features.columns)}")
    print(f"Shape: {features.shape}")
    print(f"Date range: {features.index[0].date()} to {features.index[-1].date()}")

    values = features.values

    if standardize:
        scalar = StandardScaler()
        values = scalar.fit_transform(values)
        print("Standardization: Applied (zero mean, unit variance)")
    else:
        print("Standardization: None (raw features)")

    return values, features.index


def compute_kernel_bandwidth(signal: np.ndarray) -> float:
    """
    Compute RBF bandwidth using median heuristic.
    """
    sigma = np.median(np.abs(signal - np.median(signal)))
    gamma = 1.0 / (2 * sigma**2)
    print(f"Median heuristic: sigma={sigma:.6f}, gamma={gamma:.4f}")
    return gamma


# %%
# =============================================================================
# Run Analysis
# =============================================================================


def run_sliding_window_analysis(
    signal: np.ndarray,
    kernel_fn,
    kernel_params: Dict,
    window: int,
    step: int,
    n_permutations: int,
) -> List[Dict]:
    """
    Execute sliding window MMD with timing.
    """
    n_windows = (len(signal) - 2 * window) // step
    print("\nRunning sliding window MMD...")
    print(f"  Window size: {window} days")
    print(f"  Step size: {step} days")
    print(f"  Permutations: {n_permutations}")
    print(f"  Estimated windows: ~{n_windows}")

    start_time = time.time()

    results = sliding_window_mmd(
        data=signal,
        kernel_fn=kernel_fn,
        kernel_params=kernel_params,
        window=window,
        step=step,
        n_permutations=n_permutations,
    )

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({len(results)} windows)")

    return results


# %%
# =============================================================================
# Validation
# =============================================================================

# Known market events for validation (approximate dates)
KNOWN_EVENTS = {
    "COVID Crash": ("2020-02-19", "2020-03-23"),
    "COVID Recovery": ("2020-03-24", "2020-08-31"),
    "2022 Drawdown Start": ("2022-01-03", "2022-01-31"),
    "2022 Bottom": ("2022-09-01", "2022-10-31"),
    "2023 Rally": ("2023-01-01", "2023-03-31"),
}


def validate_against_known_events(
    results_df: pd.DataFrame,
    events: Dict[str, tuple] = KNOWN_EVENTS,
) -> pd.DataFrame:
    """
    Check if detected signals correspond to known market events.
    """
    print("\n" + "=" * 60)
    print("Validation Against Known Events")
    print("=" * 60)

    validation_rows = []

    for event_name, (start, end) in events.items():
        try:
            period = results_df.loc[start:end]
            if len(period) == 0:
                print(f"\n{event_name}: No data in range {start} to {end}")
                continue

            max_std = period["std_from_null"].max()
            max_mmd = period["mmd"].max()
            max_kta = period["kta_val"].max()
            max_date = period["std_from_null"].idxmax()

            print(f"\n{event_name} ({start} to {end}):")
            print(f"  Peak std_from_null: {max_std:.2f} on {max_date.date()}")
            print(f"  Peak MMD: {max_mmd:.4f}")
            print(f"  Peak KTA: {max_kta:.3f}")

            validation_rows.append(
                {
                    "event": event_name,
                    "start": start,
                    "end": end,
                    "peak_std": max_std,
                    "peak_mmd": max_mmd,
                    "peak_kta": max_kta,
                    "peak_date": max_date,
                },
            )
        except KeyError:
            print(f"\n{event_name}: Date range outside data")

    return pd.DataFrame(validation_rows)


def summarize_boundaries(
    boundaries: pd.DatetimeIndex,
    results_df: pd.DataFrame,
) -> None:
    """
    Print summary of detected regime boundaries.
    """
    print("\n" + "=" * 60)
    print(f"Detected Regime Boundaries (n={len(boundaries)})")
    print("=" * 60)

    if len(boundaries) == 0:
        print("No boundaries detected. Consider lowering the threshold.")
        return

    for i, b in enumerate(boundaries, 1):
        row = results_df.loc[b]
        print(
            f"  {i}. {b.date()}  |  std={row['std_from_null']:.1f}  |  MMD={row['mmd']:.4f}",
        )


# %%
# =============================================================================
# Main Execution
# =============================================================================

# --- Prepare data ---
signal, date_index = prepare_signal(FEATURE_GROUP, standardize=STANDARDIZE)
gamma = compute_kernel_bandwidth(signal)
kernel_params = {"gamma": gamma}

# --- Run MMD analysis ---
results = run_sliding_window_analysis(
    signal=signal,
    kernel_fn=rbf,
    kernel_params=kernel_params,
    window=WINDOW,
    step=STEP,
    n_permutations=N_PERMUTATIONS,
)

# --- Convert to DataFrame with dates ---
results_df = results_to_dataframe(results, date_index)

print("\n" + "=" * 60)
print("Results Summary Statistics")
print("=" * 60)
print(results_df[["mmd", "std_from_null", "kta_val"]].describe().round(3))

# %%
# =============================================================================
# Validation Against Known Events
# =============================================================================

validation_df = validate_against_known_events(results_df)

# %%
# =============================================================================
# Detect Boundaries
# =============================================================================

boundaries = find_regime_boundaries(
    results_df,
    metric=METRIC,
    threshold=THRESHOLD,
    min_gap_days=MIN_GAP_DAYS,
)
summarize_boundaries(boundaries, results_df)

# %%
# =============================================================================
# Main Figure: 4-Panel Regime Detection
# =============================================================================

fig1, axes1 = plot_regime_detection_panel(
    price_series=df["Close"],
    results_df=results_df,
    metric=METRIC,
    threshold=THRESHOLD,
    min_gap_days=MIN_GAP_DAYS,
    title=f"Market Regime Detection via MMD (window={WINDOW}d, {FEATURE_GROUP} features)",
)

# %%
# =============================================================================
# Summary Figure: Price with Boundaries
# =============================================================================

fig2, ax2 = plot_regime_boundaries_summary(
    price_series=df["Close"],
    results_df=results_df,
    boundaries=boundaries,
    window_days=WINDOW,
)

# %%
# =============================================================================
# Threshold Sensitivity
# =============================================================================

print("Threshold Sensitivity Analysis")
print("-" * 40)
for thresh in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
    b = find_regime_boundaries(
        results_df,
        threshold=thresh,
        min_gap_days=MIN_GAP_DAYS,
    )
    print(f"  threshold={thresh:5.1f}  →  {len(b):2d} boundaries")

# %%
