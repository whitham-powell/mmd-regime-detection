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
# # Kernel Comparison for Regime Detection
#
# This notebook compares how different kernel functions affect regime detection.
# We use the same sliding window MMD approach with RBF, polynomial, and linear kernels.

# %%
# =============================================================================
# Imports
# =============================================================================

import time
from typing import Callable, Dict, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kta import linear, polynomial, rbf
from sklearn.preprocessing import StandardScaler

from regime_detection import (
    find_regime_boundaries,
    load_ohlcv,
    make_features,
    results_to_dataframe,
    sliding_window_mmd,
)

# %%
# =============================================================================
# Configuration
# =============================================================================

TICKER = "SPY"
START_DATE = "2020-01-01"
END_DATE = None
PERIOD = None

# Sliding window parameters (same as demo.py for fair comparison)
WINDOW = 30
STEP = 5
N_PERMUTATIONS = 500  # Reduced for speed since we're running multiple kernels

# Boundary detection parameters
METRIC = "std_from_null"
THRESHOLD = 10.0
MIN_GAP_DAYS = 20

# Feature selection
FEATURE_GROUP = "base"

# Standardization
STANDARDIZE = True  # Recommended for kernel methods


# %%
# =============================================================================
# Data Loading
# =============================================================================

df = load_ohlcv(TICKER, start=START_DATE, end=END_DATE, period=PERIOD)
print(f"Loaded {TICKER}: {len(df)} rows")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")


# %%
# =============================================================================
# Data Preparation
# =============================================================================


def prepare_signal(
    df: pd.DataFrame,
    feature_group: str = "base",
    standardize: bool = True,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load and prepare feature matrix for MMD analysis.
    """
    features = make_features(df, feature_group)
    print(f"Feature group: {feature_group}")
    print(f"Features: {list(features.columns)}")
    print(f"Shape: {features.shape}")
    print(f"Date range: {features.index[0].date()} to {features.index[-1].date()}")

    values = features.values

    if standardize:
        scaler = StandardScaler()
        values = scaler.fit_transform(values)
        print("Standardization: Applied (zero mean, unit variance)")
    else:
        print("Standardization: None (raw features)")

    return values, features.index


# %%
# =============================================================================
# Kernel Definitions
# =============================================================================


def get_kernel_configs(signal: np.ndarray) -> Dict[str, Tuple[Callable, Dict]]:
    """
    Define kernels and their parameters for comparison.

    Returns dict of {name: (kernel_fn, kernel_params)}
    """
    # Median heuristic for RBF
    sigma = np.median(np.abs(signal - np.median(signal)))
    gamma_rbf = 1.0 / (2 * sigma**2)

    return {
        "RBF (median)": (rbf, {"gamma": gamma_rbf}),
        "Polynomial (d=2)": (polynomial, {"degree": 2, "c": 1.0}),
        "Polynomial (d=3)": (polynomial, {"degree": 3, "c": 1.0}),
        "Linear": (linear, {}),
    }


# %%
# =============================================================================
# Run Comparison
# =============================================================================

# Prepare data
signal, date_index = prepare_signal(df, FEATURE_GROUP, standardize=STANDARDIZE)
kernel_configs = get_kernel_configs(signal)

# Store results and metrics
results_by_kernel = {}
metrics_rows = []

for kernel_name, (kernel_fn, kernel_params) in kernel_configs.items():
    print(f"\nRunning {kernel_name}...")
    start_time = time.time()

    results = sliding_window_mmd(
        data=signal,
        kernel_fn=kernel_fn,
        kernel_params=kernel_params,
        window=WINDOW,
        step=STEP,
        n_permutations=N_PERMUTATIONS,
    )

    elapsed = time.time() - start_time
    results_df = results_to_dataframe(results, date_index)
    results_by_kernel[kernel_name] = results_df

    # Compute metrics
    boundaries = find_regime_boundaries(
        results_df,
        metric=METRIC,
        threshold=THRESHOLD,
        min_gap_days=MIN_GAP_DAYS,
    )

    metrics_rows.append(
        {
            "Kernel": kernel_name,
            "Boundaries": len(boundaries),
            "Peak Std": results_df["std_from_null"].max(),
            "Mean Std": results_df["std_from_null"].mean(),
            "Runtime (s)": elapsed,
        },
    )

    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Boundaries detected: {len(boundaries)}")

# %%
# =============================================================================
# Summary Table (DataFrame)
# =============================================================================

summary_df = pd.DataFrame(metrics_rows)
summary_df = summary_df.set_index("Kernel")
summary_df["Peak Std"] = summary_df["Peak Std"].round(2)
summary_df["Mean Std"] = summary_df["Mean Std"].round(2)
summary_df["Runtime (s)"] = summary_df["Runtime (s)"].round(1)

print("\n" + "=" * 70)
print("Kernel Comparison Summary")
print("=" * 70)
print(summary_df.to_string())

# %%
# =============================================================================
# Comparison Figure: Std from Null by Kernel
# =============================================================================

fig, axes = plt.subplots(
    len(kernel_configs),
    1,
    figsize=(14, 3 * len(kernel_configs)),
    sharex=True,
)

colors = ["blue", "green", "orange", "purple"]

for ax, (kernel_name, results_df), color in zip(
    axes,
    results_by_kernel.items(),
    colors,
):
    ax.plot(
        results_df.index,
        results_df["std_from_null"],
        color=color,
        lw=1,
        label=kernel_name,
    )
    ax.axhline(THRESHOLD, color="red", ls="--", lw=1, alpha=0.7)

    # Mark boundaries
    boundaries = find_regime_boundaries(
        results_df,
        metric=METRIC,
        threshold=THRESHOLD,
        min_gap_days=MIN_GAP_DAYS,
    )
    for b in boundaries:
        ax.axvline(b, color="red", alpha=0.3, lw=1)

    ax.set_ylabel("Std from Null", fontsize=10)
    ax.set_title(f"{kernel_name} ({len(boundaries)} boundaries)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

# Format x-axis
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
axes[-1].set_xlabel("Date", fontsize=10)

fig.suptitle(
    f"Kernel Comparison: Regime Detection (window={WINDOW}d, threshold={THRESHOLD})",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()
plt.show()


# %%
# =============================================================================
# Overlay Figure: All Kernels on One Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 5))

for (kernel_name, results_df), color in zip(results_by_kernel.items(), colors):
    ax.plot(
        results_df.index,
        results_df["std_from_null"],
        color=color,
        lw=1,
        label=kernel_name,
        alpha=0.8,
    )

ax.axhline(THRESHOLD, color="red", ls="--", lw=1.5, label=f"Threshold ({THRESHOLD})")
ax.set_ylabel("Std from Null", fontsize=10)
ax.set_xlabel("Date", fontsize=10)
ax.set_title(
    f"Kernel Comparison Overlay (window={WINDOW}d)",
    fontsize=12,
    fontweight="bold",
)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()


# %%
# =============================================================================
# Price with Boundaries by Kernel (Side-by-Side)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.flatten()

# Get price data aligned to results range
first_results = list(results_by_kernel.values())[0]
start_date = first_results.index[0]
end_date = first_results.index[-1]
price_aligned = df["Close"].loc[start_date:end_date]

for ax, (kernel_name, results_df), color in zip(
    axes,
    results_by_kernel.items(),
    colors,
):
    # Plot price
    ax.plot(price_aligned.index, price_aligned.values, "k-", lw=1)

    # Mark boundaries
    boundaries = find_regime_boundaries(
        results_df,
        metric=METRIC,
        threshold=THRESHOLD,
        min_gap_days=MIN_GAP_DAYS,
    )
    for b in boundaries:
        ax.axvline(b, color=color, alpha=0.7, lw=1.5, ls="--")

    ax.set_title(f"{kernel_name} ({len(boundaries)} boundaries)", fontsize=11)
    ax.grid(True, alpha=0.3)

# Format axes
for ax in axes[-2:]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Date", fontsize=10)

for ax in axes[::2]:
    ax.set_ylabel("Price", fontsize=10)

fig.suptitle(
    f"{TICKER} Detected Regime Boundaries by Kernel",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Standardization Effect
#
# Compare regime detection with and without feature standardization (RBF kernel only).

# %%
# =============================================================================
# Standardization Comparison (RBF only)
# =============================================================================

# Run RBF with and without standardization
standardization_results = {}

for standardize, label in [(True, "Standardized"), (False, "Raw")]:
    print(f"\nRunning RBF ({label})...")

    sig, idx = prepare_signal(df, FEATURE_GROUP, standardize=standardize)

    # Recompute median heuristic for this data
    sigma = np.median(np.abs(sig - np.median(sig)))
    gamma = 1.0 / (2 * sigma**2)

    start_time = time.time()
    results = sliding_window_mmd(
        data=sig,
        kernel_fn=rbf,
        kernel_params={"gamma": gamma},
        window=WINDOW,
        step=STEP,
        n_permutations=N_PERMUTATIONS,
    )
    elapsed = time.time() - start_time

    results_df = results_to_dataframe(results, idx)
    standardization_results[label] = {
        "results_df": results_df,
        "gamma": gamma,
        "runtime": elapsed,
    }

    boundaries = find_regime_boundaries(
        results_df,
        metric=METRIC,
        threshold=THRESHOLD,
        min_gap_days=MIN_GAP_DAYS,
    )
    print(f"  γ = {gamma:.4f}")
    print(f"  Boundaries: {len(boundaries)}")
    print(f"  Runtime: {elapsed:.1f}s")

# %%
# =============================================================================
# Standardization Comparison Figure
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

colors_std = {"Standardized": "blue", "Raw": "orange"}

for ax, (label, data) in zip(axes, standardization_results.items()):
    results_df = data["results_df"]
    color = colors_std[label]

    ax.plot(
        results_df.index,
        results_df["std_from_null"],
        color=color,
        lw=1,
        label=f"RBF ({label})",
    )
    ax.axhline(THRESHOLD, color="red", ls="--", lw=1, alpha=0.7)

    boundaries = find_regime_boundaries(
        results_df,
        metric=METRIC,
        threshold=THRESHOLD,
        min_gap_days=MIN_GAP_DAYS,
    )
    for b in boundaries:
        ax.axvline(b, color="red", alpha=0.3, lw=1)

    ax.set_ylabel("Std from Null", fontsize=10)
    ax.set_title(
        f"RBF {label} (γ={data['gamma']:.4f}, {len(boundaries)} boundaries)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
axes[-1].set_xlabel("Date", fontsize=10)

fig.suptitle(
    "Effect of Feature Standardization on Regime Detection",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# Standardization Summary Table
# =============================================================================

std_rows = []
for label, data in standardization_results.items():
    results_df = data["results_df"]
    boundaries = find_regime_boundaries(
        results_df,
        metric=METRIC,
        threshold=THRESHOLD,
        min_gap_days=MIN_GAP_DAYS,
    )
    std_rows.append(
        {
            "Features": label,
            "γ (median heuristic)": data["gamma"],
            "Boundaries": len(boundaries),
            "Peak Std": results_df["std_from_null"].max(),
            "Mean Std": results_df["std_from_null"].mean(),
        },
    )

std_summary_df = pd.DataFrame(std_rows).set_index("Features")
std_summary_df = std_summary_df.round(2)
print("\nStandardization Comparison:")
print(std_summary_df.to_string())

# %% [markdown]
# ## Observations
#
# ### Key Findings
#
# 1. **Kernel choice has limited impact on major regime detection**
#    - All kernels successfully detect the COVID crash (Feb-Mar 2020)
#    - Strong distributional shifts are captured regardless of kernel
#    - This suggests the signal is robust, not an artifact of kernel choice
#
# 2. **Sensitivity differences**
#    - RBF with median heuristic provides balanced sensitivity
#    - Linear kernel may miss subtle nonlinear regime changes
#    - Polynomial kernels can be more sensitive (more boundaries) or less,
#      depending on the degree and data characteristics
#
# 3. **Runtime considerations**
#    - Linear kernel is typically fastest (simple inner product)
#    - RBF and polynomial have similar complexity
#    - For real-time applications, linear may be preferred if it captures
#      the same major events
#
# 4. **Feature standardization matters**
#    - Standardized features typically detect more boundaries
#    - Without standardization, high-magnitude features (e.g., log_volume) dominate
#    - Standardization allows all features to contribute equally to kernel distances
#    - The median heuristic γ changes significantly with standardization
#
# ### Implications
#
# - For this dataset, **RBF with median heuristic is a reasonable default**
# - The convergence across kernels suggests the detected regimes are
#   **genuine distributional shifts**, not kernel-specific artifacts
# - **Feature standardization is recommended** for multi-feature inputs
# - Feature engineering (what goes into the kernel) may matter more than
#   kernel choice itself — see feature comparison experiments
#
# ### Limitations
#
# - Results may differ with different feature sets
# - Polynomial kernel parameters (degree, c) were not extensively tuned
# - Permutation count reduced for speed; final results should use more
