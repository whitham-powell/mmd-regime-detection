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
# # Window Size Comparison for Regime Detection
#
# This notebook compares how different sliding window sizes affect regime detection.
# Smaller windows are more sensitive (detect more changes), larger windows are more robust.

# %%
# =============================================================================
# Imports
# =============================================================================

import time
from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kta import rbf
from sklearn.preprocessing import StandardScaler

from regime_detection import (
    df,
    find_regime_boundaries,
    make_features,
    results_to_dataframe,
    sliding_window_mmd,
)

# %%
# =============================================================================
# Configuration
# =============================================================================

# Window sizes to compare (in trading days)
WINDOWS = [20, 30, 45, 60, 90]

# Fixed parameters
STEP = 5
N_PERMUTATIONS = 500  # Reduced for speed

# Boundary detection
METRIC = "std_from_null"
THRESHOLD = 10.0
MIN_GAP_DAYS = 20

# Features
FEATURE_GROUP = "base"
STANDARDIZE = True


# %%
# =============================================================================
# Data Preparation
# =============================================================================


def prepare_signal(
    feature_group: str = "base",
    standardize: bool = True,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load and prepare feature matrix for MMD analysis.
    """
    features = make_features(feature_group)
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
# Run Comparison
# =============================================================================

# Prepare data
signal, date_index = prepare_signal(FEATURE_GROUP, standardize=STANDARDIZE)

# Compute kernel bandwidth (same for all windows)
sigma = np.median(np.abs(signal - np.median(signal)))
gamma = 1.0 / (2 * sigma**2)
print(f"\nRBF γ (median heuristic): {gamma:.4f}")

# Store results
results_by_window = {}
metrics_rows = []

for window in WINDOWS:
    print(f"\nRunning window={window}...")
    start_time = time.time()

    results = sliding_window_mmd(
        data=signal,
        kernel_fn=rbf,
        kernel_params={"gamma": gamma},
        window=window,
        step=STEP,
        n_permutations=N_PERMUTATIONS,
    )

    elapsed = time.time() - start_time
    results_df = results_to_dataframe(results, date_index)
    results_by_window[window] = results_df

    # Compute metrics
    boundaries = find_regime_boundaries(
        results_df,
        metric=METRIC,
        threshold=THRESHOLD,
        min_gap_days=MIN_GAP_DAYS,
    )

    metrics_rows.append(
        {
            "Window (days)": window,
            "Windows Evaluated": len(results),
            "Boundaries": len(boundaries),
            "Peak Std": results_df["std_from_null"].max(),
            "Mean Std": results_df["std_from_null"].mean(),
            "Runtime (s)": elapsed,
        },
    )

    print(f"  Windows evaluated: {len(results)}")
    print(f"  Boundaries detected: {len(boundaries)}")
    print(f"  Completed in {elapsed:.1f}s")

# %%
# =============================================================================
# Summary Table
# =============================================================================

summary_df = pd.DataFrame(metrics_rows)
summary_df = summary_df.set_index("Window (days)")
summary_df["Peak Std"] = summary_df["Peak Std"].round(2)
summary_df["Mean Std"] = summary_df["Mean Std"].round(2)
summary_df["Runtime (s)"] = summary_df["Runtime (s)"].round(1)

print("\n" + "=" * 70)
print("Window Size Comparison Summary")
print("=" * 70)
print(summary_df.to_string())


# %%
# =============================================================================
# Comparison Figure: Std from Null by Window Size
# =============================================================================

fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(14, 3 * len(WINDOWS)), sharex=True)

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(WINDOWS)))

for ax, (window, results_df), color in zip(axes, results_by_window.items(), colors):
    ax.plot(
        results_df.index,
        results_df["std_from_null"],
        color=color,
        lw=1,
        label=f"window={window}d",
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
    ax.set_title(f"Window = {window} days ({len(boundaries)} boundaries)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

# Format x-axis
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
axes[-1].set_xlabel("Date", fontsize=10)

fig.suptitle(
    f"Window Size Comparison: Regime Detection (RBF, threshold={THRESHOLD})",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()
plt.show()


# %%
# =============================================================================
# Overlay Figure: All Window Sizes
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 5))

for (window, results_df), color in zip(results_by_window.items(), colors):
    ax.plot(
        results_df.index,
        results_df["std_from_null"],
        color=color,
        lw=1,
        label=f"{window}d",
        alpha=0.8,
    )

ax.axhline(THRESHOLD, color="red", ls="--", lw=1.5, label=f"Threshold ({THRESHOLD})")
ax.set_ylabel("Std from Null", fontsize=10)
ax.set_xlabel("Date", fontsize=10)
ax.set_title(
    "Window Size Comparison Overlay (RBF kernel)",
    fontsize=12,
    fontweight="bold",
)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9, title="Window")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()


# %%
# =============================================================================
# Price with Boundaries by Window Size
# =============================================================================

n_windows = len(WINDOWS)
n_cols = 2
n_rows = (n_windows + 1) // n_cols

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(14, 4 * n_rows),
    sharex=True,
    sharey=True,
)
axes = axes.flatten()

# Get price data aligned to smallest window results (most restrictive range)
largest_window = max(WINDOWS)
results_smallest = results_by_window[largest_window]
start_date = results_smallest.index[0]
end_date = results_smallest.index[-1]
price_aligned = df["Close"].loc[start_date:end_date]

for ax, (window, results_df), color in zip(axes, results_by_window.items(), colors):
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
        if start_date <= b <= end_date:
            ax.axvline(b, color=color, alpha=0.7, lw=1.5, ls="--")

    ax.set_title(f"Window = {window}d ({len(boundaries)} boundaries)", fontsize=11)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for ax in axes[n_windows:]:
    ax.set_visible(False)

# Format axes
for ax in axes[n_cols * (n_rows - 1) : n_windows]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Date", fontsize=10)

for i, ax in enumerate(axes[:n_windows]):
    if i % n_cols == 0:
        ax.set_ylabel("Price", fontsize=10)

fig.suptitle(
    "Detected Regime Boundaries by Window Size",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout()
plt.show()


# %%
# =============================================================================
# Boundary Count vs Window Size
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 5))

window_sizes = list(results_by_window.keys())
boundary_counts = [
    len(
        find_regime_boundaries(
            results_df,
            metric=METRIC,
            threshold=THRESHOLD,
            min_gap_days=MIN_GAP_DAYS,
        ),
    )
    for results_df in results_by_window.values()
]

ax.plot(window_sizes, boundary_counts, "o-", markersize=8, lw=2, color="steelblue")
ax.set_xlabel("Window Size (days)", fontsize=11)
ax.set_ylabel("Boundaries Detected", fontsize=11)
ax.set_title(
    f"Sensitivity vs Window Size (threshold={THRESHOLD})",
    fontsize=12,
    fontweight="bold",
)
ax.grid(True, alpha=0.3)

# Annotate points
for w, count in zip(window_sizes, boundary_counts):
    ax.annotate(
        str(count),
        (w, count),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=10,
    )

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Step Size Comparison
#
# How does step size affect detection with a fixed window?

# %%
# =============================================================================
# Step Size Comparison (fixed window=30)
# =============================================================================

STEPS_TO_TRY = [1, 5, 10, 15]
FIXED_WINDOW = 30

step_results = {}
step_metrics = []

for step in STEPS_TO_TRY:
    print(f"\nRunning step={step}...")
    start_time = time.time()

    results = sliding_window_mmd(
        data=signal,
        kernel_fn=rbf,
        kernel_params={"gamma": gamma},
        window=FIXED_WINDOW,
        step=step,
        n_permutations=N_PERMUTATIONS,
    )

    elapsed = time.time() - start_time
    results_df = results_to_dataframe(results, date_index)
    step_results[step] = results_df

    boundaries = find_regime_boundaries(
        results_df,
        metric=METRIC,
        threshold=THRESHOLD,
        min_gap_days=MIN_GAP_DAYS,
    )

    step_metrics.append(
        {
            "Step (days)": step,
            "Windows Evaluated": len(results),
            "Boundaries": len(boundaries),
            "Peak Std": results_df["std_from_null"].max(),
            "Runtime (s)": elapsed,
        },
    )

    print(
        f"  Windows: {len(results)}, Boundaries: {len(boundaries)}, Time: {elapsed:.1f}s",
    )

# %%
# =============================================================================
# Step Size Summary Table
# =============================================================================

step_summary_df = pd.DataFrame(step_metrics).set_index("Step (days)")
step_summary_df["Peak Std"] = step_summary_df["Peak Std"].round(2)
step_summary_df["Runtime (s)"] = step_summary_df["Runtime (s)"].round(1)

print("\n" + "=" * 70)
print(f"Step Size Comparison (window={FIXED_WINDOW}d)")
print("=" * 70)
print(step_summary_df.to_string())

# %%
# =============================================================================
# Step Size Overlay Figure
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 5))

step_colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(STEPS_TO_TRY)))

for (step, results_df), color in zip(step_results.items(), step_colors):
    ax.plot(
        results_df.index,
        results_df["std_from_null"],
        color=color,
        lw=1,
        label=f"step={step}d",
        alpha=0.8,
    )

ax.axhline(THRESHOLD, color="red", ls="--", lw=1.5, label=f"Threshold ({THRESHOLD})")
ax.set_ylabel("Std from Null", fontsize=10)
ax.set_xlabel("Date", fontsize=10)
ax.set_title(
    f"Step Size Comparison (window={FIXED_WINDOW}d, RBF kernel)",
    fontsize=12,
    fontweight="bold",
)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9, title="Step")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# Runtime vs Step Size
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

steps = [m["Step (days)"] for m in step_metrics]
runtimes = [m["Runtime (s)"] for m in step_metrics]
n_windows = [m["Windows Evaluated"] for m in step_metrics]
n_boundaries = [m["Boundaries"] for m in step_metrics]

# Runtime
ax1.plot(steps, runtimes, "o-", markersize=8, lw=2, color="steelblue")
ax1.set_xlabel("Step Size (days)", fontsize=11)
ax1.set_ylabel("Runtime (s)", fontsize=11)
ax1.set_title("Runtime vs Step Size", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)
for s, r in zip(steps, runtimes):
    ax1.annotate(
        f"{r:.1f}s",
        (s, r),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

# Boundaries
ax2.plot(steps, n_boundaries, "o-", markersize=8, lw=2, color="darkgreen")
ax2.set_xlabel("Step Size (days)", fontsize=11)
ax2.set_ylabel("Boundaries Detected", fontsize=11)
ax2.set_title("Boundaries vs Step Size", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)
for s, b in zip(steps, n_boundaries):
    ax2.annotate(
        str(b),
        (s, b),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Observations
#
# ### Window Size Findings
#
# 1. **Larger windows = more boundaries detected**
#    - 20-day windows: ~23 boundaries
#    - 60-90 day windows: ~68-69 boundaries
#    - This is the opposite of naive intuition
#
# 2. **Why larger windows detect more**
#    - More samples per window → more statistical power
#    - Permutation null distribution becomes tighter with more samples
#    - Smaller distributional differences become statistically significant
#    - Std from null is systematically higher for larger windows
#
# 3. **Smaller windows have less power**
#    - 20-day windows: std_from_null rarely exceeds 30
#    - 90-day windows: std_from_null regularly reaches 60-80
#    - Same threshold (10.0) means different effective sensitivity
#
# 4. **All windows detect major events**
#    - COVID crash visible across all window sizes
#    - Larger windows show it more prominently (higher peak)
#
# ### Step Size Findings
#
# 5. **Step size affects both runtime and sensitivity**
#    - Step=1: 48 boundaries, ~35s runtime
#    - Step=5: 45 boundaries, ~7s runtime (5x faster, 6% fewer boundaries)
#    - Step=10: 33 boundaries, ~3.5s runtime (10x faster, 31% fewer boundaries)
#    - Step=15: 32 boundaries, ~2.3s runtime (15x faster, 33% fewer boundaries)
#
# 6. **Larger steps act as implicit smoothing**
#    - Coarser resolution skips short-lived peaks
#    - Useful for reducing noise if detection is too sensitive
#    - Step=5 is a good balance: minimal boundary loss, significant speedup
#
# 7. **Step size as a tuning parameter**
#    - Too many boundaries? Increase step size (cheaper than re-tuning threshold)
#    - Need fine temporal precision? Use step=1
#
# ### Implications
#
# - **Threshold should scale with window size** for comparable sensitivity
# - Larger windows are better for detecting *any* regime change (more power)
# - Smaller windows may miss subtle shifts but are more conservative
# - **Step size is a cheap way to reduce noisy detections** (before adjusting threshold)
# - For apples-to-apples comparison, consider normalizing by window-specific null distribution
#
# ### Limitations
#
# - Fixed threshold across window sizes is not ideal
# - Larger windows have shorter valid date range (edges trimmed)
# - Runtime increases with window size and decreases with step size
# - Window x step interaction not fully explored
