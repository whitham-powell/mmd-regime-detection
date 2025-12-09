# scratch.py

# %%
import time
from kta import kta, kta_torch, alignment, alignment_torch, rbf, rbf_torch
import numpy as np
from src.mmd import sliding_window_mmd
from src.features import features_base, df
from src.plots import (
    find_regime_boundaries,
    plot_regime_detection_panel,
    plot_regime_boundaries_summary,
    plot_sliding_window_kta,
    plot_sliding_window_mmds,
    plot_sliding_window_pvals,
    plot_sliding_window_std_from_null,
    results_to_dataframe,
)

# %% Configuration
np.random.seed(42)
WINDOW = 30
STEP = 5
N_PERMUTATIONS = 1000

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
    window=WINDOW,
    step=STEP,
    n_permutations=N_PERMUTATIONS,
)
elapsed = time.time() - start_time
print(f"Done in {elapsed:.1f} seconds ({len(results)} windows evaluated)")

# %% Convert results to DataFrame
results_df = results_to_dataframe(results, features_base.index)
print("\nResults summary:")
print(results_df.describe())

# %% Check COVID period specifically
print("\n=== COVID Period (Feb-Apr 2020) ===")
covid_period = results_df.loc["2020-02-01":"2020-04-30"]
if len(covid_period) > 0:
    print(covid_period[["mmd", "std_from_null", "kta_val"]])
    print(
        f"\nMax std_from_null during COVID: {covid_period['std_from_null'].max():.2f}"
    )
    print(f"Max MMD during COVID: {covid_period['mmd'].max():.4f}")
else:
    print("No data in this range (check your date alignment)")


# %%
# === Find regime boundaries ===
# Try different thresholds to see what works
for thresh in [5.0, 10.0, 15.0]:
    boundaries = find_regime_boundaries(
        results_df, metric="std_from_null", threshold=thresh, min_gap_days=20
    )
    print(f"\nThreshold={thresh}: {len(boundaries)} boundaries detected")
    if len(boundaries) > 0:
        for b in boundaries:
            print(f"  {b.strftime('%Y-%m-%d')}")

# %%
# === Main visualization: 4-panel plot ===
fig, axes = plot_regime_detection_panel(
    price_series=df["Close"],
    results_df=results_df,
    metric="std_from_null",
    threshold=10.0,  # adjust based on what you found above
    min_gap_days=20,
    title=f"Market Regime Detection via MMD (window={WINDOW}d)",
    save_path="regime_detection_panel.png",  # saves figure
)

# %%
# === Simpler summary plot ===
boundaries = find_regime_boundaries(
    results_df, metric="std_from_null", threshold=10.0, min_gap_days=20
)

fig2, ax2 = plot_regime_boundaries_summary(
    price_series=df["Close"],
    results_df=results_df,
    boundaries=boundaries,
    window_days=WINDOW,
)

# %%
plot_sliding_window_mmds(results)
plot_sliding_window_pvals(results)
plot_sliding_window_std_from_null(results)
plot_sliding_window_kta(results)

# %%
# %%
# === Quick experiment: compare different windows ===

WINDOWS_TO_TRY = [30, 60, 90]
comparison_results = {}

for w in WINDOWS_TO_TRY:
    print(f"Running window={w}...")
    res = sliding_window_mmd(
        data=signal,
        kernel_fn=kernel_fn,
        kernel_params=kernel_params,
        window=w,
        step=STEP,
        n_permutations=500,  # fewer for speed
    )
    comparison_results[w] = results_to_dataframe(res, features_base.index)

# Compare boundary detection across windows
for w, res_df in comparison_results.items():
    b = find_regime_boundaries(res_df, threshold=10.0)
    print(f"Window={w}: {len(b)} boundaries")

# %%


# %% wont run in this files just keeping save for later manual use
# =============================================================================
# LaTeX Export (for slides)
# =============================================================================

# Generate LaTeX table
latex_table = summary_df.to_latex(
    caption="Kernel Comparison for Regime Detection",
    label="tab:kernel_comparison",
    float_format="%.2f",
)
print("\nLaTeX Table:")
print("-" * 70)
print(latex_table)
