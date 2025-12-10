# MMD Regime Detection

Market regime detection using Maximum Mean Discrepancy (MMD) with kernel methods.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This package implements a nonparametric approach to detecting regime changes in financial time series. Rather than assuming a specific parametric model (e.g., hidden Markov models with Gaussian emissions), we use **Maximum Mean Discrepancy (MMD)** as a kernel two-sample test to identify when the distribution of market behavior has shifted.

The method compares sliding windows of data before and after each time point, flagging significant distributional differences as regime boundaries.

## Key Features

- **Nonparametric detection**: No assumptions about the form of distributional change
- **Sliding window MMD**: Compare before/after windows at each candidate change point
- **Permutation testing**: Rigorous statistical significance via permutation null distribution
- **Multiple kernels**: RBF (Gaussian), polynomial, and linear kernels supported
- **Rich feature engineering**: Log prices, intraday shape, volatility structure, and more
- **Visualization tools**: Publication-ready plots for regime boundaries and diagnostics

## Installation

### From GitHub

```bash
pip install git+https://github.com/whitham-powell/mmd-regime-detection.git
```

### For Development

```bash
git clone https://github.com/whitham-powell/mmd-regime-detection.git
cd mmd-regime-detection

# Using uv (recommended)
uv sync

# Or using pip
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from kta import rbf

from regime_detection import (
    df,
    make_features,
    sliding_window_mmd,
    find_regime_boundaries,
    plot_regime_detection_panel,
    results_to_dataframe,
)

# 1. Load and prepare features
features = make_features("base")  # log prices + log volume
scaler = StandardScaler()
signal = scaler.fit_transform(features.values)

# 2. Set kernel bandwidth via median heuristic
sigma = np.median(np.abs(signal - np.median(signal)))
gamma = 1.0 / (2 * sigma**2)

# 3. Run sliding window MMD
results = sliding_window_mmd(
    data=signal,
    kernel_fn=rbf,
    kernel_params={"gamma": gamma},
    window=30,           # 30 trading days per window
    step=5,              # test every 5 days
    n_permutations=500,  # permutations for null distribution
)

# 4. Convert to DataFrame and find boundaries
results_df = results_to_dataframe(results, features.index)
boundaries = find_regime_boundaries(results_df, threshold=10.0)

# 5. Visualize
fig, axes = plot_regime_detection_panel(
    price_series=df["Close"],
    results_df=results_df,
    threshold=10.0,
)
```

## How It Works

### Maximum Mean Discrepancy

MMD measures the distance between two probability distributions by comparing their embeddings in a reproducing kernel Hilbert space (RKHS):

$$\text{MMD}(P, Q) = \|\mu_P - \mu_Q\|_{\mathcal{H}}$$

where $\mu_P = \mathbb{E}_{X \sim P}[k(\cdot, X)]$ is the **kernel mean embedding** of distribution $P$.

For a **characteristic kernel** (e.g., RBF), MMD = 0 if and only if $P = Q$, making it a powerful tool for detecting any distributional difference.

### Sliding Window Detection

At each candidate change point $t$:

1. Extract **before window**: observations from $[t - w, t)$
2. Extract **after window**: observations from $[t, t + w)$
3. Compute MMD² between the two samples
4. Run permutation test to assess significance
5. Flag as boundary if test statistic exceeds threshold

### Permutation Test

Under the null hypothesis $H_0: P = Q$:

1. Pool samples: $Z = X \cup Y$
2. Randomly permute and split into pseudo-samples
3. Compute MMD² for permuted data
4. Repeat to build null distribution
5. Report p-value or standard deviations from null mean

## Configuration

### Main Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `window` | Days per window (before and after) | 30 | 20–60 |
| `step` | Days between tests | 5 | 1–10 |
| `n_permutations` | Permutations for null distribution | 500 | 200–1000 |
| `threshold` | Std from null to flag boundary | 10.0 | 5–15 |
| `min_gap_days` | Merge nearby detections | 20 | 10–40 |

### Feature Groups

```python
from regime_detection import make_features, FEATURE_GROUPS

# Available groups
print(FEATURE_GROUPS.keys())
# ['base', 'intraday_shape', 'cross_day', 'shape_dynamics',
#  'vol_structure', 'sma', 'moving_true_range', 'all']

# Single group
X = make_features("base")

# Multiple groups
X = make_features(["base", "vol_structure"])
```

| Group | Features |
|-------|----------|
| `base` | log_open, log_high, log_low, log_close, log_vol |
| `intraday_shape` | body, range, upper_wick, lower_wick, CLV |
| `vol_structure` | rv_10, rv_30, rv_60, rv_90 (realized volatility) |
| `all` | All 33 features |

### Kernel Selection

```python
from kta import rbf, polynomial, linear

# RBF (Gaussian) - recommended default
results = sliding_window_mmd(..., kernel_fn=rbf, kernel_params={"gamma": gamma})

# Polynomial
results = sliding_window_mmd(..., kernel_fn=polynomial, kernel_params={"degree": 2, "c": 1.0})

# Linear
results = sliding_window_mmd(..., kernel_fn=linear, kernel_params={})
```

## Project Structure

```
mmd-regime-detection/
├── src/regime_detection/
│   ├── __init__.py      # Public API exports
│   ├── mmd.py           # Core MMD computation and sliding window
│   ├── features.py      # Data loading and feature engineering
│   └── plots.py         # Visualization functions
├── notebooks/
│   ├── demo.py          # Main demonstration notebook
│   └── experiments/
│       ├── kernel_comparison.py   # Compare RBF, polynomial, linear
│       └── window_comparison.py   # Analyze window/step size effects
├── tests/
├── pyproject.toml
├── Makefile
└── README.md
```

## Notebooks

All notebooks use [Jupytext](https://jupytext.readthedocs.io/) for version control (`.py` files synced with `.ipynb`).

| Notebook | Description |
|----------|-------------|
| `demo.py` | End-to-end regime detection on SPY |
| `kernel_comparison.py` | Compare kernels and standardization effects |
| `window_comparison.py` | Analyze window size and step size sensitivity |

```bash
# Sync .py ↔ .ipynb
make sync

# Execute notebooks and extract figures
make plots
```

## Development

```bash
# Install with dev dependencies
uv sync  # or: pip install -e ".[dev]"

# Run tests
make test

# Pre-commit hooks (black, isort, flake8)
pre-commit install
pre-commit run --all-files
```

## Results

The method successfully detects known market regime changes:

| Detected Boundary | Market Event |
|-------------------|--------------|
| Feb 2020 | COVID-19 crash onset |
| Mar–Apr 2020 | Fed intervention / recovery |
| Jan 2022 | 2022 bear market start |
| Oct 2022 | 2022 market bottom |
| Oct–Nov 2023 | Bull market acceleration |

### Parameter Sensitivity

**Window size**: Larger windows have more statistical power but less temporal precision. Counter-intuitively, larger windows detect *more* boundaries because the permutation null distribution becomes tighter.

**Step size**: Primarily affects runtime and acts as implicit smoothing. Step=5 offers ~5× speedup over step=1 with minimal boundary loss.

**Kernel choice**: All kernels detect major events (COVID crash). RBF with median heuristic is a robust default. Feature standardization is essential for multi-feature inputs.

## References

- Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B., & Smola, A. (2012). [A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html). *JMLR*, 13:723–773.

- Muandet, K., Fukumizu, K., Sriperumbudur, B., & Schölkopf, B. (2017). [Kernel Mean Embedding of Distributions: A Review and Beyond](https://doi.org/10.1561/2200000060). *Foundations and Trends in Machine Learning*, 10(1–2):1–141.

- Harchaoui, Z. & Cappé, O. (2007). Retrospective Multiple Change-Point Estimation with Kernels. *IEEE Workshop on Statistical Signal Processing*, pp. 768–772.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Elijah Whitham-Powell

---

*This project was developed as part of STAT 671 (Statistical Learning I) at [Portland State University].*
