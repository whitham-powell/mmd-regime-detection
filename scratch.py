# scratch.py

# %%
import time
from kta import kta, kta_torch, alignment, alignment_torch, rbf, rbf_torch
import numpy as np
from src.mmd import sliding_window_mmd
from src.features import features_base
from src.plots import (
    plot_sliding_window_kta,
    plot_sliding_window_mmds,
    plot_sliding_window_pvals,
    plot_sliding_window_std_from_null,
)


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
    window=30,
    step=5,
    n_permutations=1000,
)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
# %%
plot_sliding_window_mmds(results)
plot_sliding_window_pvals(results)
plot_sliding_window_std_from_null(results)
plot_sliding_window_kta(results)
