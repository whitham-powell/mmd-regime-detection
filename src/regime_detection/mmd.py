# mmd.py

from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np
from kta import kta
from numpy.typing import NDArray


def mmd_squared(
    X: NDArray,
    Y: NDArray,
    kernel_fn: Optional[Callable] = None,
    kernel_params: Optional[Mapping[str, Any]] = None,
    debug: bool = False,
) -> float:
    if kernel_fn is None:
        raise ValueError("A kernel function must be provided.")
    Kxx = kernel_fn(X, X, **(kernel_params or {}))
    Kyy = kernel_fn(Y, Y, **(kernel_params or {}))
    Kxy = kernel_fn(X, Y, **(kernel_params or {}))

    if debug:
        print(f"Kxx shape: {Kxx.shape}, Kyy shape: {Kyy.shape}, Kxy shape: {Kxy.shape}")
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    m, n = len(X), len(Y)

    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    return (
        Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2 * Kxy.sum() / (m * n)
    )


def mmd_permutation_test(
    X: NDArray,
    Y: NDArray,
    n_permutations: int = 1000,
    kernel_fn: Optional[Callable] = None,
    kernel_params: Optional[Mapping[str, Any]] = None,
) -> Tuple:
    observed = mmd_squared(X, Y, kernel_fn, kernel_params)
    pooled = np.vstack([X, Y])
    m = len(X)

    null_dist = []

    for _ in range(n_permutations):
        perm = np.random.permutation(len(pooled))
        null_dist.append(
            mmd_squared(pooled[perm[:m]], pooled[perm[m:]], kernel_fn, kernel_params),
        )

    null_dist = np.array(null_dist)
    p_value = (null_dist >= observed).mean()
    return observed, p_value, null_dist.mean(), null_dist.std()


def sliding_window_mmd(
    data,
    window: int = 60,
    step: int = 5,
    n_permutations: int = 1000,
    kernel_fn: Optional[Callable] = None,
    kernel_params: Optional[Mapping[str, Any]] = None,
):
    data = np.array(data)
    n = len(data)

    results = []

    for t in range(window, n - window, step):
        before = data[t - window : t]
        after = data[t : t + window]

        mmd, p_val, null_mean, null_std = mmd_permutation_test(
            before,
            after,
            n_permutations,
            kernel_fn,
            kernel_params,
        )
        std_from_null = (mmd - null_mean) / null_std if null_std > 0 else 0.0

        kta_value = compute_kta(before, after, kernel_fn, kernel_params)
        results.append(
            {
                "t": t,
                "mmd": mmd,
                "p_val": p_val,
                "std_from_null": std_from_null,
                "kta_val": kta_value,
            },
        )

    return results


def compute_kta(X, Y, kernel_fn, kernel_params, debug=False):
    params = kernel_params or {}

    Z = np.concatenate([X, Y])
    K = kernel_fn(Z, Z, **params)

    n_x = X.shape[0]
    n_y = Y.shape[0]

    y = np.empty(n_x + n_y, dtype=np.float32)
    y[:n_x] = -1
    y[n_x:] = 1
    if debug:
        print(
            f"n_x={n_x}, n_y={n_y}, n_x+n_y={n_x+n_y}, K.shape={K.shape}, y.shape={y.shape}, Z.shape={Z.shape}",
        )
    return kta(K, y)
