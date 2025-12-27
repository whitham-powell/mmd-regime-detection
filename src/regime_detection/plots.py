# plots.py

from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def results_to_dataframe(
    results: List[Dict],
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Convert sliding window MMD results to a DataFrame with actual dates.

    Parameters
    ----------
    results : list of dicts
        Output from sliding_window_mmd, each dict has keys:
        't', 'mmd', 'p_val', 'std_from_null', 'kta_val'
    date_index : pd.DatetimeIndex
        Index from the feature DataFrame used in the analysis
        (e.g., features_base.index)

    Returns
    -------
    pd.DataFrame
        Results with 'date' as index, columns for each metric
    """
    df_results = pd.DataFrame(results)
    df_results["date"] = date_index[df_results["t"]]
    df_results = df_results.set_index("date")
    return df_results


def find_regime_boundaries(
    results_df: pd.DataFrame,
    metric: str = "std_from_null",
    threshold: float = 10.0,
    min_gap_days: int = 20,
) -> pd.DatetimeIndex:
    """
    Extract discrete regime boundary dates from results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from results_to_dataframe
    metric : str
        Column to threshold ('std_from_null', 'mmd', 'kta_val')
    threshold : float
        Values above this are flagged as boundaries
    min_gap_days : int
        Minimum days between reported boundaries (merges nearby detections)

    Returns
    -------
    pd.DatetimeIndex
        Dates identified as regime boundaries
    """
    # Find all points exceeding threshold
    significant = results_df[results_df[metric] > threshold].index

    if len(significant) == 0:
        return pd.DatetimeIndex([])

    # Merge nearby detections
    boundaries = [significant[0]]
    for date in significant[1:]:
        if (date - boundaries[-1]).days > min_gap_days:
            boundaries.append(date)

    return pd.DatetimeIndex(boundaries)


def plot_regime_detection_panel(
    price_series: pd.Series,
    results_df: pd.DataFrame,
    ticker: str = "SPY",
    metric: str = "std_from_null",
    threshold: float = 10.0,
    min_gap_days: int = 20,
    figsize: Tuple[int, int] = (14, 10),
    title: str = "Market Regime Detection via MMD",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a 4-panel figure for regime detection validation.

    Parameters
    ----------
    price_series : pd.Series
        Price series with DatetimeIndex (e.g., df['Close'])
    results_df : pd.DataFrame
        Output from results_to_dataframe
    metric : str
        Which metric to use for boundary detection
    threshold : float
        Threshold for boundary detection
    min_gap_days : int
        Minimum days between boundaries
    figsize : tuple
        Figure size
    title : str
        Main figure title
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig, axes : tuple
        Matplotlib figure and axes array
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Align price to results date range
    start_date = results_df.index[0]
    end_date = results_df.index[-1]
    price_aligned = price_series.loc[start_date:end_date]

    # Find boundary dates
    boundaries = find_regime_boundaries(
        results_df,
        metric=metric,
        threshold=threshold,
        min_gap_days=min_gap_days,
    )

    # Panel 1: Price with regime boundaries
    axes[0].plot(price_aligned.index, price_aligned.values, "k-", lw=1, label=ticker)
    for b in boundaries:
        axes[0].axvline(b, color="red", alpha=0.7, lw=1.5, ls="--")
    axes[0].set_ylabel("Price", fontsize=10)
    axes[0].set_title(title, fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(0, 1.0),
        borderaxespad=0,
        fontsize=9,
    )

    # Panel 2: MMD²
    axes[1].plot(results_df.index, results_df["mmd"], "b-", lw=1)
    for b in boundaries:
        axes[1].axvline(b, color="red", alpha=0.3, lw=1)
    axes[1].set_ylabel("MMD²", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Std from Null (or selected metric)
    axes[2].plot(results_df.index, results_df["std_from_null"], "g-", lw=1)
    axes[2].axhline(
        threshold,
        color="red",
        ls="--",
        lw=1.5,
        label=f"threshold={threshold}",
    )
    for b in boundaries:
        axes[2].axvline(b, color="red", alpha=0.3, lw=1)
    axes[2].set_ylabel("Std from Null", fontsize=10)
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: KTA
    axes[3].plot(results_df.index, results_df["kta_val"], color="purple", lw=1)
    for b in boundaries:
        axes[3].axvline(b, color="red", alpha=0.3, lw=1)
    axes[3].set_ylabel("KTA", fontsize=10)
    axes[3].set_xlabel("Date", fontsize=10)
    axes[3].grid(True, alpha=0.3)

    # Format x-axis with dates
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add boundary count annotation
    n_boundaries = len(boundaries)
    fig.text(
        0.99,
        0.01,
        f"Detected boundaries: {n_boundaries}",
        ha="right",
        va="bottom",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig, axes


def plot_regime_boundaries_summary(
    price_series: pd.Series,
    results_df: pd.DataFrame,
    boundaries: pd.DatetimeIndex,
    ticker: str = "SPY",
    window_days: int = 30,
    figsize: Tuple[int, int] = (12, 5),
    annotate: bool = True,
    max_annotations: int = 10,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Simple single-panel plot: price with detected boundaries.

    Parameters
    ----------
    price_series : pd.Series
        Price series with DatetimeIndex
    results_df : pd.DataFrame
        Output from results_to_dataframe
    boundaries : pd.DatetimeIndex
        Output from find_regime_boundaries
    window_days : int
        Window size used (for annotation)
    figsize : tuple
        Figure size
    annotate : bool
        Whether to add date annotations on the plot
    max_annotations : int
        Maximum number of boundaries to annotate (prevents clutter)

    Returns
    -------
    fig, ax : tuple
    """
    fig, ax = plt.subplots(figsize=figsize)

    start_date = results_df.index[0]
    end_date = results_df.index[-1]
    price_aligned = price_series.loc[start_date:end_date]

    ax.plot(price_aligned.index, price_aligned.values, "k-", lw=1, label=ticker)

    # Draw boundary lines
    for b in boundaries:
        ax.axvline(b, color="red", alpha=0.7, lw=1.5, ls="--")

    # Annotate boundaries (limited to avoid clutter)
    if annotate and len(boundaries) > 0:
        # If too many boundaries, only annotate a subset
        if len(boundaries) > max_annotations:
            # Annotate first, last, and evenly spaced in between
            indices = np.linspace(0, len(boundaries) - 1, max_annotations, dtype=int)
            boundaries_to_annotate = boundaries[indices]
        else:
            boundaries_to_annotate = boundaries

        # Alternate y-positions to reduce overlap
        y_min, y_max = price_aligned.min(), price_aligned.max()
        y_range = y_max - y_min
        y_positions = [
            y_max - 0.05 * y_range,  # top
            y_max - 0.15 * y_range,  # upper
            y_max - 0.25 * y_range,  # middle-upper
            y_min + 0.25 * y_range,  # middle-lower
            y_min + 0.15 * y_range,  # lower
            y_min + 0.05 * y_range,  # bottom
        ]

        for i, b in enumerate(boundaries_to_annotate):
            y_pos = y_positions[i % len(y_positions)]
            ax.annotate(
                b.strftime("%Y-%m-%d"),
                xy=(b, y_pos),
                fontsize=8,
                color="red",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.set_title(f"Detected Regime Boundaries (window={window_days} days)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.1), borderaxespad=0)

    plt.tight_layout()
    return fig, ax


# --- Original plotting functions (kept for backwards compatibility) ---


def plot_sliding_window_mmds(
    results,
    title="Sliding Window MMD Analysis",
    subtitle="MMDs vs Time",
    save_fig=False,
):
    mmds = [r["mmd"] for r in results]
    times = [r["t"] for r in results]
    full_title = title + ("\n" + subtitle if subtitle else "")
    plt.figure(figsize=(12, 6))
    plt.plot(times, mmds, label="MMD")
    plt.xlabel("Time")
    plt.ylabel("MMD")
    plt.title(full_title)
    plt.legend()
    plt.show()
    if save_fig:
        plt.savefig(f"{title.replace(' ', '-').lower()}.png")


def plot_sliding_window_pvals(
    results,
    title="Sliding Window MMD Analysis",
    subtitle="P-values vs Time",
    save_fig=False,
):
    p_values = [r["p_val"] for r in results]
    times = [r["t"] for r in results]
    full_title = title + ("\n" + subtitle if subtitle else "")
    plt.figure(figsize=(12, 6))
    plt.plot(times, p_values, label="p-value")
    plt.axhline(y=0.05, color="r", linestyle="--", label="0.05 Significance Level")
    plt.xlabel("Time")
    plt.ylabel("p-value")
    plt.title(full_title)
    plt.legend()
    plt.show()
    if save_fig:
        plt.savefig(f"{title.replace(' ', '-').lower()}.png")


def plot_sliding_window_std_from_null(
    results,
    title="Sliding Window MMD Analysis",
    subtitle="Standard Deviations from Null vs Time",
    save_fig=False,
):
    std_from_null = [r["std_from_null"] for r in results]
    times = [r["t"] for r in results]
    full_title = title + ("\n" + subtitle if subtitle else "")
    plt.figure(figsize=(12, 6))
    plt.plot(times, std_from_null, label="Std from Null")
    plt.xlabel("Time")
    plt.ylabel("Standard Deviations from Null")
    plt.title(full_title)
    plt.legend()
    plt.show()
    if save_fig:
        plt.savefig(f"{title.replace(' ', '-').lower()}.png")


def plot_sliding_window_kta(
    results,
    title="Sliding Window MMD Analysis",
    subtitle="KTA Values vs Time",
    save_fig=False,
):
    kta_values = [r["kta_val"] for r in results]
    times = [r["t"] for r in results]
    full_title = title + ("\n" + subtitle if subtitle else "")
    plt.figure(figsize=(12, 6))
    plt.plot(times, kta_values, label="KTA Value")
    plt.xlabel("Time")
    plt.ylabel("KTA Value")
    plt.title(full_title)
    plt.legend()
    plt.show()
    if save_fig:
        plt.savefig(f"{title.replace(' ', '-').lower()}.png")
