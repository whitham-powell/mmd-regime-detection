# plots.py

import matplotlib.pyplot as plt


# TODO: needs typing
# TODO: these plot functions are basically identical, could probably be combined into one with an argument for which metric to plot
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
