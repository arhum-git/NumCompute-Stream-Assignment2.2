"""Lightweight matplotlib visualisations for streaming experiments."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib

# Use a non-interactive backend so tests and demos can save figures on machines
# without a working GUI/Tk installation (common on fresh Windows/Python setups).
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def _finish(save_path: str | None = None, show: bool = True):
    """Save and/or display the current figure, returning its axes."""

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return plt.gca()


def plot_metric_over_time(metric_values: Sequence[float], title: str = "Metric over time", ylabel: str = "Metric", save_path: str | None = None, show: bool = True):
    """Plot a scalar metric across streaming chunks.

    Parameters
    ----------
    metric_values : sequence of float
        One metric value per processed chunk.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save_path : str, optional
        If provided, save the figure to this path.
    show : bool
        Whether to call ``plt.show()`` after plotting.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.
    """

    x = np.arange(1, len(metric_values) + 1)
    plt.figure()
    plt.plot(x, metric_values, marker="o")
    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    return _finish(save_path, show)


def compare_models(metric1: Sequence[float], metric2: Sequence[float], labels: tuple[str, str] = ("Model 1", "Model 2"), save_path: str | None = None, show: bool = True):
    """Compare two streaming metric histories on a single plot."""

    x = np.arange(1, max(len(metric1), len(metric2)) + 1)
    plt.figure()
    plt.plot(x[:len(metric1)], metric1, marker="o", label=labels[0])
    plt.plot(x[:len(metric2)], metric2, marker="s", label=labels[1])
    plt.title("Streaming model comparison")
    plt.xlabel("Chunk")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return _finish(save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, save_path: str | None = None, show: bool = True):
    """Visualise latest-chunk predictions against ground-truth labels/values."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    x = np.arange(y_true.size)
    plt.figure()
    plt.scatter(x, y_true, label="Ground truth", alpha=0.7)
    plt.scatter(x, y_pred, label="Predictions", alpha=0.7)
    plt.title("Predictions vs ground truth")
    plt.xlabel("Sample")
    plt.ylabel("Class / value")
    plt.legend()
    return _finish(save_path, show)
