"""Streaming statistics for chunk-wise data analysis."""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class StreamingStats:
    """Streaming mean, variance, quantile and histogram summaries.

    Parameters
    ----------
    bins : int
        Number of bins used by :meth:`histogram`.
    hist_range : tuple[float, float], optional
        Fixed histogram range. If omitted, NumPy infers the range from data.
    window_size : int, optional
        If supplied, quantiles and histograms use only the latest ``window_size``
        rows. Mean and variance remain exact over all rows seen so far.

    Notes
    -----
    Mean and variance are updated with a mergeable Welford-style algorithm that
    ignores ``NaN`` values feature-wise and handles zero-count chunks safely.
    """

    def __init__(self, bins: int = 10, hist_range: tuple[float, float] | None = None, window_size: int | None = None):
        """Initialise empty streaming statistics accumulators."""

        self.bins = bins
        self.hist_range = hist_range
        self.window_size = window_size
        self.window = deque(maxlen=window_size) if window_size else None
        self.n = 0
        self.mean_ = None
        self.M2_ = None
        self.values_ = []

    def update_stats(self, X_chunk: Any):
        """Merge one chunk into the running statistics.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features) or (n_samples,)
            Numeric chunk. ``NaN`` entries are ignored in mean/variance updates.
        """

        X = np.asarray(X_chunk, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        mask = ~np.isnan(X)
        counts = mask.sum(axis=0).astype(float)
        safe = np.where(mask, X, 0.0)
        chunk_mean = np.divide(safe.sum(axis=0), counts, out=np.zeros(X.shape[1]), where=counts > 0)
        chunk_M2 = (np.where(mask, X - chunk_mean, 0.0) ** 2).sum(axis=0)
        if self.mean_ is None:
            self.mean_ = chunk_mean
            self.M2_ = chunk_M2
            self.n = counts
        else:
            total = self.n + counts
            delta = chunk_mean - self.mean_
            self.mean_ = self.mean_ + delta * counts / np.maximum(total, 1)
            self.M2_ = self.M2_ + chunk_M2 + (delta ** 2) * self.n * counts / np.maximum(total, 1)
            self.n = total
        if self.window is not None:
            for row in X:
                self.window.append(row)
        else:
            self.values_.append(X)
        return self

    @property
    def mean(self):
        """Return the running feature-wise mean or ``None`` before updates."""

        return self.mean_

    @property
    def variance(self):
        """Return population variance for all rows seen so far."""

        if self.M2_ is None:
            return None
        return np.divide(self.M2_, self.n, out=np.zeros_like(self.M2_), where=self.n > 0)

    def quantile(self, q: float | list[float] | np.ndarray) -> np.ndarray:
        """Return feature-wise quantiles from retained data."""

        data = self._data()
        if data.size == 0:
            return np.array([])
        return np.nanquantile(data, q, axis=0)

    def histogram(self, feature: int = 0):
        """Return ``np.histogram`` counts and bin edges for one feature."""

        data = self._data()
        if data.size == 0:
            return np.histogram(np.array([]), bins=self.bins, range=self.hist_range)
        data = data[:, feature]
        data = data[~np.isnan(data)]
        return np.histogram(data, bins=self.bins, range=self.hist_range)

    def _data(self) -> np.ndarray:
        """Return retained rows for quantile and histogram calculations."""

        if self.window is not None:
            if not self.window:
                return np.empty((0, 1))
            return np.asarray(list(self.window), dtype=float)
        if not self.values_:
            return np.empty((0, 1))
        return np.vstack(self.values_)
