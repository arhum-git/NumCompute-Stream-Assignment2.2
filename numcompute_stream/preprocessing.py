"""Streaming preprocessing utilities for NumCompute-Stream."""
from __future__ import annotations

from typing import Any

import numpy as np


class StandardScaler:
    """Incremental standard scaler using mergeable running mean/variance.

    Parameters
    ----------
    with_mean : bool
        If True, subtract the running feature means during transformation.
    with_std : bool
        If True, divide by the running feature standard deviations.
    eps : float
        Minimum variance floor used to avoid division by zero.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-12):
        """Create an unfitted streaming scaler."""

        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps
        self.n_seen_ = 0
        self.mean_ = None
        self.var_ = None
        self.M2_ = None
        self.scale_ = None

    def partial_fit(self, X: Any, y: Any | None = None):
        """Update running mean and variance from one numeric chunk.

        ``NaN`` values are ignored when estimating statistics. Columns with zero
        observed variance receive a safe scale of ``sqrt(eps)``.
        """

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        mask = ~np.isnan(X)
        counts = mask.sum(axis=0).astype(float)
        safe = np.where(mask, X, 0.0)
        chunk_sum = safe.sum(axis=0)
        chunk_mean = np.divide(chunk_sum, counts, out=np.zeros(X.shape[1]), where=counts > 0)
        centered = np.where(mask, X - chunk_mean, 0.0)
        chunk_M2 = (centered ** 2).sum(axis=0)
        if self.mean_ is None:
            self.mean_ = chunk_mean
            self.M2_ = chunk_M2
            self.n_seen_ = counts
        else:
            if X.shape[1] != self.mean_.shape[0]:
                raise ValueError(f"Expected {self.mean_.shape[0]} features, got {X.shape[1]}")
            total = self.n_seen_ + counts
            delta = chunk_mean - self.mean_
            self.mean_ = np.where(total > 0, self.mean_ + delta * counts / np.maximum(total, 1), self.mean_)
            self.M2_ = self.M2_ + chunk_M2 + (delta ** 2) * self.n_seen_ * counts / np.maximum(total, 1)
            self.n_seen_ = total
        self.var_ = np.divide(self.M2_, self.n_seen_, out=np.zeros_like(self.M2_), where=self.n_seen_ > 0)
        self.scale_ = np.sqrt(np.maximum(self.var_, self.eps))
        return self

    def fit(self, X: Any, y: Any | None = None):
        """Reset and estimate statistics from a complete batch."""

        self.n_seen_ = 0
        self.mean_ = self.var_ = self.M2_ = self.scale_ = None
        return self.partial_fit(X, y)

    def transform(self, X: Any) -> np.ndarray:
        """Scale a numeric array using the current running statistics."""

        if self.mean_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"Expected {self.mean_.shape[0]} features, got {X.shape[1]}")
        X = np.where(np.isnan(X), self.mean_, X)
        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
        return X

    def fit_transform(self, X: Any, y: Any | None = None) -> np.ndarray:
        """Fit the scaler and return the transformed batch."""

        return self.fit(X, y).transform(X)


class Imputer:
    """Streaming missing-value imputer.

    Parameters
    ----------
    strategy : {"mean", "median", "constant"}
        Statistic used to fill missing values. Median stores observed values for
        exact coursework-scale streaming medians.
    fill_value : float
        Constant fill value used when ``strategy='constant'``.
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0):
        """Create an unfitted streaming imputer."""

        if strategy not in {"mean", "median", "constant"}:
            raise ValueError("strategy must be 'mean', 'median', or 'constant'")
        self.strategy = strategy
        self.fill_value = fill_value
        self.n_seen_ = None
        self.sum_ = None
        self.values_ = None
        self.statistics_ = None

    def partial_fit(self, X: Any, y: Any | None = None):
        """Update missing-value statistics from one chunk."""

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if self.strategy == "constant":
            self.statistics_ = np.full(X.shape[1], self.fill_value, dtype=float)
            return self
        mask = ~np.isnan(X)
        if self.n_seen_ is None:
            self.n_seen_ = np.zeros(X.shape[1], dtype=float)
            self.sum_ = np.zeros(X.shape[1], dtype=float)
            self.values_ = [[] for _ in range(X.shape[1])]
        elif X.shape[1] != self.n_seen_.shape[0]:
            raise ValueError(f"Expected {self.n_seen_.shape[0]} features, got {X.shape[1]}")
        self.n_seen_ += mask.sum(axis=0)
        self.sum_ += np.where(mask, X, 0.0).sum(axis=0)
        if self.strategy == "mean":
            self.statistics_ = np.divide(self.sum_, self.n_seen_, out=np.zeros_like(self.sum_), where=self.n_seen_ > 0)
        else:
            for j in range(X.shape[1]):
                vals = X[mask[:, j], j]
                if vals.size:
                    self.values_[j].extend(vals.tolist())
            self.statistics_ = np.array([np.median(v) if len(v) else 0.0 for v in self.values_])
        return self

    def fit(self, X: Any, y: Any | None = None):
        """Reset and estimate imputation statistics from a complete batch."""

        self.n_seen_ = self.sum_ = self.values_ = self.statistics_ = None
        return self.partial_fit(X, y)

    def transform(self, X: Any) -> np.ndarray:
        """Replace ``NaN`` entries using the current imputation statistics."""

        if self.statistics_ is None:
            raise RuntimeError("Imputer must be fitted before transform")
        X = np.asarray(X, dtype=float).copy()
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != self.statistics_.shape[0]:
            raise ValueError(f"Expected {self.statistics_.shape[0]} features, got {X.shape[1]}")
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.statistics_, inds[1])
        return X

    def fit_transform(self, X: Any, y: Any | None = None) -> np.ndarray:
        """Fit the imputer and return the transformed batch."""

        return self.fit(X, y).transform(X)


class OneHotEncoder:
    """Incremental one-hot encoder that expands columns as categories arrive.

    Parameters
    ----------
    handle_unknown : {"ignore", "error"}
        Behaviour when :meth:`transform` sees a category not observed during
        fitting. ``"ignore"`` leaves all zeros for that feature block.
    """

    def __init__(self, handle_unknown: str = "ignore"):
        """Create an unfitted one-hot encoder."""

        if handle_unknown not in {"ignore", "error"}:
            raise ValueError("handle_unknown must be 'ignore' or 'error'")
        self.handle_unknown = handle_unknown
        self.categories_ = None

    def partial_fit(self, X: Any, y: Any | None = None):
        """Update known categories from one chunk."""

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.categories_ is None:
            self.categories_ = [list() for _ in range(X.shape[1])]
        if X.shape[1] != len(self.categories_):
            raise ValueError("X has different number of columns than fitted encoder")
        for j in range(X.shape[1]):
            for val in np.unique(X[:, j]):
                if val not in self.categories_[j]:
                    self.categories_[j].append(val)
            self.categories_[j] = sorted(self.categories_[j], key=lambda z: str(z))
        return self

    def fit(self, X: Any, y: Any | None = None):
        """Reset categories and learn them from a complete batch."""

        self.categories_ = None
        return self.partial_fit(X, y)

    def transform(self, X: Any) -> np.ndarray:
        """Return one-hot encoded columns for the current category mapping."""

        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before transform")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(self.categories_):
            raise ValueError("X has different number of columns than fitted encoder")
        cols = []
        for j, cats in enumerate(self.categories_):
            out = np.zeros((X.shape[0], len(cats)), dtype=float)
            mapping = {c: k for k, c in enumerate(cats)}
            for i, val in enumerate(X[:, j]):
                if val in mapping:
                    out[i, mapping[val]] = 1.0
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category {val!r} in column {j}")
            cols.append(out)
        return np.hstack(cols) if cols else np.empty((X.shape[0], 0))

    def fit_transform(self, X: Any, y: Any | None = None) -> np.ndarray:
        """Fit the encoder and return the encoded batch."""

        return self.fit(X, y).transform(X)
