"""Internal shared helpers for NumCompute-Stream.

The public package modules import these helpers to keep validation and NaN
handling consistent across trees, ensembles, and future streaming estimators.
They are deliberately small, NumPy-only, and private to the package.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def validate_X_y(X: Any, y: Any | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate feature and target arrays for supervised estimators.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix. Values are converted to ``float`` so missing
        numerical entries can be represented as ``NaN``.
    y : array-like of shape (n_samples,), optional
        Target labels. If supplied, it is flattened to one dimension.

    Returns
    -------
    tuple
        ``(X_arr, y_arr)`` where ``X_arr`` is 2D and ``y_arr`` is either 1D or
        ``None``.

    Raises
    ------
    ValueError
        If ``X`` is not 2D or if ``X`` and ``y`` have inconsistent lengths.
    """

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X_arr.shape}")
    y_arr = None
    if y is not None:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = y_arr.reshape(-1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must contain the same number of rows: "
                f"{X_arr.shape[0]} != {y_arr.shape[0]}"
            )
    return X_arr, y_arr


def column_means(X: np.ndarray) -> np.ndarray:
    """Return NaN-safe column means, replacing all-NaN columns with zero."""

    with np.errstate(invalid="ignore"):
        means = np.nanmean(X, axis=0)
    return np.where(np.isnan(means), 0.0, means)


def replace_nan(X: np.ndarray, fill_values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Replace ``NaN`` values with supplied or computed column means.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix that may contain ``NaN`` values.
    fill_values : array-like of shape (n_features,), optional
        Values used to fill missing entries. If omitted, values are estimated
        from ``X`` using :func:`column_means`.

    Returns
    -------
    tuple
        Cleaned array and the fill values used.
    """

    X_arr = np.asarray(X, dtype=float)
    means = column_means(X_arr) if fill_values is None else np.asarray(fill_values, dtype=float)
    return np.where(np.isnan(X_arr), means, X_arr), means
