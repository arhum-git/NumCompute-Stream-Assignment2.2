"""Small NumPy CSV utilities for streaming demos."""
from __future__ import annotations

import csv
from typing import Any, Iterator

import numpy as np


def read_csv_numpy(path: str, skip_header: bool = True, delimiter: str = ",", dtype: Any = float) -> np.ndarray:
    """Read a CSV file into a NumPy array, treating empty cells as ``NaN``.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    skip_header : bool
        If True, skip the first row.
    delimiter : str
        Field separator.
    dtype : dtype
        NumPy dtype for the output array.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_rows, n_columns)``.
    """

    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if skip_header:
            next(reader, None)
        for row in reader:
            rows.append([np.nan if cell == "" else cell for cell in row])
    return np.asarray(rows, dtype=dtype)


def chunk_array(X: np.ndarray, y: np.ndarray | None = None, chunk_size: int = 32) -> Iterator[np.ndarray | tuple[np.ndarray, np.ndarray]]:
    """Yield consecutive chunks from ``X`` and optional aligned targets ``y``.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, ...)
        Data to split into stream chunks.
    y : np.ndarray, optional
        Target array with the same first dimension as ``X``.
    chunk_size : int
        Maximum number of rows per yielded chunk.

    Yields
    ------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Feature chunk alone, or ``(X_chunk, y_chunk)`` when targets are supplied.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    for start in range(0, X.shape[0], chunk_size):
        end = start + chunk_size
        if y is None:
            yield X[start:end]
        else:
            yield X[start:end], y[start:end]
