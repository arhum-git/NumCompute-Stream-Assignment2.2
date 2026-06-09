"""High-level trainer for streaming chunks."""
from __future__ import annotations

import sys
from typing import Any

import numpy as np

from .metrics import StreamingMetric


class StreamTrainer:
    """Wrap a streaming model and log per-chunk training metrics.

    Parameters
    ----------
    model : object
        Estimator implementing ``partial_fit`` or ``fit`` plus ``predict``.
    metric : StreamingMetric, optional
        Cumulative metric accumulator. Defaults to streaming accuracy.

    Attributes
    ----------
    logs : dict[str, list]
        Per-chunk ``chunk``, ``chunk_accuracy``, ``cumulative_accuracy`` and
        ``memory_bytes`` histories for plotting and reporting.
    chunk_id : int
        Number of chunks processed through :meth:`fit_chunk`.
    """

    def __init__(self, model: Any, metric: StreamingMetric | None = None):
        """Create a trainer for a model and optional streaming metric."""

        self.model = model
        self.metric = metric if metric is not None else StreamingMetric("accuracy")
        self.logs = {"chunk": [], "chunk_accuracy": [], "cumulative_accuracy": [], "memory_bytes": []}
        self.chunk_id = 0

    def fit_chunk(self, X: Any, y: Any) -> float:
        """Fit or update the wrapped model on one chunk and append logs.

        Returns
        -------
        float
            Accuracy of the updated model on the same chunk.
        """

        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y)
        else:
            self.model.fit(X, y)
        preds = self.model.predict(X)
        y_arr = np.asarray(y).reshape(-1)
        chunk_acc = float(np.mean(preds == y_arr))
        self.metric.update(y_arr, preds)
        self.chunk_id += 1
        self.logs["chunk"].append(self.chunk_id)
        self.logs["chunk_accuracy"].append(chunk_acc)
        self.logs["cumulative_accuracy"].append(self.metric.result())
        self.logs["memory_bytes"].append(self._memory_footprint())
        return chunk_acc

    def score_chunk(self, X: Any, y: Any) -> float:
        """Score the current model on a chunk without updating model or logs."""

        preds = self.model.predict(X)
        return float(np.mean(preds == np.asarray(y).reshape(-1)))

    def reset(self):
        """Reset trainer logs and metric accumulator without replacing the model."""

        self.chunk_id = 0
        self.logs = {"chunk": [], "chunk_accuracy": [], "cumulative_accuracy": [], "memory_bytes": []}
        self.metric.reset()
        return self

    def _memory_footprint(self) -> int:
        """Estimate model memory in bytes from object and buffered arrays."""

        total = sys.getsizeof(self.model)
        for attr in ("X_seen_", "y_seen_"):
            obj = getattr(self.model, attr, None)
            if hasattr(obj, "nbytes"):
                total += obj.nbytes
        return int(total)
