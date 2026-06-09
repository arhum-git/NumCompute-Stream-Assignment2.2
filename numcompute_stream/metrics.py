"""Streaming classification metrics.

The classes in this module expose a common ``update`` / ``result`` / ``reset``
interface so they can be used directly by ``StreamTrainer`` or independently in
demos and tests.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class StreamingConfusionMatrix:
    """Incrementally accumulated confusion matrix with dynamic class expansion."""

    def __init__(self, classes: Any | None = None):
        """Create an empty confusion matrix with optional known classes."""

        self.classes = None if classes is None else list(classes)
        self.matrix = None if classes is None else np.zeros((len(self.classes), len(self.classes)), dtype=int)

    def _ensure(self, labels: np.ndarray) -> None:
        """Expand the matrix so every supplied label has a row and column."""

        labels = list(labels)
        if self.classes is None:
            self.classes = sorted(set(labels))
            self.matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        new = [x for x in labels if x not in self.classes]
        if new:
            self.classes.extend(sorted(new))
            old = self.matrix
            self.matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)
            self.matrix[: old.shape[0], : old.shape[1]] = old

    def update(self, y_true: Any, y_pred: Any):
        """Add a chunk of true/predicted labels."""

        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("y_true and y_pred must have the same length")
        self._ensure(np.concatenate([y_true, y_pred]))
        idx = {c: i for i, c in enumerate(self.classes)}
        true_idx = np.fromiter((idx[t] for t in y_true), dtype=int, count=y_true.size)
        pred_idx = np.fromiter((idx[p] for p in y_pred), dtype=int, count=y_pred.size)
        np.add.at(self.matrix, (true_idx, pred_idx), 1)
        return self

    def result(self) -> np.ndarray:
        """Return a copy of the current confusion matrix."""

        if self.matrix is None:
            return np.zeros((0, 0), dtype=int)
        return self.matrix.copy()

    def reset(self):
        """Clear counts while preserving known class labels."""

        self.matrix = None if self.classes is None else np.zeros((len(self.classes), len(self.classes)), dtype=int)
        return self


class StreamingMetric:
    """Streaming accuracy, precision, recall, or F1 metric.

    ``window_size`` enables rolling-window evaluation over the most recent samples.
    Macro averaging is used for precision, recall and F1.
    """

    def __init__(self, metric: str = "accuracy", classes: Any | None = None, average: str = "macro", window_size: int | None = None):
        """Initialise a streaming metric accumulator.

        Parameters
        ----------
        metric : {"accuracy", "precision", "recall", "f1"}
            Metric returned by :meth:`result`.
        classes : array-like, optional
            Initial class labels. New labels are added dynamically.
        average : str
            Averaging strategy for precision/recall/F1. Only ``"macro"`` is
            implemented to keep semantics clear.
        window_size : int, optional
            If supplied, compute the metric on the most recent samples only.
        """

        if metric not in {"accuracy", "precision", "recall", "f1"}:
            raise ValueError("metric must be accuracy, precision, recall, or f1")
        if average != "macro":
            raise ValueError("Only macro averaging is currently supported")
        self.metric = metric
        self.average = average
        self.window_size = window_size
        self.cm = StreamingConfusionMatrix(classes)
        self.window = deque(maxlen=window_size) if window_size else None

    def update(self, y_true: Any, y_pred: Any):
        """Update the metric with one chunk of labels and predictions."""

        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        if self.window is not None:
            self.window.extend(zip(y_true, y_pred))
            if self.window:
                yt, yp = zip(*self.window)
                self.cm = StreamingConfusionMatrix(self.cm.classes)
                self.cm.update(np.array(yt), np.array(yp))
        else:
            self.cm.update(y_true, y_pred)
        return self

    def result(self) -> float:
        """Return the current scalar metric value."""

        cm = self.cm.result()
        if cm.size == 0 or cm.sum() == 0:
            return 0.0
        if self.metric == "accuracy":
            return float(np.trace(cm) / cm.sum())
        tp = np.diag(cm).astype(float)
        precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
        recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
        return float(np.mean({"precision": precision, "recall": recall, "f1": f1}[self.metric]))

    def reset(self):
        """Reset accumulated labels while keeping configuration unchanged."""

        self.cm.reset()
        if self.window is not None:
            self.window.clear()
        return self


class StreamingAUC:
    """Binary ROC-AUC accumulator with tie-aware average ranks."""

    def __init__(self, positive_label: Any = 1, window_size: int | None = None):
        """Create a binary streaming ROC-AUC accumulator."""

        self.positive_label = positive_label
        self.window_size = window_size
        self.data = deque(maxlen=window_size) if window_size else []

    def update(self, y_true: Any, y_score: Any):
        """Add one chunk of binary labels and continuous scores."""

        y_true = np.asarray(y_true).reshape(-1)
        y_score = np.asarray(y_score, dtype=float).reshape(-1)
        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError("y_true and y_score must have the same length")
        self.data.extend((t, float(s)) for t, s in zip(y_true, y_score))
        return self

    @staticmethod
    def _average_ranks(scores: np.ndarray) -> np.ndarray:
        """Return tie-aware average ranks for 1D scores.

        The implementation sorts once, identifies equal-score groups with
        vectorised boundary detection, and loops only over unique tie groups.
        """

        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        boundaries = np.concatenate(([0], np.where(np.diff(sorted_scores) != 0)[0] + 1, [scores.size]))
        ranks_sorted = np.empty(scores.size, dtype=float)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            ranks_sorted[start:end] = (start + 1 + end) / 2.0
        ranks = np.empty(scores.size, dtype=float)
        ranks[order] = ranks_sorted
        return ranks

    def result(self) -> float:
        """Return cumulative or rolling binary ROC-AUC."""

        if len(self.data) == 0:
            return 0.0
        y, s = zip(*self.data)
        y = (np.asarray(y) == self.positive_label).astype(int)
        s = np.asarray(s, dtype=float)
        pos = int(y.sum())
        neg = int(y.size - pos)
        if pos == 0 or neg == 0:
            return 0.0
        ranks = self._average_ranks(s)
        return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))

    def reset(self):
        """Clear accumulated score-label pairs."""

        self.data.clear()
        return self


def accuracy_score(y_true: Any, y_pred: Any) -> float:
    """Functional accuracy helper."""

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    return float(np.mean(y_true == y_pred))
