"""Streaming-compatible decision tree classifier.

This module intentionally uses only NumPy and plain Python data structures.  The
``partial_fit`` method keeps a cumulative stream buffer and regrows a bounded-depth
tree after each chunk.  That design is deterministic, easy to test, and suitable
for coursework-scale online-learning simulations where the emphasis is on
algorithmic clarity rather than production-scale Hoeffding trees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._utils import column_means as _column_means, replace_nan as _replace_nan, validate_X_y as _validate_X_y


@dataclass
class Split:
    """Best split found for a tree node."""

    feature: int | None
    threshold: float | None
    gain: float


class DecisionTreeClassifier:
    """Depth-limited decision tree classifier with a streaming ``partial_fit`` API.

    Parameters
    ----------
    max_depth:
        Maximum tree depth.  ``0`` creates a single majority-class leaf.
    min_samples_split:
        Minimum samples required before attempting a split.
    criterion:
        Either ``"gini"`` or ``"entropy"``.
    max_features:
        Number of candidate features at each split. Supports ``None``, integers,
        fractions in ``(0, 1]``, ``"sqrt"`` and ``"log2"``.
    random_state:
        Seed for deterministic feature sub-sampling.

    Notes
    -----
    ``partial_fit`` accumulates observed chunks and rebuilds the tree.  This gives
    stable online behaviour while preserving exact greedy tree learning over all
    data seen so far. Missing values are imputed using training-stream means and
    the same means are reused at prediction time.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ):
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")

        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.criterion = criterion
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)

        self.root_: dict[str, Any] | None = None
        self.classes_: np.ndarray | None = None
        self.X_seen_: np.ndarray | None = None
        self.y_seen_: np.ndarray | None = None
        self.feature_means_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def components(self) -> dict[str, Any]:
        """Describe the computational components used by this estimator."""

        return {
            "model": "DecisionTreeClassifier",
            "criterion": self.criterion,
            "split_search": "greedy threshold search over unique feature midpoints",
            "streaming_strategy": "cumulative chunks + bounded-depth regrowth",
            "nan_strategy": "training-stream column mean imputation",
            "tie_strategy": "deterministic smallest sorted class label",
        }

    def _impurity(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        if self.criterion == "gini":
            return float(1.0 - np.sum(p**2))
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def _majority(y: np.ndarray) -> Any:
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def _feature_indices(self, n_features: int) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_features)
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                k = max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                k = max(1, int(np.log2(n_features)))
            else:
                raise ValueError("max_features string must be 'sqrt' or 'log2'")
        elif isinstance(self.max_features, float):
            if not (0 < self.max_features <= 1):
                raise ValueError("float max_features must be in (0, 1]")
            k = max(1, int(np.ceil(self.max_features * n_features)))
        else:
            k = int(self.max_features)
        k = min(max(1, k), n_features)
        return self.rng.choice(n_features, size=k, replace=False)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Split:
        n, m = X.shape
        parent = self._impurity(y)
        best = Split(feature=None, threshold=None, gain=0.0)

        for j in self._feature_indices(m):
            order = np.argsort(X[:, j], kind="mergesort")
            xj = X[order, j]
            yj = y[order]
            unique_values = np.unique(xj)
            if unique_values.size <= 1:
                continue
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                left_count = np.searchsorted(xj, threshold, side="right")
                if left_count == 0 or left_count == n:
                    continue
                y_left = yj[:left_count]
                y_right = yj[left_count:]
                gain = parent - (left_count / n) * self._impurity(y_left) - ((n - left_count) / n) * self._impurity(y_right)
                if gain > best.gain + 1e-15:
                    best = Split(feature=int(j), threshold=float(threshold), gain=float(gain))
        return best

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict[str, Any]:
        node: dict[str, Any] = {"prediction": self._majority(y), "n_samples": int(y.size), "depth": int(depth)}
        if depth >= self.max_depth or y.size < self.min_samples_split or np.unique(y).size == 1:
            node["leaf"] = True
            return node

        split = self._best_split(X, y)
        if split.feature is None or split.threshold is None or split.gain <= 0:
            node["leaf"] = True
            return node

        left_mask = X[:, split.feature] <= split.threshold
        node.update({"leaf": False, "feature": split.feature, "threshold": split.threshold, "gain": split.gain})
        node["left"] = self._build(X[left_mask], y[left_mask], depth + 1)
        node["right"] = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X: Any, y: Any):
        """Fit the tree from a complete batch."""

        X_arr, y_arr = _validate_X_y(X, y)
        X_clean, means = _replace_nan(X_arr)
        self.n_features_in_ = X_clean.shape[1]
        self.feature_means_ = means
        self.classes_ = np.unique(y_arr)
        self.X_seen_ = X_clean.copy()
        self.y_seen_ = y_arr.copy()
        self.root_ = self._build(X_clean, y_arr, 0)
        return self

    def partial_fit(self, X: Any, y: Any, classes: Any | None = None):
        """Update the tree with the next stream chunk."""

        X_arr, y_arr = _validate_X_y(X, y)
        if classes is not None:
            self.classes_ = np.asarray(classes)
        if self.X_seen_ is None:
            return self.fit(X_arr, y_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_arr.shape[1]}")

        combined_raw = np.vstack([self.X_seen_, np.asarray(X_arr, dtype=float)])
        self.feature_means_ = _column_means(combined_raw)
        combined_clean, _ = _replace_nan(combined_raw, self.feature_means_)
        self.X_seen_ = combined_clean
        self.y_seen_ = np.concatenate([self.y_seen_, y_arr])
        if self.classes_ is None:
            self.classes_ = np.unique(self.y_seen_)
        else:
            self.classes_ = np.unique(np.concatenate([self.classes_, np.unique(y_arr)]))
        self.root_ = self._build(self.X_seen_, self.y_seen_, 0)
        return self

    def _prepare_predict_X(self, X: Any) -> np.ndarray:
        X_arr, _ = _validate_X_y(X)
        if self.root_ is None or self.feature_means_ is None:
            raise RuntimeError("DecisionTreeClassifier must be fitted before predict")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_arr.shape[1]}")
        X_clean, _ = _replace_nan(X_arr, self.feature_means_)
        return X_clean

    def _predict_one(self, x: np.ndarray) -> Any:
        node = self.root_
        while not node.get("leaf", False):
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["prediction"]

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for ``X``.

        Tree traversal is inherently branch-dependent, so each sample follows its
        own path from root to leaf in ``O(max_depth)`` time. The split search and
        preprocessing logic are vectorised; the compact per-sample traversal keeps
        the node representation readable for coursework-scale streaming demos.
        """

        X_clean = self._prepare_predict_X(X)
        return np.fromiter((self._predict_one(row) for row in X_clean), dtype=self.classes_.dtype, count=X_clean.shape[0])

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return one-hot probabilities based on hard tree predictions."""

        pred = self.predict(X)
        classes = np.asarray(self.classes_)
        out = np.zeros((pred.size, classes.size), dtype=float)
        for i, cls in enumerate(classes):
            out[:, i] = pred == cls
        return out

    def score(self, X: Any, y: Any) -> float:
        """Return mean classification accuracy."""

        return float(np.mean(self.predict(X) == np.asarray(y).reshape(-1)))
