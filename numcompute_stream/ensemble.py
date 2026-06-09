"""Streaming tree ensembles.

The ensemble implementation supports bagging and random-forest style feature
sub-sampling while keeping the public API aligned with the single tree estimator.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import column_means as _column_means, replace_nan as _replace_nan, validate_X_y as _validate_X_y
from .tree import DecisionTreeClassifier


class EnsembleClassifier:
    """Bagging or random-forest classifier with ``partial_fit`` support.

    Parameters mirror ``DecisionTreeClassifier`` where possible. Each stream
    update adds the new chunk to the cumulative buffer, samples bootstrap datasets
    for each tree, and refits the bounded-depth base learners.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        method: str = "random_forest",
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = None,
    ):
        if method not in {"bagging", "random_forest"}:
            raise ValueError("method must be 'bagging' or 'random_forest'")
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")

        self.n_estimators = int(n_estimators)
        self.method = method
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features if method == "random_forest" else None
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.trees_: list[DecisionTreeClassifier] = self._make_trees()
        self.X_seen_: np.ndarray | None = None
        self.y_seen_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.feature_means_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def _make_trees(self) -> list[DecisionTreeClassifier]:
        """Create independent base trees with deterministic derived seeds."""

        return [
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=None if self.random_state is None else self.random_state + i,
            )
            for i in range(self.n_estimators)
        ]

    def components(self) -> dict[str, Any]:
        """Describe the computational components used by this estimator."""

        return {
            "model": "EnsembleClassifier",
            "method": self.method,
            "base_estimator": "DecisionTreeClassifier",
            "n_estimators": self.n_estimators,
            "bootstrap": self.bootstrap,
            "feature_subsampling": self.max_features,
            "voting": "deterministic majority vote",
            "streaming_strategy": "cumulative chunks + per-tree refit",
        }

    def fit(self, X: Any, y: Any):
        """Fit the ensemble from a complete batch."""

        self.X_seen_ = None
        self.y_seen_ = None
        self.feature_means_ = None
        self.n_features_in_ = None
        self.trees_ = self._make_trees()
        return self.partial_fit(X, y)

    def partial_fit(self, X: Any, y: Any, classes: Any | None = None):
        """Update all trees with the next stream chunk."""

        X_arr, y_arr = _validate_X_y(X, y)
        if classes is not None:
            self.classes_ = np.asarray(classes)
        elif self.classes_ is None:
            self.classes_ = np.unique(y_arr)

        if self.X_seen_ is None:
            self.n_features_in_ = X_arr.shape[1]
            self.feature_means_ = _column_means(X_arr)
            self.X_seen_, _ = _replace_nan(X_arr, self.feature_means_)
            self.y_seen_ = y_arr.copy()
        else:
            if X_arr.shape[1] != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} features, got {X_arr.shape[1]}")
            raw = np.vstack([self.X_seen_, X_arr])
            self.feature_means_ = _column_means(raw)
            self.X_seen_, _ = _replace_nan(raw, self.feature_means_)
            self.y_seen_ = np.concatenate([self.y_seen_, y_arr])

        self.classes_ = np.unique(np.concatenate([np.asarray(self.classes_), np.unique(y_arr)]))
        n = self.X_seen_.shape[0]
        for tree in self.trees_:
            idx = self.rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            tree.fit(self.X_seen_[idx], self.y_seen_[idx])
        return self

    def _tree_predictions(self, X: Any) -> np.ndarray:
        """Return a matrix of per-tree predictions with shape ``(n_samples, n_trees)``."""

        if not self.trees_ or self.trees_[0].root_ is None:
            raise RuntimeError("EnsembleClassifier must be fitted before predict")
        return np.vstack([tree.predict(X) for tree in self.trees_]).T

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels using vectorised deterministic majority vote.

        Votes are counted for every known class across all trees at once. Ties are
        resolved by ``np.argmax`` over sorted ``classes_``, which selects the first
        class deterministically.
        """

        preds = self._tree_predictions(X)
        classes = np.asarray(self.classes_)
        votes = np.stack([(preds == cls).sum(axis=1) for cls in classes], axis=1)
        return classes[np.argmax(votes, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return vote proportions for each class."""

        preds = self._tree_predictions(X)
        classes = np.asarray(self.classes_)
        proba = np.zeros((preds.shape[0], classes.size), dtype=float)
        for i, cls in enumerate(classes):
            proba[:, i] = np.mean(preds == cls, axis=1)
        return proba

    def score(self, X: Any, y: Any) -> float:
        """Return mean classification accuracy for ``X`` and ``y``."""

        return float(np.mean(self.predict(X) == np.asarray(y).reshape(-1)))


class RandomForestClassifier(EnsembleClassifier):
    """Convenience random-forest classifier alias with explicit class name."""

    def __init__(self, *args, **kwargs):
        """Initialise a random forest by fixing ``method='random_forest'``."""

        kwargs.setdefault("method", "random_forest")
        super().__init__(*args, **kwargs)
