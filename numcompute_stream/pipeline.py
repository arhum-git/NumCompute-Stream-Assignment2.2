"""Composable streaming pipeline API."""
from __future__ import annotations

from typing import Any

import numpy as np


class Pipeline:
    """Chain transformers and a final estimator with consistent streaming APIs.

    Each step is a ``(name, object)`` pair. Intermediate steps may implement
    ``partial_fit``/``fit`` and must implement ``transform``. The final step must
    implement ``partial_fit`` or ``fit`` and ``predict``.
    """

    def __init__(self, steps: list[tuple[str, Any]]):
        if not steps:
            raise ValueError("Pipeline requires at least one step")
        names = [name for name, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError("Pipeline step names must be unique")
        self.steps = list(steps)
        self.named_steps = dict(steps)

    def components(self) -> dict[str, Any]:
        """Return a compact description of the pipeline components."""

        return {
            name: step.components() if hasattr(step, "components") else step.__class__.__name__
            for name, step in self.steps
        }

    def _transform_until_model(self, X: Any, y: Any | None = None, fit: bool = False):
        Xt = X
        for name, step in self.steps[:-1]:
            if fit:
                if hasattr(step, "partial_fit"):
                    step.partial_fit(Xt, y)
                elif hasattr(step, "fit"):
                    step.fit(Xt, y)
            if not hasattr(step, "transform"):
                raise TypeError(f"Intermediate pipeline step {name!r} must implement transform")
            Xt = step.transform(Xt)
        return Xt

    def partial_fit(self, X: Any, y: Any, classes: Any | None = None):
        """Incrementally fit transformers and final estimator on one chunk."""

        Xt = self._transform_until_model(X, y, fit=True)
        model = self.steps[-1][1]
        if hasattr(model, "partial_fit"):
            try:
                model.partial_fit(Xt, y, classes=classes)
            except TypeError:
                model.partial_fit(Xt, y)
        elif hasattr(model, "fit"):
            model.fit(Xt, y)
        else:
            raise TypeError("Final pipeline step must implement partial_fit or fit")
        return self

    def fit(self, X: Any, y: Any):
        """Fit the pipeline using the same behaviour as a single streaming chunk."""

        return self.partial_fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        """Transform input data and return final estimator predictions."""

        Xt = self._transform_until_model(X, fit=False)
        model = self.steps[-1][1]
        if not hasattr(model, "predict"):
            raise TypeError("Final pipeline step must implement predict")
        return model.predict(Xt)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return predicted probabilities when the final estimator supports them."""

        Xt = self._transform_until_model(X, fit=False)
        model = self.steps[-1][1]
        if not hasattr(model, "predict_proba"):
            raise TypeError("Final pipeline step does not implement predict_proba")
        return model.predict_proba(Xt)

    def score(self, X: Any, y: Any) -> float:
        """Return classification accuracy."""

        pred = self.predict(X)
        y_arr = np.asarray(y).reshape(-1)
        if pred.shape[0] != y_arr.shape[0]:
            raise ValueError("Predictions and y must have the same length")
        return float(np.mean(pred == y_arr))
