# Assignment 2.2 Specs: NumCompute-Stream

## Module Changes & New Components
All components must now support **streaming settings**, with support for incremental `.partial_fit()` and per-chunk updates.

### `stream.py`
- Implements `StreamTrainer`, managing model + pipeline + logging
- Supports `.fit_chunk(X, y)` and `.score_chunk(X, y)` logic
- Logs per-chunk metrics, memory footprint, and cumulative accuracy

### `tree.py`
- `DecisionTreeClassifier`: depth-limited, Gini or entropy-based
- `.partial_fit(X_chunk, y_chunk)` for online growth
- Support config: `max_depth`, `min_samples_split`, `max_features`

### `ensemble.py`
- `EnsembleClassifier`: manages N decision trees with at least one ensemble method
- `.partial_fit()` and `.predict()` must support streaming adaptation

### `preprocessing.py`
- Scalers such as `StandardScaler` must support `.partial_fit()`
- Maintain running mean/variance using a stable online method
- `Imputer` updates missing-value estimates on the fly
- `OneHotEncoder` expands categories incrementally

### `stats.py`
- All statistical functions redesigned for streaming
- Implement chunk-based mean, variance, quantiles, and histograms
- Add `update_stats(X_chunk)` API

### `metrics.py`
- Classification metrics such as accuracy, precision, recall, and F1 must support:
  - Streaming updates via `update(y_true_chunk, y_pred_chunk)`
  - `reset()` and `result()` methods
  - Confusion matrix and AUC accumulation over time
  - Rolling-window metrics

### `pipeline.py`
- Must support `partial_fit()` for models and transformers
- Support incremental transformation and prediction in chained pipelines

```python
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipe.partial_fit(X_chunk, y_chunk)
```

### `visualise.py`
- Provide reusable plotting functions using `matplotlib`
- Must be usable across scripts, demos, or pipeline logs
- Required plots:
  - `plot_metric_over_time(metric_values, title, ylabel)`
  - `compare_models(metric1, metric2, labels)`
  - `plot_predictions_vs_ground_truth(y_true, y_pred)`
- Should support options for saving to file or inline display.
