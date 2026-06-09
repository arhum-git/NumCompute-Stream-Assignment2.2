## What is included

```text
numcompute_stream/
├── io.py              # NumPy CSV loading and chunk generation
├── preprocessing.py   # Streaming StandardScaler, Imputer, OneHotEncoder
├── stats.py           # Streaming mean, variance, quantiles, histograms
├── metrics.py         # Streaming accuracy, precision, recall, F1, confusion matrix, AUC
├── tree.py            # DecisionTreeClassifier with partial_fit
├── ensemble.py        # Bagging / Random-Forest style ensemble classifier
├── pipeline.py        # Streaming pipeline with partial_fit
├── stream.py          # StreamTrainer with per-chunk logs
└── visualise.py       # Matplotlib plotting utilities
```


## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Quick usage

```python
import numpy as np
from numcompute_stream import Pipeline, StandardScaler, EnsembleClassifier

X = np.random.randn(200, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", EnsembleClassifier(n_estimators=5, max_depth=4, random_state=42))
])

for start in range(0, len(X), 32):
    X_chunk = X[start:start + 32]
    y_chunk = y[start:start + 32]
    pipe.partial_fit(X_chunk, y_chunk)

pred = pipe.predict(X)
print("Accuracy:", np.mean(pred == y))
```

## Streaming trainer

```python
from numcompute_stream import DecisionTreeClassifier, StreamTrainer

trainer = StreamTrainer(DecisionTreeClassifier(max_depth=5))

for X_chunk, y_chunk in chunks:
    trainer.fit_chunk(X_chunk, y_chunk)

print(trainer.logs["cumulative_accuracy"])
```

## Visualisation

```python
from numcompute_stream.visualise import plot_metric_over_time

plot_metric_over_time(
    trainer.logs["cumulative_accuracy"],
    title="Cumulative accuracy over chunks",
    ylabel="Accuracy"
)
```

## Demo

Open and run:

```text
demo/stream_demo.ipynb
```

The notebook demonstrates loading data, splitting into chunks, training a streaming pipeline, logging metrics, and visualising results.

## Benchmark

Run:

```bash
python benchmark/stream_benchmark.py
```

This compares a single decision tree against a small tree ensemble under streaming chunk updates.

## Tests

Run:

```bash
pytest
```

