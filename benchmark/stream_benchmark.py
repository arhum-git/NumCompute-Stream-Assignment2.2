"""Streaming benchmark for base tree vs ensemble.
Run: python benchmark/stream_benchmark.py
"""
import os
import sys
import time
import platform
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from numcompute_stream import DecisionTreeClassifier, EnsembleClassifier, StreamTrainer
from numcompute_stream.io import chunk_array


def make_data(n=400, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = ((X[:, 0] + X[:, 1] * 0.5 - X[:, 2] * 0.25) > 0).astype(int)
    return X, y


def run_model(model, X, y, chunk_size=50):
    trainer = StreamTrainer(model)
    start = time.perf_counter()
    for Xc, yc in chunk_array(X, y, chunk_size):
        trainer.fit_chunk(Xc, yc)
    elapsed = time.perf_counter() - start
    return elapsed, trainer.logs["cumulative_accuracy"][-1]


if __name__ == "__main__":
    X, y = make_data()
    tree_time, tree_acc = run_model(DecisionTreeClassifier(max_depth=4), X, y)
    ens_time, ens_acc = run_model(EnsembleClassifier(n_estimators=5, max_depth=4, random_state=42), X, y)
    print("Streaming benchmark")
    print(f"DecisionTreeClassifier: time={tree_time:.4f}s cumulative_accuracy={tree_acc:.4f}")
    print(f"EnsembleClassifier:     time={ens_time:.4f}s cumulative_accuracy={ens_acc:.4f}")
    print(f"Environment: Python {platform.python_version()}, NumPy {np.__version__}")
