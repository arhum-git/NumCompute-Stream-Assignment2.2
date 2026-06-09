import os
import numpy as np

from numcompute_stream import (
    DecisionTreeClassifier, EnsembleClassifier, RandomForestClassifier,
    StandardScaler, Imputer, OneHotEncoder, Pipeline, StreamTrainer,
    StreamingMetric, StreamingConfusionMatrix, StreamingAUC, StreamingStats
)
from numcompute_stream.io import chunk_array, read_csv_numpy
from numcompute_stream.visualise import (
    plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth
)


def toy_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [2, 2]], dtype=float)
    y = np.array([0, 0, 1, 1, 1, 1])
    return X, y


def test_standard_scaler_partial_fit_mean():
    X, _ = toy_data()
    sc = StandardScaler().partial_fit(X[:3]).partial_fit(X[3:])
    assert np.allclose(sc.mean_, np.nanmean(X, axis=0))


def test_standard_scaler_transform_zero_mean():
    X, _ = toy_data()
    Z = StandardScaler().fit_transform(X)
    assert np.allclose(Z.mean(axis=0), [0, 0], atol=1e-8)


def test_standard_scaler_handles_nan():
    X = np.array([[1., np.nan], [3., 5.]])
    Z = StandardScaler().fit(X).transform(X)
    assert not np.isnan(Z).any()


def test_imputer_mean_streaming():
    X = np.array([[1., np.nan], [3., 7.], [np.nan, 9.]])
    imp = Imputer().partial_fit(X[:1]).partial_fit(X[1:])
    out = imp.transform(X)
    assert not np.isnan(out).any()
    assert np.allclose(imp.statistics_, [2., 8.])


def test_imputer_constant():
    X = np.array([[np.nan, 2.]])
    out = Imputer(strategy="constant", fill_value=-1).fit_transform(X)
    assert out[0, 0] == -1


def test_imputer_median():
    X = np.array([[1.], [100.], [2.]])
    imp = Imputer(strategy="median").fit(X)
    assert imp.statistics_[0] == 2


def test_one_hot_incremental_expands_categories():
    enc = OneHotEncoder().partial_fit(np.array(["a", "b"])).partial_fit(np.array(["c"]))
    out = enc.transform(np.array(["a", "c"]))
    assert out.shape == (2, 3)


def test_one_hot_unknown_ignore():
    enc = OneHotEncoder().fit(np.array(["a"]))
    assert enc.transform(np.array(["z"])).sum() == 0


def test_decision_tree_fit_predict_shape():
    X, y = toy_data()
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert clf.predict(X).shape == y.shape


def test_decision_tree_learns_simple_pattern():
    X, y = toy_data()
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert clf.score(X, y) >= 0.8


def test_decision_tree_partial_fit():
    X, y = toy_data()
    clf = DecisionTreeClassifier(max_depth=3).partial_fit(X[:3], y[:3]).partial_fit(X[3:], y[3:])
    assert clf.score(X, y) >= 0.8


def test_decision_tree_entropy():
    X, y = toy_data()
    clf = DecisionTreeClassifier(max_depth=3, criterion="entropy").fit(X, y)
    assert clf.predict(X).size == y.size


def test_decision_tree_nan_safe():
    X, y = toy_data()
    X[0, 0] = np.nan
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert clf.predict(X).size == y.size


def test_decision_tree_tie_prediction_deterministic():
    X = np.array([[0.], [0.]])
    y = np.array([1, 0])
    clf = DecisionTreeClassifier(max_depth=0).fit(X, y)
    assert clf.predict(np.array([[0.]]))[0] == 0


def test_ensemble_predict_shape():
    X, y = toy_data()
    ens = EnsembleClassifier(n_estimators=3, max_depth=3, random_state=1).fit(X, y)
    assert ens.predict(X).shape == y.shape


def test_ensemble_partial_fit_score():
    X, y = toy_data()
    ens = EnsembleClassifier(n_estimators=3, max_depth=3, random_state=1)
    ens.partial_fit(X[:3], y[:3]).partial_fit(X[3:], y[3:])
    assert ens.score(X, y) >= 0.5


def test_random_forest_alias():
    rf = RandomForestClassifier(n_estimators=2)
    assert isinstance(rf, EnsembleClassifier)


def test_streaming_confusion_matrix():
    cm = StreamingConfusionMatrix(classes=[0, 1]).update([0, 1, 1], [0, 0, 1]).result()
    assert cm.tolist() == [[1, 0], [1, 1]]


def test_streaming_accuracy():
    m = StreamingMetric("accuracy").update([0, 1], [0, 0]).update([1], [1])
    assert np.isclose(m.result(), 2 / 3)


def test_streaming_precision():
    m = StreamingMetric("precision", classes=[0, 1]).update([0, 1, 1], [0, 0, 1])
    assert 0 <= m.result() <= 1


def test_streaming_recall():
    m = StreamingMetric("recall", classes=[0, 1]).update([0, 1, 1], [0, 0, 1])
    assert 0 <= m.result() <= 1


def test_streaming_f1():
    m = StreamingMetric("f1", classes=[0, 1]).update([0, 1, 1], [0, 0, 1])
    assert 0 <= m.result() <= 1


def test_streaming_metric_reset():
    m = StreamingMetric("accuracy").update([1], [1])
    m.reset()
    assert m.result() == 0.0


def test_rolling_window_metric():
    m = StreamingMetric("accuracy", window_size=2).update([0, 1, 1], [0, 0, 1])
    assert np.isclose(m.result(), 0.5)


def test_streaming_auc():
    auc = StreamingAUC().update([0, 1, 1, 0], [0.1, 0.8, 0.7, 0.2]).result()
    assert auc > 0.9


def test_streaming_stats_mean_variance():
    s = StreamingStats().update_stats(np.array([[1., 2.], [3., 4.]])).update_stats(np.array([[5., 6.]]))
    assert np.allclose(s.mean, [3., 4.])
    assert np.all(s.variance >= 0)


def test_streaming_stats_quantile():
    s = StreamingStats().update_stats(np.array([[1.], [2.], [3.]]))
    assert s.quantile(0.5)[0] == 2


def test_streaming_stats_histogram():
    s = StreamingStats(bins=2).update_stats(np.array([[1.], [2.], [3.]]))
    counts, edges = s.histogram(0)
    assert counts.sum() == 3


def test_pipeline_partial_fit_predict():
    X, y = toy_data()
    pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(max_depth=3))])
    pipe.partial_fit(X[:3], y[:3]).partial_fit(X[3:], y[3:])
    assert pipe.predict(X).shape == y.shape


def test_stream_trainer_logs():
    X, y = toy_data()
    tr = StreamTrainer(DecisionTreeClassifier(max_depth=3))
    tr.fit_chunk(X[:3], y[:3])
    tr.fit_chunk(X[3:], y[3:])
    assert len(tr.logs["cumulative_accuracy"]) == 2


def test_stream_trainer_score_chunk():
    X, y = toy_data()
    tr = StreamTrainer(DecisionTreeClassifier(max_depth=3))
    tr.fit_chunk(X, y)
    assert 0 <= tr.score_chunk(X, y) <= 1


def test_chunk_array():
    X, y = toy_data()
    chunks = list(chunk_array(X, y, chunk_size=2))
    assert len(chunks) == 3


def test_read_csv_numpy(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b,c\n1,2,0\n3,,1\n")
    arr = read_csv_numpy(str(p))
    assert arr.shape == (2, 3)
    assert np.isnan(arr[1, 1])


def test_visualise_metric_save(tmp_path):
    path = tmp_path / "metric.png"
    plot_metric_over_time([0.5, 0.8], save_path=str(path), show=False)
    assert path.exists()


def test_visualise_compare_save(tmp_path):
    path = tmp_path / "compare.png"
    compare_models([0.5, 0.8], [0.4, 0.7], save_path=str(path), show=False)
    assert path.exists()


def test_visualise_predictions_save(tmp_path):
    path = tmp_path / "preds.png"
    plot_predictions_vs_ground_truth([0, 1], [0, 0], save_path=str(path), show=False)
    assert path.exists()


def test_tree_uses_training_nan_means_for_prediction():
    X = np.array([[0., 0.], [2., 2.], [4., 4.]])
    y = np.array([0, 1, 1])
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    pred = clf.predict(np.array([[np.nan, np.nan]]))
    assert pred.shape == (1,)
    assert not np.isnan(clf.feature_means_).any()


def test_tree_rejects_wrong_feature_count():
    X, y = toy_data()
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
    try:
        clf.predict(np.array([[1., 2., 3.]]))
    except ValueError as exc:
        assert "Expected" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_ensemble_predict_proba_vote_rows_sum_to_one():
    X, y = toy_data()
    ens = EnsembleClassifier(n_estimators=5, max_depth=3, random_state=2).fit(X, y)
    proba = ens.predict_proba(X)
    assert proba.shape == (X.shape[0], len(np.unique(y)))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_streaming_auc_ties_are_half_credit():
    auc = StreamingAUC().update([0, 1], [0.5, 0.5]).result()
    assert np.isclose(auc, 0.5)


def test_confusion_matrix_dynamic_class_expansion():
    cm = StreamingConfusionMatrix().update([0], [0]).update([1], [2]).result()
    assert cm.shape == (3, 3)
    assert cm.sum() == 2


def test_pipeline_duplicate_step_names_rejected():
    try:
        Pipeline([("scale", StandardScaler()), ("scale", DecisionTreeClassifier())])
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_components_are_introspectable():
    pipe = Pipeline([("scale", StandardScaler()), ("model", RandomForestClassifier(n_estimators=2))])
    comps = pipe.components()
    assert "scale" in comps and "model" in comps


def test_predict_before_fit_raises():
    """predict must raise RuntimeError on an unfitted tree."""
    import pytest

    clf = DecisionTreeClassifier()
    with pytest.raises(RuntimeError, match="fitted"):
        clf.predict(np.array([[1., 2.]]))


def test_ensemble_bagging_method():
    """Bagging should work without random-forest feature subsampling."""
    X, y = toy_data()
    ens = EnsembleClassifier(n_estimators=3, method="bagging", max_depth=2, random_state=0)
    ens.fit(X, y)
    assert ens.predict(X).shape == y.shape
    assert ens.max_features is None


def test_tree_zero_variance_chunk():
    """Zero-variance features should produce a valid majority-leaf model."""
    X = np.ones((8, 2))
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset(set(np.unique(y)))


def test_standard_scaler_with_std_false():
    """with_std=False should centre values without normalising variance."""
    X = np.array([[1., 2.], [5., 10.]], dtype=float)
    sc = StandardScaler(with_std=False).fit(X)
    Z = sc.transform(X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-8)
    assert not np.allclose(Z.std(axis=0), 1.0)


def test_pipeline_score():
    """Pipeline.score should return a valid accuracy value."""
    X, y = toy_data()
    pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(max_depth=3))])
    pipe.fit(X, y)
    s = pipe.score(X, y)
    assert 0.0 <= s <= 1.0


def test_one_hot_handle_unknown_error():
    """handle_unknown='error' should reject unseen categories."""
    import pytest

    enc = OneHotEncoder(handle_unknown="error").fit(np.array(["a", "b"]))
    with pytest.raises(ValueError, match="Unknown category"):
        enc.transform(np.array(["z"]))


def test_stream_trainer_memory_log_grows():
    """memory_bytes log should have one entry per fit_chunk call."""
    X, y = toy_data()
    tr = StreamTrainer(DecisionTreeClassifier(max_depth=3))
    for Xc, yc in zip(np.split(X, 2), np.split(y, 2)):
        tr.fit_chunk(Xc, yc)
    assert len(tr.logs["memory_bytes"]) == 2
    assert all(m > 0 for m in tr.logs["memory_bytes"])


def test_stream_trainer_reset_clears_logs_and_metric():
    """StreamTrainer.reset should clear logs and metric state."""
    X, y = toy_data()
    tr = StreamTrainer(DecisionTreeClassifier(max_depth=3))
    tr.fit_chunk(X, y)
    tr.reset()
    assert tr.logs["chunk"] == []
    assert tr.metric.result() == 0.0
