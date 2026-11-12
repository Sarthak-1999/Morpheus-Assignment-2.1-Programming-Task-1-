"""Integration, streaming regressions and edge cases.

These tests deliberately exercise *interfaces between components*, not just
single methods.  The small, deterministic fixtures permit independent manual
verification, while the Titanic test uses the original NumCompute CSV reader.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from numcompute_stream.io import iter_supervised_csv
from numcompute_stream.metrics import (
    AUC, Accuracy, ConfusionMatrix, StreamingClassificationMetrics, binary_auc,
)
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler
from numcompute_stream.stats import StreamingStats
from numcompute_stream.stream import StreamTrainer, estimate_memory_bytes
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture
def seed_data():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(72, 3))
    X[::11, 0] = np.nan
    X[::9, 1] = np.nan
    y = np.where(np.nan_to_num(X[:, 0]) + X[:, 2] > 0, 'positive', 'negative')
    return X, y


@pytest.mark.parametrize('cuts', [[1, 3, 5, 12, 26], [2, 2, 2, 2, 2, 2, 14], [7, 7, 7, 7]])
def test_running_statistics_match_single_batch(seed_data, cuts):
    X, _ = seed_data
    ref = StreamingStats().fit(X)
    stream = StreamingStats()
    start = 0
    for size in cuts:
        stream.partial_fit(X[start:start + size])
        start += size
    stream.partial_fit(X[start:])
    np.testing.assert_allclose(stream.mean(), ref.mean(), atol=1e-12)
    np.testing.assert_allclose(stream.variance(), ref.variance(), atol=1e-12)
    np.testing.assert_allclose(stream.quantile([.25, .5, .75]), ref.quantile([.25, .5, .75]))
    np.testing.assert_array_equal(stream.histogram(bins=7)[0], ref.histogram(bins=7)[0])


@pytest.mark.parametrize('cut', [1, 5, 27, 55])
def test_stream_standardise_matches_batch(seed_data, cut):
    X, _ = seed_data
    stream = StandardScaler().partial_fit(X[:cut]).partial_fit(X[cut:])
    batch = StandardScaler().fit(X)
    np.testing.assert_allclose(stream.mean_, batch.mean_, atol=1e-12)
    np.testing.assert_allclose(stream.var_, batch.var_, atol=1e-12)
    np.testing.assert_allclose(stream.transform(X), batch.transform(X), equal_nan=True)


@pytest.mark.parametrize('cut', [1, 11, 35, 71])
def test_stream_impute_mean_matches_batch(seed_data, cut):
    X, _ = seed_data
    stream = Imputer().partial_fit(X[:cut]).partial_fit(X[cut:])
    batch = Imputer().fit(X)
    np.testing.assert_allclose(stream.statistics_, batch.statistics_)
    np.testing.assert_allclose(stream.transform(X), batch.transform(X))


def test_two_stage_transformer_chain_no_missing_output(seed_data):
    X, y = seed_data
    pipe = Pipeline([
        ('impute', Imputer(strategy='mean')),
        ('scale', StandardScaler()),
        ('tree', DecisionTreeClassifier(buffer_size=12, max_depth=3))])
    for sl in (slice(0, 8), slice(8, 16), slice(16, 33), slice(33, 50)):
        pipe.partial_fit(X[sl], y[sl])
    assert pipe.n_retrains_ == 3
    assert pipe.n_buffered_samples_ == 0
    predictions = pipe.predict(X[50:])
    assert len(predictions) == 22
    assert np.isin(predictions, ['positive', 'negative']).all()
    np.testing.assert_allclose(pipe.predict_proba(X[50:]).sum(axis=1), 1.)


@pytest.mark.parametrize('buffer_size,chunks,expected_retrains,remaining', [
    (1, [2, 3, 4], 3, 0),
    (5, [2, 3, 4], 1, 4),
    (5, [2, 5, 2], 1, 2),
    (4, [9], 1, 0),
    (20, [2, 3, 4], 0, 9),
    (3, [1, 1, 1, 1, 1, 1], 2, 0),
])
def test_buffer_thresholds_with_variable_chunk_sizes(buffer_size, chunks, expected_retrains, remaining):
    model = DecisionTreeClassifier(buffer_size=buffer_size, max_depth=0)
    for n in chunks:
        model.partial_fit(np.zeros((n, 1)), np.zeros(n, dtype=int))
    assert model.n_retrains_ == expected_retrains
    assert model.n_buffered_samples_ == remaining
    assert model.n_samples_seen_ == sum(chunks)
    assert sum(len(v) for v in model._buffer_X) == remaining


@pytest.mark.parametrize('buffer_size,chunks,expected_retrains,remaining', [
    (2, [1, 1, 1, 1], 2, 0),
    (5, [3, 4, 2], 1, 2),
    (5, [10, 3], 1, 3),
    (15, [3, 4, 2], 0, 9),
])
def test_pipeline_buffer_matches_tree_contract(buffer_size, chunks, expected_retrains, remaining):
    pipe = Pipeline([('impute', Imputer()), ('tree', DecisionTreeClassifier(buffer_size=buffer_size))])
    for n in chunks:
        pipe.update(np.arange(n, dtype=float).reshape(-1, 1), np.zeros(n, dtype=int))
    assert pipe.n_retrains_ == expected_retrains
    assert pipe.n_buffered_samples_ == remaining
    assert pipe.n_samples_seen_ == sum(chunks)
    assert pipe.is_fitted == (expected_retrains > 0)


def test_pipeline_frozen_scaling_inference_during_partial_buffer():
    pipe = Pipeline([('scale', StandardScaler()), ('tree', DecisionTreeClassifier(max_depth=1, buffer_size=4))])
    pipe.partial_fit([[0], [1], [8], [9]], [0, 0, 1, 1])
    q = np.array([[0], [1], [8], [9]])
    old = pipe.predict(q).copy()
    original_mean = pipe._prediction_transformers[0][1].mean_.copy()
    pipe.partial_fit([[1000], [3000]], [1, 1])
    np.testing.assert_array_equal(pipe.predict(q), old)
    np.testing.assert_array_equal(pipe._prediction_transformers[0][1].mean_, original_mean)
    assert pipe.named_steps['scale'].mean_[0] != original_mean[0]


def test_frozen_multicolumn_one_hot_while_first_column_expands():
    pipe = Pipeline([('encode', OneHotEncoder()), ('tree', DecisionTreeClassifier(buffer_size=4))])
    pipe.partial_fit([['x', 'red'], ['y', 'red'], ['x', 'blue'], ['y', 'blue']], [0, 1, 0, 1])
    original = pipe.predict([['x', 'red'], ['y', 'blue']]).copy()
    old_width = pipe.model.n_features_in_
    pipe.partial_fit([['z', 'blue'], ['z', 'red']], [1, 1])
    assert pipe.named_steps['encode'].n_features_out_ == old_width + 1
    np.testing.assert_array_equal(pipe.predict([['x', 'red'], ['y', 'blue']]), original)
    pipe.partial_fit([['z', 'red'], ['z', 'blue']], [1, 1])
    assert pipe.n_retrains_ == 2
    assert pipe.model.n_features_in_ == old_width + 1
    assert pipe.predict([['z', 'blue']]).shape == (1,)


def test_missing_category_and_unknown_category_are_distinct():
    pipe = Pipeline([('encode', OneHotEncoder()), ('tree', DecisionTreeClassifier(buffer_size=3))])
    pipe.partial_fit([['a'], [None], ['b']], [0, 1, 0])
    p_unknown = pipe.predict_proba([['unseen']])
    p_missing = pipe.predict_proba([[None]])
    assert p_unknown.shape == p_missing.shape == (1, 2)
    assert np.isfinite(p_unknown).all()
    assert np.isfinite(p_missing).all()


def test_pipeline_retrain_replaces_tree_not_accumulates_old_classes():
    pipe = Pipeline([('model', DecisionTreeClassifier(buffer_size=3, max_depth=0))])
    pipe.partial_fit([[0], [1], [2]], ['a', 'a', 'a'])
    assert pipe.predict([[100]]).tolist() == ['a']
    pipe.partial_fit([[3]], ['b'])
    assert pipe.predict([[100]]).tolist() == ['a']
    pipe.partial_fit([[4], [5]], ['b', 'b'])
    assert pipe.predict([[100]]).tolist() == ['b']
    assert pipe.model.classes_.tolist() == ['b']


def test_direct_tree_final_short_buffer_is_not_retrained():
    tree = DecisionTreeClassifier(buffer_size=4, max_depth=0)
    tree.partial_fit(np.arange(4).reshape(-1, 1), np.zeros(4, dtype=int))
    tree.partial_fit(np.arange(3).reshape(-1, 1), np.ones(3, dtype=int))
    assert tree.n_buffered_samples_ == 3
    assert tree.n_retrains_ == 1
    np.testing.assert_array_equal(tree.predict([[0], [1]]), [0, 0])


@pytest.mark.parametrize('criterion', ['gini', 'entropy'])
def test_nan_and_constant_features_during_tree_retrain(criterion):
    X = np.array([[np.nan, 1.0], [0.0, 1.0], [1.0, 1.0], [np.nan, 1.0]])
    tree = DecisionTreeClassifier(criterion=criterion, max_depth=2, buffer_size=4)
    tree.partial_fit(X[:2], [1, 0])
    tree.partial_fit(X[2:], [1, 1])
    assert tree.predict(X).shape == (4,)
    np.testing.assert_allclose(tree.predict_proba(X).sum(axis=1), 1.)


@pytest.mark.parametrize('criterion', ['gini', 'entropy'])
def test_majority_ties_predict_first_sorted_label(criterion):
    tree = DecisionTreeClassifier(max_depth=0, criterion=criterion, buffer_size=2)
    tree.partial_fit([[0], [0]], ['z', 'a'])
    np.testing.assert_array_equal(tree.predict([[0], [1]]), ['a', 'a'])


def test_new_class_in_next_buffer_updates_probability_columns():
    tree = DecisionTreeClassifier(buffer_size=2, max_depth=0)
    tree.partial_fit([[0], [1]], ['a', 'b'])
    assert tree.predict_proba([[2]]).shape == (1, 2)
    tree.partial_fit([[0], [1]], ['c', 'c'])
    assert tree.classes_.tolist() == ['c']
    assert tree.predict_proba([[2]]).shape == (1, 1)


def test_prequential_scores_are_measured_before_training_change():
    trainer = StreamTrainer(DecisionTreeClassifier(max_depth=0, buffer_size=2), positive_label=1)
    first = trainer.fit_chunk([[0], [0]], [0, 0])
    second = trainer.fit_chunk([[0], [0]], [1, 1])
    third = trainer.fit_chunk([[0]], [1])
    assert not first['evaluated']
    assert second['chunk_accuracy'] == 0.0
    assert third['chunk_accuracy'] == 1.0
    assert trainer.result()['support'] == 3
    assert trainer.result()['n_warmup_samples'] == 2


def test_trainer_score_only_changes_metrics_not_training_or_history():
    model = DecisionTreeClassifier(buffer_size=2).partial_fit([[0], [1]], [0, 1])
    trainer = StreamTrainer(model)
    before = (model.n_samples_seen_, model.n_retrains_, model.n_buffered_samples_)
    trainer.score_chunk([[0], [1], [1]], [0, 1, 1])
    assert (model.n_samples_seen_, model.n_retrains_, model.n_buffered_samples_) == before
    assert trainer.history_ == []
    assert trainer.result()['support'] == 3


def test_trainer_cumulative_accuracy_is_weighted_by_samples():
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=2, max_depth=0))
    trainer.fit_chunk([[0], [0]], [0, 0])
    trainer.fit_chunk([[1]], [0])  # 1/1 correct
    trainer.fit_chunk([[2], [3], [4]], [1, 1, 1])  # 0/3 correct; retrain on four
    assert trainer.result()['accuracy'] == pytest.approx(.25)
    assert trainer.result()['support'] == 4


def test_logged_retraining_counter_monotonic_and_thresholds():
    trainer = StreamTrainer(Pipeline([('tree', DecisionTreeClassifier(buffer_size=5))]))
    sizes = [1, 2, 1, 1, 4, 1, 2, 3]
    for n in sizes:
        trainer.fit_chunk(np.zeros((n, 1)), np.zeros(n, dtype=int))
    history = trainer.history_
    assert [r['n_retrains'] for r in history] == [0, 0, 0, 1, 1, 2, 2, 3]
    assert [r['buffered_samples'] for r in history] == [1, 3, 4, 0, 4, 0, 2, 0]
    assert [r['chunk'] for r in history] == list(range(1, 9))
    assert all(r['update_ms'] >= 0 for r in history)
    assert all(r['memory_bytes'] > 0 for r in history)


def test_trainer_confusion_matrix_sum_matches_evaluated_rows(seed_data):
    X, y = seed_data
    trainer = StreamTrainer(Pipeline([('impute', Imputer()), ('tree', DecisionTreeClassifier(buffer_size=12))]))
    for start in range(0, len(X), 8):
        trainer.fit_chunk(X[start:start+8], y[start:start+8])
    summary = trainer.result()
    assert summary['confusion_matrix'].sum() == summary['support']
    assert summary['support'] + summary['n_warmup_samples'] == len(X)
    assert len(trainer.history_) == 9
    assert summary['n_retrains'] == 4  # nine chunks of 8 at threshold 12


def test_metrics_streaming_equals_confusion_matrix_reference():
    truth = np.array([0, 1, 2, 2, 1, 0, 0, 2, 1])
    pred = np.array([0, 0, 1, 2, 1, 2, 0, 2, 1])
    m = StreamingClassificationMetrics(window_size=4)
    for ix in (slice(0, 2), slice(2, 5), slice(5, 9)):
        m.update(truth[ix], pred[ix])
    expected = np.array([[2, 0, 1], [1, 2, 0], [0, 1, 2]])
    np.testing.assert_array_equal(m.result()['confusion_matrix'], expected)
    assert m.result()['support'] == 9
    assert m.result()['accuracy'] == pytest.approx(6/9)
    assert m.result()['rolling_accuracy'] == pytest.approx(3/4)


@pytest.mark.parametrize('window_size,expected', [(1, 0), (2, .5), (3, 1/3), (4, .5), (6, 4/6)])
def test_rolling_metrics_respect_sample_count_not_chunk_count(window_size, expected):
    m = StreamingClassificationMetrics(window_size=window_size)
    m.update([0, 0, 0], [0, 0, 0])
    m.update([0], [1])
    m.update([1, 1], [1, 0])
    assert m.result()['rolling_accuracy'] == pytest.approx(expected)


def test_metric_reset_discards_confusion_scores_and_rolling_window():
    m = StreamingClassificationMetrics(window_size=2, positive_label=1)
    m.update([0, 1], [0, 1], [.1, .9])
    assert m.result()['auc'] == 1
    m.reset()
    result = m.result()
    assert result['support'] == 0
    assert math.isnan(result['auc'])
    assert math.isnan(result['rolling_accuracy'])
    assert not m._score_history


@pytest.mark.parametrize('positive', [0, 1])
def test_auc_positive_label_orientation(positive):
    y = np.array([0, 1, 0, 1])
    s = np.array([.1, .9, .2, .8])
    scores = s if positive == 1 else 1-s
    assert binary_auc(y, scores, positive_label=positive) == 1


def test_auc_tied_scores_grouping_across_chunks():
    m = AUC(positive_label=1)
    m.update([0, 1], [.5, .5])
    m.update([0, 1], [.5, .5])
    assert m.result() == .5


def test_metrics_failed_update_does_not_change_prior_counts():
    m = StreamingClassificationMetrics(labels=[0, 1], window_size=2)
    m.update([0, 1], [0, 1])
    before = m.result()['confusion_matrix'].copy()
    with pytest.raises(ValueError, match='finite'):
        m.update([1], [1], [np.nan])
    np.testing.assert_array_equal(m.result()['confusion_matrix'], before)
    assert m.result()['support'] == 2


def test_metrics_failed_score_update_does_not_discover_new_labels():
    m = StreamingClassificationMetrics()
    m.update(['a'], ['a'])
    with pytest.raises(ValueError, match='finite'):
        m.update(['new'], ['new'], [np.nan])
    assert m.result()['labels'].tolist() == ['a']


def test_invalid_tree_chunk_does_not_mutate_buffer():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        tree.partial_fit([[1], [np.inf]], [1, 1])
    assert tree.n_buffered_samples_ == 2
    assert tree.n_samples_seen_ == 2


def test_invalid_pipeline_feature_count_does_not_increment_seen():
    pipe = Pipeline([('tree', DecisionTreeClassifier(buffer_size=3))])
    pipe.partial_fit([[0, 0]], [0])
    with pytest.raises(ValueError, match='Expected'):
        pipe.partial_fit([[1]], [1])
    assert pipe.n_samples_seen_ == 1
    assert pipe.n_buffered_samples_ == 1


def test_csv_loading_and_streaming_trainer_end_to_end(tmp_path):
    path = tmp_path / 'small.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['unused', 'x1', 'x2', 'label'])
        for i in range(13):
            writer.writerow([f'row{i}', i, '' if i % 3 == 0 else i / 2, i % 2])
    chunks = list(iter_supervised_csv(path, features=['x1', 'x2'], target='label', chunksize=4))
    assert [len(y) for _, y in chunks] == [4, 4, 4, 1]
    assert np.isnan(chunks[0][0][0, 1])
    pipeline = Pipeline([('imp', Imputer()), ('scale', MinMaxScaler()),
                         ('tree', DecisionTreeClassifier(buffer_size=5, max_depth=2))])
    trainer = StreamTrainer(pipeline, positive_label='1', window_size=5)
    for X, y in chunks:
        trainer.fit_chunk(X, y)
    result = trainer.result()
    assert result['n_samples_seen'] == 13
    assert result['n_warmup_samples'] == 8
    assert result['support'] == 5
    assert result['n_retrains'] == 2
    assert result['buffered_samples'] == 0
    assert len(trainer.history_) == 4


def test_titanic_csv_reader_and_pipeline_contract():
    csv_path = Path(__file__).resolve().parents[1] / 'demo' / 'data' / 'titanic.csv'
    pipeline = Pipeline([('impute', Imputer()), ('scale', StandardScaler()),
                         ('tree', DecisionTreeClassifier(buffer_size=200, max_depth=4))])
    trainer = StreamTrainer(pipeline, positive_label='1', window_size=100)
    sizes = []
    for features, labels in iter_supervised_csv(csv_path, features=['pclass', 'age', 'fare'],
                                                target='survived', chunksize=100):
        sizes.append(len(labels))
        trainer.fit_chunk(features, labels)
    result = trainer.result()
    assert sizes == [100] * 8 + [91]
    assert result['n_samples_seen'] == 891
    assert result['n_warmup_samples'] == 200
    assert result['support'] == 691
    assert result['buffered_samples'] == 91
    assert result['n_retrains'] == 4
    assert result['confusion_matrix'].sum() == 691
    assert np.isfinite(result['accuracy'])


def test_stream_memory_estimate_handles_shared_references():
    shared = np.arange(100, dtype=float)
    data = [shared, shared, {'same': shared}]
    size = estimate_memory_bytes(data)
    assert size > shared.nbytes
    assert size < estimate_memory_bytes([shared.copy(), shared.copy(), {'same': shared.copy()}])


def test_random_data_reproducible_by_fixed_tree_seed():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(45, 5))
    y = (X[:, 0] + X[:, 3] > 0).astype(int)
    a = DecisionTreeClassifier(max_depth=3, max_features='sqrt', random_state=17, buffer_size=9)
    b = DecisionTreeClassifier(max_depth=3, max_features='sqrt', random_state=17, buffer_size=9)
    for ix in range(0, 45, 3):
        a.partial_fit(X[ix:ix+3], y[ix:ix+3])
        b.partial_fit(X[ix:ix+3], y[ix:ix+3])
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert a.n_retrains_ == b.n_retrains_ == 5


def test_real_stream_trainer_history_renders_dashboard(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from numcompute_stream.visualise import plot_stream_dashboard

    model = Pipeline([('model', DecisionTreeClassifier(buffer_size=2))])
    trainer = StreamTrainer(model, positive_label='1', window_size=3)
    for X, y in [([[0], [1]], ['0', '1']),
                 ([[2]], ['1']),
                 ([[3], [4]], ['0', '1'])]:
        trainer.fit_chunk(X, y)
    file = tmp_path / 'dashboard.png'
    fig, axes = plot_stream_dashboard(trainer.history_, save_path=file)
    try:
        assert axes.shape == (2, 2)
        assert file.exists() and file.stat().st_size > 1000
    finally:
        plt.close(fig)


def test_real_tree_predictions_render_scored_chunk_plot(tmp_path):
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from numcompute_stream.visualise import plot_predictions_vs_ground_truth

    tree = DecisionTreeClassifier(buffer_size=4, max_depth=2)
    tree.partial_fit([[0], [1], [5], [6]], [0, 0, 1, 1])
    actual = np.array([0, 0, 1, 1])
    predicted = tree.predict([[0], [4], [5], [6]])
    file = tmp_path / 'predictions.png'
    fig, ax = plot_predictions_vs_ground_truth(actual, predicted, save_path=file)
    try:
        assert file.exists() and file.stat().st_size > 1000
        assert str(len(actual)) in ax.get_title()
    finally:
        plt.close(fig)


def test_trainer_warmup_never_included_in_cumulative_metrics():
    tree = DecisionTreeClassifier(buffer_size=5, max_depth=0)
    trainer = StreamTrainer(tree, window_size=5)
    for _ in range(4):
        row = trainer.fit_chunk([[0]], [0])
        assert not row['evaluated']
        assert math.isnan(row['chunk_accuracy'])
        assert row['n_evaluated_total'] == 0
    fifth = trainer.fit_chunk([[0]], [0])
    assert not fifth['evaluated']
    sixth = trainer.fit_chunk([[0], [0]], [1, 1])
    assert sixth['evaluated'] and sixth['chunk_accuracy'] == 0
    assert trainer.result()['n_warmup_samples'] == 5
    assert trainer.result()['support'] == 2


def test_streaming_auc_accumulates_scores_only_on_scored_chunks():
    m = StreamingClassificationMetrics(positive_label='yes')
    m.update(['yes', 'no'], ['yes', 'no'], [.9, .1])
    m.update(['yes', 'no'], ['no', 'yes'])  # No probabilities for these rows.
    assert m.result()['support'] == 4
    assert m.result()['auc'] == 1.0
    assert sum(len(a) for a, _ in m._score_history) == 2


def test_partial_fit_after_immediate_fit_respects_new_threshold():
    pipe = Pipeline([('scale', StandardScaler()),
                     ('tree', DecisionTreeClassifier(buffer_size=3, max_depth=0))])
    pipe.fit([[0], [1]], [0, 0])
    assert pipe.is_fitted and pipe.n_buffered_samples_ == 0
    pipe.partial_fit([[20], [21]], [1, 1])
    assert pipe.n_retrains_ == 0
    assert pipe.predict([[100]]).tolist() == [0]
    pipe.partial_fit([[22]], [1])
    assert pipe.n_retrains_ == 1
    assert pipe.predict([[100]]).tolist() == [1]
