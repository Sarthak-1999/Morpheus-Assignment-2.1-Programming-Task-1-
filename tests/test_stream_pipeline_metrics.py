"""Streaming metrics, frozen-transform pipelines, and prequential training."""
import math

import numpy as np
import pytest

from numcompute_stream import (Accuracy, AUC, ConfusionMatrix, DecisionTreeClassifier,
    F1Score, Imputer, OneHotEncoder, Pipeline, Precision, Recall, StandardScaler,
    StreamTrainer, StreamingClassificationMetrics, binary_auc)
from numcompute_stream.stream import estimate_memory_bytes


def test_metrics_basic_counts():
    m = StreamingClassificationMetrics().update([0, 0, 1, 1], [0, 1, 1, 0]).result()
    assert m['accuracy'] == .5
    assert m['precision'] == .5 and m['recall'] == .5 and m['f1'] == .5
    np.testing.assert_array_equal(m['confusion_matrix'], [[1, 1], [1, 1]])


def test_metrics_incremental_equals_batch():
    a = StreamingClassificationMetrics().update([0, 1], [0, 0]).update([1, 0], [1, 0])
    b = StreamingClassificationMetrics().update([0, 1, 1, 0], [0, 0, 1, 0])
    np.testing.assert_array_equal(a.result()['confusion_matrix'], b.result()['confusion_matrix'])


def test_metrics_dynamic_new_class():
    m = StreamingClassificationMetrics().update(['cat'], ['dog']).update(['dog', 'fox'], ['fox', 'dog'])
    assert list(m.result()['labels']) == ['cat', 'dog', 'fox']
    assert m.result()['confusion_matrix'].sum() == 3


def test_metrics_fixed_labels():
    m = StreamingClassificationMetrics(labels=[1, 0])
    np.testing.assert_array_equal(m.update([0, 1], [1, 1]).result()['confusion_matrix'], [[1, 0], [1, 0]])
    with pytest.raises(ValueError, match='Unknown'):
        m.update([2], [1])


def test_metrics_label_reset():
    m = StreamingClassificationMetrics().update([0, 1], [1, 1])
    assert m.reset().result()['support'] == 0


def test_metrics_empty_undefined():
    value = StreamingClassificationMetrics().result()
    assert math.isnan(value['accuracy']) and math.isnan(value['auc'])


def test_metrics_missing_label_rejected():
    with pytest.raises(ValueError, match='missing'):
        StreamingClassificationMetrics().update([0, np.nan], [1, 0])


@pytest.mark.parametrize('actual,predicted', [([0], [0, 1]), ([[0]], [0]), ([], [])])
def test_metrics_shape_validation(actual, predicted):
    with pytest.raises(ValueError):
        StreamingClassificationMetrics().update(actual, predicted)


def test_metrics_fixed_labels_scores():
    m = StreamingClassificationMetrics(labels=[0, 1], positive_label=1)
    m.update([0, 1], [0, 1], [0.1, .9])
    assert m.result()['auc'] == 1


def test_auc_all_tied_is_half():
    assert binary_auc([0, 1, 0, 1], [.5, .5, .5, .5]) == .5


def test_auc_unambiguous_perfect():
    assert binary_auc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == 1.


def test_auc_inverted_is_zero():
    assert binary_auc([0, 1], [1., 0.]) == 0.


def test_auc_one_class_nan():
    assert math.isnan(binary_auc([1, 1], [.3, .9]))


def test_auc_missing_labels_rejected():
    with pytest.raises(ValueError, match='missing'):
        binary_auc([0, float('nan')], [.1, .9])


def test_auc_unknown_scores_rejected():
    with pytest.raises(ValueError, match='finite'):
        binary_auc([0, 1], [0.5, float('nan')])


def test_metrics_auc_across_chunks():
    metric = StreamingClassificationMetrics(positive_label=1)
    metric.update([0, 1], [0, 1], [.2, .8])
    metric.update([1, 0], [1, 0], [.9, .1])
    assert metric.result()['auc'] == 1.


def test_rolling_window_keeps_only_latest_samples():
    metric = StreamingClassificationMetrics(window_size=3)
    metric.update([0, 0], [1, 1])
    metric.update([1, 1, 1], [1, 1, 1])
    assert metric.result()['rolling_accuracy'] == 1.
    assert metric.result()['accuracy'] == 3/5


def test_rolling_partial_chunk_trim():
    m = StreamingClassificationMetrics(window_size=2)
    m.update([1, 0, 0, 1], [1, 1, 1, 1])
    assert m.result()['rolling_accuracy'] == .5


def test_metric_wrappers():
    for cls, expected in [(Accuracy, .5), (Precision, .5), (Recall, .5), (F1Score, .5)]:
        m = cls().update([0, 1], [1, 1]).update([1, 0], [0, 0])
        assert m.result() == expected
        assert math.isnan(m.reset().result())


def test_confusion_matrix_incremental():
    m = ConfusionMatrix(labels=[0, 1]).update([0, 1], [0, 0])
    np.testing.assert_array_equal(m.result(), [[1, 0], [1, 0]])
    np.testing.assert_array_equal(m.reset().result(), [[0, 0], [0, 0]])


def test_auc_wrapper():
    m = AUC().update([0, 1], [.1, .9])
    assert m.result() == 1.
    assert math.isnan(m.reset().result())


def test_pipeline_requires_tree_last():
    with pytest.raises(TypeError, match='DecisionTreeClassifier'):
        Pipeline([('scale', StandardScaler())])


def test_pipeline_unique_names():
    with pytest.raises(ValueError, match='unique'):
        Pipeline([('x', StandardScaler()), ('x', DecisionTreeClassifier())])


def test_pipeline_unfitted_prediction():
    pipe = Pipeline([('model', DecisionTreeClassifier(buffer_size=2))])
    with pytest.raises(RuntimeError, match='not trained'):
        pipe.predict([[1.]])


def test_pipeline_buffer_threshold():
    pipe = Pipeline([('impute', Imputer()), ('scale', StandardScaler()),
                     ('tree', DecisionTreeClassifier(buffer_size=4))])
    pipe.partial_fit([[1.], [2.]], [0, 0])
    assert not pipe.is_fitted and pipe.n_buffered_samples_ == 2
    pipe.partial_fit([[3.], [4.]], [1, 1])
    assert pipe.is_fitted and pipe.n_retrains_ == 1 and pipe.n_buffered_samples_ == 0
    assert pipe.predict([[1.], [4.]]).tolist() == [0, 1]


def test_pipeline_keeps_old_tree_and_transform_during_buffer():
    pipe = Pipeline([('scale', StandardScaler()),
                     ('tree', DecisionTreeClassifier(max_depth=1, buffer_size=4))])
    pipe.partial_fit([[0.], [1.], [2.], [3.]], [0, 0, 1, 1])
    prediction = pipe.predict([[0.], [3.]]).copy()
    frozen_mean = pipe._prediction_transformers[0][1].mean_.copy()
    pipe.partial_fit([[1000.]], [1])
    np.testing.assert_array_equal(pipe.predict([[0.], [3.]]), prediction)
    np.testing.assert_array_equal(pipe._prediction_transformers[0][1].mean_, frozen_mean)
    assert pipe.named_steps['scale'].mean_[0] != frozen_mean[0]


def test_pipeline_retrain_replaces_snapshot():
    pipe = Pipeline([('scale', StandardScaler()), ('tree', DecisionTreeClassifier(buffer_size=2))])
    pipe.partial_fit([[0.], [1.]], [0, 1])
    before = pipe._prediction_transformers[0][1].mean_[0]
    pipe.partial_fit([[4.], [5.]], [1, 1])
    assert pipe.n_retrains_ == 2
    assert pipe._prediction_transformers[0][1].mean_[0] != before


def test_pipeline_expanding_one_hot_works_with_frozen_prediction():
    pipe = Pipeline([('onehot', OneHotEncoder()), ('tree', DecisionTreeClassifier(buffer_size=2))])
    pipe.partial_fit([['cat'], ['dog']], [0, 1])
    original = pipe.predict([['cat'], ['dog']]).copy()
    pipe.partial_fit([['fox']], [1])
    assert pipe.named_steps['onehot'].n_features_out_ == 3
    np.testing.assert_array_equal(pipe.predict([['cat'], ['dog']]), original)
    pipe.partial_fit([['fox']], [1])
    assert pipe.n_retrains_ == 2
    assert pipe.model.n_features_in_ == 3
    assert pipe.n_buffered_samples_ == 0
    np.testing.assert_array_equal(pipe.predict([['fox']]), [1])


def test_pipeline_fit_immediate():
    pipe = Pipeline([('scale', StandardScaler()), ('tree', DecisionTreeClassifier(buffer_size=100))])
    pipe.fit([[1], [4]], [0, 1])
    assert pipe.is_fitted and pipe.n_retrains_ == 0
    np.testing.assert_array_equal(pipe.predict([[1], [4]]), [0, 1])


def test_pipeline_features_changed_rejected():
    pipe = Pipeline([('tree', DecisionTreeClassifier(buffer_size=3))])
    pipe.partial_fit([[1., 2.]], [1])
    with pytest.raises(ValueError, match='Expected 2'):
        pipe.partial_fit([[1.]], [0])


def test_pipeline_validation_y():
    pipe = Pipeline([('tree', DecisionTreeClassifier(buffer_size=2))])
    with pytest.raises(ValueError, match='same'):
        pipe.partial_fit([[1], [2]], [1])


def test_pipeline_copies_input_buffer():
    X = np.array([[1], [2]])
    pipe = Pipeline([('tree', DecisionTreeClassifier(buffer_size=4))])
    pipe.partial_fit(X, [0, 0]); X[:] = 99
    assert pipe._buffer_X[0][0, 0] == 1


def test_pipeline_pred_proba_rows_sum_one():
    pipe = Pipeline([('tree', DecisionTreeClassifier(buffer_size=2))])
    pipe.partial_fit([[1], [2]], [0, 1])
    np.testing.assert_allclose(pipe.predict_proba([[1], [2]]).sum(axis=1), 1.)


def test_trainer_skips_warmup_and_test_then_train():
    pipe = Pipeline([('tree', DecisionTreeClassifier(buffer_size=4, max_depth=0))])
    trainer = StreamTrainer(pipe, window_size=2)
    t1 = trainer.fit_chunk([[0], [1]], [0, 0])
    t2 = trainer.fit_chunk([[2], [3]], [0, 0])
    assert not t1['evaluated'] and not t2['evaluated']
    assert t2['n_retrains'] == 1
    t3 = trainer.fit_chunk([[4], [5]], [1, 1])
    assert t3['evaluated'] and t3['chunk_accuracy'] == 0.
    assert trainer.result()['support'] == 2
    assert trainer.result()['n_warmup_samples'] == 4
    assert t3['n_retrains'] == 1  # Old tree stays until next threshold.


def test_trainer_retrains_then_changes_predictions():
    model = DecisionTreeClassifier(buffer_size=2, max_depth=0)
    trainer = StreamTrainer(model)
    trainer.fit_chunk([[0], [1]], [0, 0])
    trainer.fit_chunk([[2], [3]], [1, 1])
    assert model.predict([[9]])[0] == 1
    assert trainer.result()['n_retrains'] == 2


def test_trainer_score_only_does_not_fit():
    model = DecisionTreeClassifier(buffer_size=2)
    trainer = StreamTrainer(model)
    model.fit([[0], [1]], [0, 1])
    before = trainer.result()['n_samples_seen']
    score = trainer.score_chunk([[0], [1]], [0, 1])
    assert score['evaluated'] and score['chunk_accuracy'] == 1
    assert trainer.result()['support'] == 2
    assert trainer.result()['n_samples_seen'] == before
    assert not trainer.history_


def test_trainer_logging_fields():
    trainer = StreamTrainer(Pipeline([('scale', StandardScaler()),
                                     ('tree', DecisionTreeClassifier(buffer_size=2))]))
    line = trainer.fit_chunk([[0], [1]], [0, 1])
    assert line['memory_bytes'] > 0 and line['update_ms'] >= 0
    assert set(['chunk', 'n_evaluated', 'cumulative_accuracy', 'auc', 'n_retrains', 'buffered_samples']).issubset(line)
    assert len(trainer.history_) == 1


def test_trainer_bare_tree():
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=1))
    trainer.fit_chunk([[0.]], [0])
    assert trainer.fit_chunk([[1.]], [1])['evaluated']


def test_trainer_update_alias():
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=1))
    trainer.update([[0]], [0])
    assert trainer.result()['n_chunks_seen'] == 1


def test_trainer_invalid_estimator():
    with pytest.raises(TypeError, match='Pipeline'):
        StreamTrainer(object())


def test_estimate_memory_positive():
    assert estimate_memory_bytes({'array': np.ones((10, 4))}) > 10 * 4 * 8


def test_window_size_invalid():
    for value in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            StreamingClassificationMetrics(window_size=value)
