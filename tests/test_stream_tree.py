"""Regression and streaming tests for Assignment 2.2's custom tree."""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture
def training_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_fit_returns_self(training_data):
    X, y = training_data
    tree = DecisionTreeClassifier()
    assert tree.fit(X, y) is tree


def test_fit_predict_simple_separable(training_data):
    X, y = training_data
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), y)


def test_fit_entropy_predict_simple_separable(training_data):
    X, y = training_data
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=1).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), y)


def test_depth_zero_only_majority_class(training_data):
    X, y = training_data
    tree = DecisionTreeClassifier(max_depth=0).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), np.zeros(len(y), dtype=int))


def test_min_samples_split_prevents_split(training_data):
    X, y = training_data
    tree = DecisionTreeClassifier(min_samples_split=7).fit(X, y)
    assert len(np.unique(tree.predict(X))) == 1


def test_constant_feature_does_not_crash():
    X = np.ones((8, 2))
    y = np.array([0, 1] * 4)
    pred = DecisionTreeClassifier().fit(X, y).predict(X)
    np.testing.assert_array_equal(pred, np.zeros(8, dtype=int))


def test_single_class_data():
    X = np.array([[0.0], [2.0], [3.0]])
    y = np.array(["cat", "cat", "cat"])
    np.testing.assert_array_equal(DecisionTreeClassifier().fit(X, y).predict(X), y)


def test_string_labels():
    X = np.array([[0.0], [0.1], [0.9], [1.0]])
    y = np.array(["a", "a", "b", "b"])
    np.testing.assert_array_equal(DecisionTreeClassifier().fit(X, y).predict(X), y)


def test_predict_proba_normalised(training_data):
    X, y = training_data
    proba = DecisionTreeClassifier().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))
    assert proba.shape == (6, 2)


def test_predict_proba_is_consistent_with_predict(training_data):
    X, y = training_data
    tree = DecisionTreeClassifier().fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), tree.classes_[tree.predict_proba(X).argmax(axis=1)])


def test_nan_feature_routes_right_during_fit_and_predict():
    X = np.array([[0.0], [1.0], [np.nan], [np.nan]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), y)


def test_all_nan_feature_no_split():
    X = np.full((4, 1), np.nan)
    y = np.array([0, 0, 0, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), np.zeros(4, dtype=int))


def test_multiple_numeric_features():
    X = np.array([[10., 0.], [11., 0.], [10., 1.], [11., 1.]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), y)


def test_deterministic_max_features_with_fixed_seed():
    X = np.random.default_rng(32).normal(size=(100, 6))
    y = (X[:, 1] + X[:, 4] > 0).astype(int)
    a = DecisionTreeClassifier(max_features="sqrt", random_state=11).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=11).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_max_features_fraction(training_data):
    X, y = training_data
    tree = DecisionTreeClassifier(max_features=0.6).fit(X, y)
    assert tree.predict(X).shape == (6,)


def test_fit_with_empty_data_fails():
    with pytest.raises(ValueError, match="at least one"):
        DecisionTreeClassifier().fit(np.empty((0, 1)), np.empty(0))


def test_bad_x_dimensionality_fails():
    with pytest.raises(ValueError, match="2D"):
        DecisionTreeClassifier().fit([1, 2], [0, 1])


def test_bad_y_dimensionality_fails():
    with pytest.raises(ValueError, match="1D"):
        DecisionTreeClassifier().fit([[1], [2]], [[0], [1]])


def test_mismatched_xy_lengths_fails():
    with pytest.raises(ValueError, match="same number"):
        DecisionTreeClassifier().fit([[1], [2]], [0])


def test_missing_y_fails():
    with pytest.raises(ValueError, match="missing labels"):
        DecisionTreeClassifier().fit([[1], [2]], [0, np.nan])


def test_infinite_x_fails():
    with pytest.raises(ValueError, match="infinity"):
        DecisionTreeClassifier().fit([[1], [np.inf]], [0, 1])


def test_bad_x_non_numeric_fails():
    with pytest.raises(TypeError, match="numeric"):
        DecisionTreeClassifier().fit([["oops"], ["bad"]], [0, 1])


def test_predict_before_fit_fails():
    with pytest.raises(RuntimeError, match="not trained"):
        DecisionTreeClassifier().predict([[0]])


def test_predict_wrong_feature_count_fails(training_data):
    X, y = training_data
    tree = DecisionTreeClassifier().fit(X, y)
    with pytest.raises(ValueError, match="Expected 1 features"):
        tree.predict([[0, 0]])


def test_partial_fit_buffers_without_training(training_data):
    X, y = training_data
    tree = DecisionTreeClassifier(buffer_size=6)
    tree.partial_fit(X[:3], y[:3])
    assert tree.n_buffered_samples_ == 3
    assert tree.n_retrains_ == 0
    with pytest.raises(RuntimeError, match="not trained"):
        tree.predict(X)


def test_partial_fit_trains_exactly_at_threshold(training_data):
    X, y = training_data
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=6)
    tree.partial_fit(X[:3], y[:3])
    tree.partial_fit(X[3:], y[3:])
    assert tree.n_retrains_ == 1
    assert tree.n_buffered_samples_ == 0
    assert tree.n_samples_seen_ == 6
    np.testing.assert_array_equal(tree.predict(X), y)


def test_partial_fit_oversized_chunk_trains_once(training_data):
    X, y = training_data
    tree = DecisionTreeClassifier(buffer_size=4).partial_fit(X, y)
    assert tree.n_retrains_ == 1
    assert tree.n_buffered_samples_ == 0


def test_partial_fit_keeps_old_tree_until_next_retrain():
    X = np.array([[0.], [1.], [2.], [3.]])
    tree = DecisionTreeClassifier(buffer_size=4, max_depth=0)
    tree.partial_fit(X, np.zeros(4, dtype=int))
    tree.partial_fit(X[:2], np.ones(2, dtype=int))
    np.testing.assert_array_equal(tree.predict(X), np.zeros(4, dtype=int))
    assert tree.n_buffered_samples_ == 2
    tree.partial_fit(X[2:], np.ones(2, dtype=int))
    np.testing.assert_array_equal(tree.predict(X), np.ones(4, dtype=int))
    assert tree.n_retrains_ == 2


def test_partial_fit_buffer_copies_mutable_input():
    X = np.array([[0.], [1.]])
    y = np.array([0, 0])
    tree = DecisionTreeClassifier(buffer_size=4, max_depth=0)
    tree.partial_fit(X, y)
    X[:] = 99
    y[:] = 1
    tree.partial_fit(np.array([[2.], [3.]]), np.array([0, 0]))
    np.testing.assert_array_equal(tree.predict([[0]]), [0])


def test_partial_fit_rejects_feature_dimension_change():
    tree = DecisionTreeClassifier(buffer_size=5)
    tree.partial_fit([[0., 1.]], [0])
    with pytest.raises(ValueError, match="Expected 2 features"):
        tree.partial_fit([[1.]], [1])
    assert tree.n_buffered_samples_ == 1


def test_fit_clears_preexisting_buffer():
    tree = DecisionTreeClassifier(buffer_size=5)
    tree.partial_fit([[0]], [1])
    tree.fit([[0], [1]], [0, 1])
    assert tree.n_buffered_samples_ == 0


def test_empty_prediction_shape(training_data):
    X, y = training_data
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.predict(np.empty((0, 1))).shape == (0,)
    assert tree.predict_proba(np.empty((0, 1))).shape == (0, 2)


@pytest.mark.parametrize("kwargs", [
    {"buffer_size": 0}, {"buffer_size": True}, {"min_samples_split": 1},
    {"max_depth": -1}, {"max_depth": 1.5}, {"max_features": 0},
    {"max_features": 1.2}, {"max_features": "invalid"},
    {"random_state": -1}, {"criterion": "mse"},
])
def test_invalid_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**kwargs)


def test_feature_count_restriction_fails():
    with pytest.raises(ValueError, match="max_features"):
        DecisionTreeClassifier(max_features=3).fit([[0, 1], [1, 2]], [0, 1])


def test_three_class_problem():
    X = np.arange(9, dtype=float).reshape(-1, 1)
    y = np.repeat(np.array(["low", "mid", "high"]), 3)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=2).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), y)
    assert tree.predict_proba(X).shape == (9, 3)


def test_large_magnitude_feature_no_overflow_warning():
    X = np.array([[-1e308], [-8e307], [8e307], [1e308]])
    y = np.array([0, 0, 1, 1])
    with np.errstate(over="raise", invalid="raise"):
        pred = DecisionTreeClassifier(max_depth=1).fit(X, y).predict(X)
    np.testing.assert_array_equal(pred, y)
