"""Regression tests adapted from the lecturer's Week 10 Classification Trees HTML.

The teaching example has 12 students with three yes/no features:
submitted, attended, study_group.  The deliberate exception is student 5.
Tests exercise the same example using our Assignment 2.2 buffered tree,
without importing scikit-learn or other ML/data packages.
"""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier


LECTURER_ROWS = np.array(
    [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 0, 1],
        [1, 0, 1, 1],  # Deliberate exception: absent but passed
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ],
    dtype=float,
)
X_STUDENTS = LECTURER_ROWS[:, :3]
Y_PASSED = LECTURER_ROWS[:, 3].astype(int)


def test_lecturer_example_has_twelve_students_three_features():
    assert X_STUDENTS.shape == (12, 3)
    assert Y_PASSED.tolist().count(0) == 7
    assert Y_PASSED.tolist().count(1) == 5


def test_lecturer_gini_at_root_matches_hand_calculation():
    classifier = DecisionTreeClassifier(criterion="gini")
    np.testing.assert_allclose(classifier._impurity([7, 5]), 35 / 72)


def test_lecturer_entropy_at_root_matches_hand_calculation():
    classifier = DecisionTreeClassifier(criterion="entropy")
    p = np.array([7 / 12, 5 / 12])
    np.testing.assert_allclose(classifier._impurity([7, 5]), -np.sum(p * np.log2(p)))


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_lecturer_root_question_is_submitted(criterion):
    model = DecisionTreeClassifier(max_depth=2, criterion=criterion).fit(
        X_STUDENTS, Y_PASSED
    )
    assert model._tree.feature == 0  # submitted
    assert model._tree.binary_split is True
    assert model._tree.threshold == pytest.approx(0.5)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_lecturer_yes_goes_left_no_goes_right(criterion):
    model = DecisionTreeClassifier(max_depth=2, criterion=criterion).fit(
        X_STUDENTS, Y_PASSED
    )
    root = model._tree
    # The lecturer's `X[:, feature] == 1` question sends YES to left.
    assert root.left.counts.tolist() == [2, 5]
    assert root.right.counts.tolist() == [5, 0]
    assert root.left.feature == 1  # attended
    assert root.left.binary_split is True
    assert root.left.left.counts.tolist() == [0, 4]  # submitted AND attended
    assert root.left.right.counts.tolist() == [2, 1]


def test_lecturer_depth_two_correctly_classifies_eleven_of_twelve():
    model = DecisionTreeClassifier(max_depth=2).fit(X_STUDENTS, Y_PASSED)
    pred = model.predict(X_STUDENTS)
    assert np.count_nonzero(pred == Y_PASSED) == 11
    assert pred[4] == 0  # The deliberate exception is not memorised.


def test_lecturer_depth_three_can_memorise_the_exception():
    model = DecisionTreeClassifier(max_depth=3).fit(X_STUDENTS, Y_PASSED)
    np.testing.assert_array_equal(model.predict(X_STUDENTS), Y_PASSED)
    assert model._tree.left.right.feature == 2  # study_group


def test_lecturer_depth_zero_is_majority_fail():
    model = DecisionTreeClassifier(max_depth=0).fit(X_STUDENTS, Y_PASSED)
    np.testing.assert_array_equal(model.predict(X_STUDENTS), np.zeros(12, dtype=int))


def test_lecturer_tied_leaf_prefers_class_zero():
    model = DecisionTreeClassifier(max_depth=0).fit(
        [[1], [1], [0], [0]], [0, 1, 0, 1]
    )
    np.testing.assert_array_equal(model.predict([[1], [0]]), [0, 0])


def test_binary_with_missing_values_preserves_nan_right_rule():
    # Binary specialisation is only used when a node has no NaNs, so the
    # general threshold search keeps missing values on the RIGHT branch.
    X = np.array([[0.0], [0.0], [1.0], [1.0], [np.nan]])
    y = np.array([0, 0, 1, 1, 0])
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert model._tree.binary_split is False
    assert model._tree.right is not None
    assert model.predict([[np.nan]]).shape == (1,)


def test_continuous_feature_still_uses_threshold_splitting():
    X = np.array([[0.1], [0.2], [0.8], [0.9]])
    model = DecisionTreeClassifier(max_depth=1).fit(X, [0, 0, 1, 1])
    assert model._tree.binary_split is False
    assert model._tree.threshold == pytest.approx(0.5)
    np.testing.assert_array_equal(model.predict(X), [0, 0, 1, 1])


def test_lecturer_dataset_retrains_only_when_buffer_is_full():
    model = DecisionTreeClassifier(max_depth=2, buffer_size=12)
    model.partial_fit(X_STUDENTS[:4], Y_PASSED[:4])
    model.partial_fit(X_STUDENTS[4:8], Y_PASSED[4:8])
    assert model.n_buffered_samples_ == 8
    assert model.n_retrains_ == 0
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict(X_STUDENTS)
    model.partial_fit(X_STUDENTS[8:], Y_PASSED[8:])
    assert model.n_retrains_ == 1
    assert model.n_buffered_samples_ == 0
    assert np.count_nonzero(model.predict(X_STUDENTS) == Y_PASSED) == 11


def test_lecturer_binary_stream_predicts_from_latest_trained_tree():
    model = DecisionTreeClassifier(max_depth=2, buffer_size=12)
    model.partial_fit(X_STUDENTS, Y_PASSED)
    predictions_before = model.predict(X_STUDENTS).copy()
    model.partial_fit(X_STUDENTS[:5], np.zeros(5, dtype=int))
    assert model.n_buffered_samples_ == 5
    np.testing.assert_array_equal(model.predict(X_STUDENTS), predictions_before)
