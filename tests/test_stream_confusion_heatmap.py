"""Confusion matrix plotting and input validation tests."""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute_stream.visualise import plot_confusion_matrix


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close('all')


def test_confusion_matrix_annotations_and_class_order():
    matrix = np.array([[356, 62], [176, 97]])
    fig, ax = plot_confusion_matrix(matrix, labels=['0', '1'])
    assert fig is ax.figure
    assert [item.get_text() for item in ax.texts] == ['356', '62', '176', '97']
    assert [item.get_text() for item in ax.get_xticklabels()] == ['0', '1']
    assert [item.get_text() for item in ax.get_yticklabels()] == ['0', '1']
    np.testing.assert_array_equal(ax.images[0].get_array(), matrix)
    assert ax.images[0].cmap.name == 'Greys'


def test_default_labels_and_single_class():
    _, ax = plot_confusion_matrix([[7]])
    assert [item.get_text() for item in ax.get_xticklabels()] == ['0']
    assert ax.texts[0].get_text() == '7'


def test_empty_counts_allowed_without_division_by_zero():
    _, ax = plot_confusion_matrix([[0, 0], [0, 0]])
    assert len(ax.texts) == 4


def test_png_export(tmp_path):
    destination = tmp_path / 'charts' / 'confusion.png'
    plot_confusion_matrix([[5, 1], [2, 4]], labels=['no', 'yes'], save_path=destination)
    assert destination.exists() and destination.stat().st_size > 1000


def test_reuses_supplied_axes():
    fig, ax = plt.subplots()
    other_fig, other_ax = plot_confusion_matrix([[1, 2], [3, 4]], ax=ax)
    assert other_fig is fig and other_ax is ax


@pytest.mark.parametrize('matrix', [[], [1, 2], [[1, 2, 3], [4, 5, 6]], [[[1]]]])
def test_invalid_shape_rejected(matrix):
    with pytest.raises(ValueError, match='square 2D'):
        plot_confusion_matrix(matrix)


@pytest.mark.parametrize('matrix', [
    [[1, -1], [0, 1]], [[1, np.nan], [0, 1]], [[1, np.inf], [0, 1]]
])
def test_invalid_counts_rejected(matrix):
    with pytest.raises(ValueError, match='nonnegative'):
        plot_confusion_matrix(matrix)


def test_fractional_counts_rejected():
    with pytest.raises(ValueError, match='whole-number'):
        plot_confusion_matrix([[1.5, 0], [0, 1]])


def test_non_numeric_counts_rejected():
    with pytest.raises(TypeError, match='numeric'):
        plot_confusion_matrix([['X', 1], [0, 1]])


@pytest.mark.parametrize('labels', [['only-one'], [['0'], ['1']], ['a', 'b', 'c']])
def test_bad_label_count_or_shape_rejected(labels):
    with pytest.raises(ValueError, match='labels'):
        plot_confusion_matrix([[1, 0], [0, 1]], labels=labels)
