"""Step 5: reusable, headless matplotlib stream visualisation tests."""
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute_stream.visualise import (
    LiveMetricPlot, compare_models, plot_metric_over_time,
    plot_predictions_vs_ground_truth, plot_stream_dashboard,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def sample_history():
    return [
        dict(chunk=1, chunk_accuracy=np.nan, cumulative_accuracy=np.nan,
             rolling_accuracy=np.nan, chunk_error=np.nan,
             cumulative_error=np.nan, auc=np.nan, update_ms=11., n_retrains=0),
        dict(chunk=2, chunk_accuracy=np.nan, cumulative_accuracy=np.nan,
             rolling_accuracy=np.nan, chunk_error=np.nan,
             cumulative_error=np.nan, auc=np.nan, update_ms=20., n_retrains=1),
        dict(chunk=3, chunk_accuracy=.7, cumulative_accuracy=.7,
             rolling_accuracy=.7, chunk_error=.3, cumulative_error=.3,
             auc=.6, update_ms=.4, n_retrains=1),
        dict(chunk=4, chunk_accuracy=.8, cumulative_accuracy=.75,
             rolling_accuracy=.75, chunk_error=.2, cumulative_error=.25,
             auc=.65, update_ms=15., n_retrains=2),
    ]


def test_metric_returns_figure_and_axis():
    fig, ax = plot_metric_over_time([.5, .75, .9], 'Accuracy', 'Accuracy')
    assert fig is ax.figure and ax.get_title() == 'Accuracy'
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3])


def test_metric_accepts_nan_warmup():
    _, ax = plot_metric_over_time([np.nan, np.nan, .6], 'Accuracy', 'Accuracy')
    assert np.isnan(ax.lines[0].get_ydata()[0])


def test_metric_custom_chunk_numbers():
    _, ax = plot_metric_over_time([.4, .5], 'AUC', 'AUC', chunk_indices=[3, 5])
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [3., 5.])


def test_metric_injected_axes_reused():
    fig, ax = plt.subplots()
    fig2, ax2 = plot_metric_over_time([.5], 'A', 'B', ax=ax)
    assert fig is fig2 and ax is ax2


def test_metric_saves_png(tmp_path):
    target = tmp_path / 'generated' / 'metric.png'
    plot_metric_over_time([.4], 'A', 'B', save_path=target)
    assert target.is_file() and target.stat().st_size > 1000


def test_metric_empty_rejected():
    with pytest.raises(ValueError, match='nonempty'):
        plot_metric_over_time([], 'A', 'B')


def test_metric_non_numeric_rejected():
    with pytest.raises(TypeError, match='numeric'):
        plot_metric_over_time(['a', 'b'], 'A', 'B')


def test_metric_infinity_rejected():
    with pytest.raises(ValueError, match='infinite'):
        plot_metric_over_time([.2, np.inf], 'A', 'B')


def test_metric_multidimensional_rejected():
    with pytest.raises(ValueError, match='1D'):
        plot_metric_over_time([[.2, .3]], 'A', 'B')


def test_metric_custom_index_length_rejected():
    with pytest.raises(ValueError, match='match'):
        plot_metric_over_time([.2, .3], 'A', 'B', chunk_indices=[1])


def test_metric_duplicate_index_rejected():
    with pytest.raises(ValueError, match='increasing'):
        plot_metric_over_time([.2, .3], 'A', 'B', chunk_indices=[3, 3])


def test_metric_invalid_axis_rejected():
    with pytest.raises(TypeError, match='Axes'):
        plot_metric_over_time([.1], 'A', 'B', ax=123)


def test_metric_save_path_extension_required(tmp_path):
    with pytest.raises(ValueError, match='extension'):
        plot_metric_over_time([.1], 'A', 'B', save_path=tmp_path / 'no_extension')


def test_compare_models_two_lines_and_legend():
    _, ax = compare_models([.6, .7], [.5, .8], ['A', 'B'])
    assert len(ax.lines) == 2
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ['A', 'B']


def test_compare_models_missing_scores():
    _, ax = compare_models([np.nan, .7], [np.nan, .6], ['small', 'large'])
    assert np.isnan(ax.lines[0].get_ydata()[0])


def test_compare_models_lengths_must_match():
    with pytest.raises(ValueError, match='same number'):
        compare_models([.6], [.5, .7], ['A', 'B'])


@pytest.mark.parametrize('labels', [['only one'], [], ['A', ''], ('A', 5), 'AB'])
def test_compare_models_labels_validation(labels):
    with pytest.raises(ValueError, match='labels'):
        compare_models([.4], [.5], labels)


def test_compare_models_custom_x_and_save(tmp_path):
    path = tmp_path / 'comparison.png'
    _, ax = compare_models([.5, .6], [.4, .7], ['A', 'B'], chunk_indices=[3, 4], save_path=path)
    assert path.exists()
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [3., 4.])


def test_predictions_with_string_classes():
    fig, ax = plot_predictions_vs_ground_truth(['no', 'yes', 'no'], ['yes', 'yes', 'no'])
    assert fig is ax.figure and '1 errors' in ax.get_title()
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ['no', 'yes']


def test_predictions_with_number_classes():
    _, ax = plot_predictions_vs_ground_truth([0, 1, 1], [0, 0, 1])
    assert '1 errors' in ax.get_title()


def test_predictions_perfect():
    _, ax = plot_predictions_vs_ground_truth([0, 1], [0, 1])
    assert '0 errors' in ax.get_title()


def test_predictions_empty_rejected():
    with pytest.raises(ValueError, match='nonempty'):
        plot_predictions_vs_ground_truth([], [])


def test_predictions_wrong_dims_rejected():
    with pytest.raises(ValueError, match='1D'):
        plot_predictions_vs_ground_truth([[1, 0]], [1, 0])


def test_predictions_wrong_length_rejected():
    with pytest.raises(ValueError, match='equal lengths'):
        plot_predictions_vs_ground_truth([0, 1], [0])


def test_predictions_nan_rejected():
    with pytest.raises(ValueError, match='NaNs'):
        plot_predictions_vs_ground_truth([0, np.nan], [0, 1])


def test_predictions_png_export(tmp_path):
    path = tmp_path / 'predictions.png'
    plot_predictions_vs_ground_truth(['0', '1'], ['1', '1'], save_path=path)
    assert path.exists() and path.stat().st_size > 1000


def test_dashboard_accepts_warmup_and_retraining():
    fig, axes = plot_stream_dashboard(sample_history())
    assert fig is axes[0, 0].figure and axes.shape == (2, 2)
    assert 'dashed' in axes[0, 0].get_title()
    assert len(axes[0, 0].lines) >= 3


def test_dashboard_requires_rows():
    with pytest.raises(ValueError, match='at least'):
        plot_stream_dashboard([])


def test_dashboard_missing_metric_rejected():
    with pytest.raises(ValueError, match='metric'):
        plot_stream_dashboard([{'chunk': 1}])


def test_dashboard_saves_png(tmp_path):
    target = tmp_path / 'dashboard.png'
    plot_stream_dashboard(sample_history(), save_path=target)
    assert target.exists() and target.stat().st_size > 1000


def test_live_plot_updates_pre_warmup_and_after():
    plot = LiveMetricPlot()
    assert plot.update(sample_history()[0]) is plot
    plot.update(sample_history()[1])
    plot.update(sample_history()[2])
    assert len(plot._chunk) == 3
    assert np.isnan(plot._acc[0]) and plot._acc[2] == .7


def test_live_plot_saves(tmp_path):
    live = LiveMetricPlot().update(sample_history()[2])
    live.save(tmp_path / 'live.png')
    assert (tmp_path / 'live.png').exists()


def test_live_plot_invalid_pause():
    for value in [-.1, np.inf]:
        with pytest.raises(ValueError, match='pause'):
            LiveMetricPlot(pause=value)


def test_live_plot_missing_field_rejected():
    with pytest.raises(ValueError, match='row'):
        LiveMetricPlot().update({'chunk': 1})


def test_live_plot_repeated_chunk_rejected():
    live = LiveMetricPlot()
    live.update(sample_history()[2])
    with pytest.raises(ValueError, match='increasing'):
        live.update(sample_history()[2])
