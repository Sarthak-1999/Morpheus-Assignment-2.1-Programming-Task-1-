"""Reusable matplotlib plots for NumCompute's streaming classification logs.

This module depends on matplotlib and NumPy, not pandas or an ML library.
All plotting functions return the ``(figure, axes)`` they create so callers
can customise or display them in VS Code, notebooks, or ordinary scripts.
``save_path`` writes a PNG (or any format matplotlib supports), whereas
``show=True`` requests on-screen display.  Neither happens by default.

Missing metrics (NaN) are plotted as gaps: this is intentional for warm-up
chunks, before the decision tree has received enough samples to train.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _metric_values(values, name: str) -> np.ndarray:
    """Validate a nonempty numerical history; NaN is allowed, infinity is not."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f'{name} must contain numeric metric values.') from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f'{name} must be a nonempty 1D sequence.')
    if np.isinf(array).any():
        raise ValueError(f'{name} must not contain infinite values.')
    return array


def _chunk_axis(chunks, size: int) -> np.ndarray:
    if chunks is None:
        return np.arange(1, size + 1)
    try:
        index = np.asarray(chunks, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError('chunk_indices must be numeric.') from exc
    if index.ndim != 1 or len(index) != size:
        raise ValueError('chunk_indices must be 1D and match metric length.')
    if not np.isfinite(index).all() or np.any(np.diff(index) <= 0):
        raise ValueError('chunk_indices must be finite and strictly increasing.')
    return index


def _axes(ax: Axes | None, *, figsize=(8, 4.5)) -> tuple[Figure, Axes]:
    if ax is None:
        return plt.subplots(figsize=figsize, layout='constrained')
    if not isinstance(ax, Axes):
        raise TypeError('ax must be a matplotlib Axes instance or None.')
    return ax.figure, ax


def _finish(fig: Figure, *, save_path=None, show: bool = False) -> None:
    if save_path is not None:
        target = Path(save_path)
        if not target.suffix:
            raise ValueError('save_path must have a filename extension (e.g. .png).')
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=160, bbox_inches='tight')
    if show:
        plt.show()


def plot_metric_over_time(metric_values, title: str, ylabel: str, *,
                          chunk_indices=None, ax: Axes | None = None,
                          save_path=None, show: bool = False) -> tuple[Figure, Axes]:
    """Plot one metric per arriving chunk (axis=0 is chronological chunks).

    NaN values are left as gaps; e.g. the model cannot predict during warm-up.
    Optional ``chunk_indices`` must be strictly increasing.  A pre-existing
    ``ax`` can be supplied for use in a larger dashboard.
    """
    values = _metric_values(metric_values, 'metric_values')
    x = _chunk_axis(chunk_indices, len(values))
    fig, ax = _axes(ax)
    ax.plot(x, values, marker='o', linewidth=1.8)
    ax.set(title=str(title), xlabel='Stream chunk', ylabel=str(ylabel))
    ax.grid(True, alpha=0.25)
    _finish(fig, save_path=save_path, show=show)
    return fig, ax


def compare_models(metric1, metric2, labels, *, chunk_indices=None,
                   title: str = 'Streaming model comparison',
                   ylabel: str = 'Metric', ax: Axes | None = None,
                   save_path=None, show: bool = False) -> tuple[Figure, Axes]:
    """Compare two aligned per-chunk metrics (e.g. different buffer sizes).

    For a fair accuracy comparison, supply values computed on the *same*
    evaluated samples; the two trainers may have different warm-up horizons.
    """
    first = _metric_values(metric1, 'metric1')
    second = _metric_values(metric2, 'metric2')
    if first.shape != second.shape:
        raise ValueError('metric1 and metric2 must contain the same number of chunks.')
    if not isinstance(labels, (list, tuple)) or len(labels) != 2 or not all(
        isinstance(label, str) and label.strip() for label in labels
    ):
        raise ValueError('labels must be two nonempty model names.')
    x = _chunk_axis(chunk_indices, len(first))
    fig, ax = _axes(ax)
    ax.plot(x, first, marker='o', linewidth=1.8, label=labels[0])
    ax.plot(x, second, marker='s', linewidth=1.8, label=labels[1])
    ax.set(title=str(title), xlabel='Stream chunk', ylabel=str(ylabel))
    ax.legend()
    ax.grid(True, alpha=0.25)
    _finish(fig, save_path=save_path, show=show)
    return fig, ax


def plot_predictions_vs_ground_truth(y_true, y_pred, *,
                                     title: str = 'Predictions vs ground truth',
                                     ax: Axes | None = None, save_path=None,
                                     show: bool = False) -> tuple[Figure, Axes]:
    """Visualise per-sample actual/predicted labels for the latest scored chunk.

    Accepts 1D classification labels, including strings.  Samples appear on
    the x axis; actual and predicted classes are on a categorical y axis.
    'x' markers identify prediction errors in the current chunk.
    """
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if actual.ndim != 1 or predicted.ndim != 1 or not len(actual):
        raise ValueError('y_true and y_pred must be nonempty 1D label arrays.')
    if len(actual) != len(predicted):
        raise ValueError('y_true and y_pred must have equal lengths.')
    if np.any(actual != actual) or np.any(predicted != predicted):
        raise ValueError('Classification labels cannot contain NaNs.')
    try:
        labels = np.unique(np.concatenate([actual, predicted]))
        true_index = np.searchsorted(labels, actual)
        pred_index = np.searchsorted(labels, predicted)
    except (TypeError, ValueError) as exc:
        raise ValueError('Classification labels must be mutually comparable.') from exc
    errors = actual != predicted
    x = np.arange(1, len(actual) + 1)
    fig, ax = _axes(ax, figsize=(9, 4.5))
    ax.scatter(x, true_index - 0.08, marker='o', s=40, alpha=0.85, label='Actual')
    ax.scatter(x, pred_index + 0.08, marker='_', s=110, alpha=0.9,
               label='Predicted')
    if errors.any():
        ax.scatter(x[errors], pred_index[errors] + 0.08, marker='x',
                   s=50, label=f'Errors ({int(errors.sum())})')
    ax.set(title=f'{title} — {len(actual)} samples, {int(errors.sum())} errors',
           xlabel='Sample number within latest evaluated chunk', ylabel='Class')
    ax.set_yticks(np.arange(len(labels)), [str(label) for label in labels])
    ax.set_ylim(-0.55, len(labels) - 0.45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2, axis='y')
    _finish(fig, save_path=save_path, show=show)
    return fig, ax



def plot_confusion_matrix(confusion_matrix, labels=None, *,
                          title: str = 'Cumulative confusion matrix',
                          ax: Axes | None = None, save_path=None,
                          show: bool = False) -> tuple[Figure, Axes]:
    """Display accumulated classification counts as a grayscale heatmap.

    Matrix rows are true classes and columns are predicted classes.
    ``labels`` must follow the same class order used to build the matrix.
    The colour scale shows counts, not normalised percentages.
    """
    try:
        counts = np.asarray(confusion_matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError('confusion_matrix must contain numeric counts.') from exc
    if counts.ndim != 2 or not counts.size or counts.shape[0] != counts.shape[1]:
        raise ValueError('confusion_matrix must be a nonempty square 2D matrix.')
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('confusion_matrix must contain finite nonnegative counts.')
    if not np.equal(counts, np.floor(counts)).all():
        raise ValueError('confusion_matrix must contain whole-number counts.')

    n_classes = counts.shape[0]
    if labels is None:
        class_names = [str(i) for i in range(n_classes)]
    else:
        values = np.asarray(labels, dtype=object)
        if values.ndim != 1 or values.size != n_classes:
            raise ValueError('labels must be 1D and match the confusion matrix size.')
        class_names = [str(value) for value in values]

    fig, ax = _axes(ax, figsize=(6, 5))
    colour_max = max(1.0, float(counts.max()))
    image = ax.imshow(counts, cmap='Greys', vmin=0, vmax=colour_max,
                      interpolation='nearest')
    for row, col in np.ndindex(counts.shape):
        count = int(counts[row, col])
        ink = 'white' if counts[row, col] > colour_max * 0.55 else 'black'
        ax.text(col, row, str(count), ha='center', va='center',
                color=ink, fontsize=12, fontweight='semibold')

    positions = np.arange(n_classes)
    ax.set_xticks(positions, class_names)
    ax.set_yticks(positions, class_names)
    ax.set(xlabel='Predicted class', ylabel='Actual class', title=title)
    ax.tick_params(axis='both', length=0)
    fig.colorbar(image, ax=ax, fraction=0.047, pad=0.04, label='Sample count')
    _finish(fig, save_path=save_path, show=show)
    return fig, ax

def _history_values(history: Sequence[Mapping], key: str) -> np.ndarray:
    if not all(isinstance(row, Mapping) and key in row for row in history):
        raise ValueError(f'Every history row must contain the {key!r} metric.')
    return _metric_values([row[key] for row in history], key)


def plot_stream_dashboard(history: Sequence[Mapping], *,
                          title: str = 'Streaming classification dashboard',
                          save_path=None, show: bool = False) -> tuple[Figure, np.ndarray]:
    """Summarise ``StreamTrainer.history_`` in four reusable panels.

    Plots chunk/cumulative/rolling accuracy, error, AUC, and training time.
    Retrain events appear as vertical dashed lines on the accuracy panel.
    """
    if not history:
        raise ValueError('history must contain at least one chunk log.')
    chunks = _history_values(history, 'chunk')
    names = ('chunk_accuracy', 'cumulative_accuracy', 'rolling_accuracy',
             'chunk_error', 'cumulative_error', 'auc', 'update_ms', 'n_retrains')
    series = {name: _history_values(history, name) for name in names}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    ax_acc, ax_err, ax_auc, ax_time = axes.flat
    for name, label in [('chunk_accuracy', 'Chunk'),
                        ('cumulative_accuracy', 'Cumulative'),
                        ('rolling_accuracy', 'Rolling')]:
        ax_acc.plot(chunks, series[name], marker='o', label=label)
    retrained = np.flatnonzero(np.diff(np.r_[0, series['n_retrains']]) > 0)
    for idx in retrained:
        ax_acc.axvline(chunks[idx], alpha=0.25, linestyle='--')
    ax_acc.set(title='Accuracy (dashed = retrain)', ylabel='Accuracy')
    ax_acc.legend(loc='best')
    for name, label in [('chunk_error', 'Chunk'), ('cumulative_error', 'Cumulative')]:
        ax_err.plot(chunks, series[name], marker='o', label=label)
    ax_err.set(title='Prediction error', ylabel='Error rate')
    ax_err.legend(loc='best')
    ax_auc.plot(chunks, series['auc'], marker='o')
    ax_auc.set(title='Cumulative ROC AUC', ylabel='AUC')
    ax_time.bar(chunks, series['update_ms'], width=0.65)
    ax_time.set(title='Per-chunk fitting time', ylabel='Milliseconds')
    for ax in axes.flat:
        ax.set_xlabel('Stream chunk')
        ax.grid(True, alpha=0.2)
    fig.suptitle(title, fontsize=14)
    _finish(fig, save_path=save_path, show=show)
    return fig, axes


class LiveMetricPlot:
    """Incrementally update accuracy lines during ``StreamTrainer.fit_chunk``.

    Use ``live.update(row)`` once per chunk.  With a GUI/interactive matplotlib
    backend, ``pause > 0`` redraws the figure immediately; in VS Code's inline
    notebook backend, the final plot appears on completing the cell.
    """

    def __init__(self, *, title='Live streaming accuracy', pause: float = 0):
        if not np.isscalar(pause) or not np.isfinite(pause) or pause < 0:
            raise ValueError('pause must be a nonnegative finite number.')
        self.pause = float(pause)
        self.fig, self.ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        self.ax.set(title=title, xlabel='Stream chunk', ylabel='Accuracy')
        self.ax.grid(True, alpha=0.25)
        self._chunk = []
        self._acc = []
        self._cumulative = []
        (self._line_chunk,) = self.ax.plot([], [], marker='o', label='Chunk accuracy')
        (self._line_cumulative,) = self.ax.plot([], [], marker='s', label='Cumulative accuracy')
        self.ax.legend()

    def update(self, row: Mapping) -> 'LiveMetricPlot':
        if not isinstance(row, Mapping) or not all(k in row for k in
            ('chunk', 'chunk_accuracy', 'cumulative_accuracy')):
            raise ValueError('row must contain chunk, chunk_accuracy and cumulative_accuracy.')
        current = _metric_values([row['chunk_accuracy'], row['cumulative_accuracy']], 'accuracy')
        chunk = float(row['chunk'])
        if not np.isfinite(chunk) or (self._chunk and chunk <= self._chunk[-1]):
            raise ValueError('chunk numbers must be finite and strictly increasing.')
        self._chunk.append(chunk)
        self._acc.append(current[0])
        self._cumulative.append(current[1])
        self._line_chunk.set_data(self._chunk, self._acc)
        self._line_cumulative.set_data(self._chunk, self._cumulative)
        self.ax.set_xlim(min(self._chunk) - 0.5, max(self._chunk) + 0.5)
        finite = np.asarray(self._acc + self._cumulative)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            ymin, ymax = max(0, float(finite.min()) - 0.1), min(1, float(finite.max()) + 0.1)
            if ymin == ymax:
                ymax += 0.05
            self.ax.set_ylim(ymin, ymax)
        self.fig.canvas.draw_idle()
        if self.pause > 0:
            plt.pause(self.pause)
        return self

    def save(self, path) -> 'LiveMetricPlot':
        _finish(self.fig, save_path=path)
        return self


__all__ = [
    'plot_metric_over_time', 'compare_models', 'plot_predictions_vs_ground_truth',
    'plot_stream_dashboard', 'plot_confusion_matrix', 'LiveMetricPlot',
]
