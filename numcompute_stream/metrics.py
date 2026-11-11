"""Incremental, NumPy-based classification evaluation for streaming data.

Cumulative counts require memory proportional to the number of classes.
Exact ROC AUC stores score/label pairs: their memory grows with evaluated
observations. Rolling statistics retain only the most recent ``window_size``
label pairs. Undefined ratios and AUC are reported as NaN (never divided by
zero silently); precision/recall/F1 use zero_division=0 by convention.
"""

from __future__ import annotations

from collections import deque
import numpy as np
import numpy.typing as npt


def _pair(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if truth.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true_chunk and y_pred_chunk must both be 1D.")
    if not len(truth) or len(truth) != len(predicted):
        raise ValueError("Nonempty y_true_chunk and y_pred_chunk must have equal lengths.")
    for field in (truth, predicted):
        if field.dtype.kind in 'fc' and not np.isfinite(field).all():
            raise ValueError("Labels must not contain missing (NaN) or infinite values.")
        if any(v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)) for v in field):
            raise ValueError("Labels must not contain missing values.")
    try:
        np.unique(np.concatenate((truth, predicted)))
    except (TypeError, ValueError) as exc:
        raise TypeError("Classification labels must be mutually comparable.") from exc
    return truth.copy(), predicted.copy()


def _counts(truth: np.ndarray, predicted: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # Fixed labels may be intentionally unsorted: map back to their supplied order.
    sorter = np.argsort(labels)
    y_idx = sorter[np.searchsorted(labels, truth, sorter=sorter)]
    p_idx = sorter[np.searchsorted(labels, predicted, sorter=sorter)]
    n = len(labels)
    return np.bincount(n * y_idx + p_idx, minlength=n * n).reshape(n, n)


def _stats(matrix: np.ndarray, labels: np.ndarray, positive_label: object | None) -> dict:
    n = int(matrix.sum())
    correct = int(np.trace(matrix))
    accuracy = correct / n if n else float('nan')
    if len(labels) == 0:
        return dict(accuracy=accuracy, error=float('nan'), precision=float('nan'),
                    recall=float('nan'), f1=float('nan'), support=0)
    tp = matrix.diagonal().astype(float)
    predicted_count = matrix.sum(axis=0)
    real_count = matrix.sum(axis=1)
    precision = np.divide(tp, predicted_count, out=np.zeros(len(labels)), where=predicted_count != 0)
    recall = np.divide(tp, real_count, out=np.zeros(len(labels)), where=real_count != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(labels)), where=precision + recall != 0)
    if positive_label is not None:
        selected = np.flatnonzero(labels == positive_label)
        if not len(selected):
            # Before positive class appears, its positive-specific scores are zero.
            pos, rec, score = 0.0, 0.0, 0.0
        else:
            i = int(selected[0]); pos, rec, score = float(precision[i]), float(recall[i]), float(f1[i])
    elif len(labels) == 2:
        pos, rec, score = float(precision[-1]), float(recall[-1]), float(f1[-1])
    else:
        # Macro-average for multiclass, and the only observed class when unary.
        pos, rec, score = float(precision.mean()), float(recall.mean()), float(f1.mean())
    return dict(accuracy=accuracy, error=1-accuracy if n else float('nan'),
                precision=pos, recall=rec, f1=score, support=n)


def binary_auc(y_true: npt.ArrayLike, scores: npt.ArrayLike, *, positive_label: object = 1) -> float:
    """Mann–Whitney ROC AUC with average ranks for ties; NaN when undefined.

    ``scores`` must be finite positive-class confidence values, not predicted
    class labels. Only plain Python and NumPy are used. One-class data gives NaN.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or not len(y) or len(y) != len(s):
        raise ValueError("y_true and scores must be equal-length, nonempty 1D arrays.")
    if not np.isfinite(s).all():
        raise ValueError("AUC scores must be finite.")
    if ((y.dtype.kind in 'fc' and not np.isfinite(y).all())
            or any(v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)) for v in y)):
        raise ValueError('AUC labels must not contain missing or infinite values.')
    if len(np.unique(y)) > 2:
        return float('nan')
    yes = y == positive_label
    n_positive = int(yes.sum()); n_negative = len(y) - n_positive
    if not n_positive or not n_negative:
        return float('nan')
    # Group equal scores, awarding half credit for a tied positive-negative pair.
    order = np.argsort(s, kind='stable')
    sorted_scores = s[order]
    sorted_positive = yes[order].astype(int)
    _, starts, sizes = np.unique(sorted_scores, return_index=True, return_counts=True)
    successes = np.add.reduceat(sorted_positive, starts)
    average_ranks = starts + (sizes + 1) / 2.0  # 1-based average ranks
    return float((np.dot(successes, average_ranks) - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


class StreamingClassificationMetrics:
    """Cumulative confusion matrix, accuracy, precision, recall, F1, AUC.

    Parameters
    ----------
    labels : sequence, optional
        Fixed label vocabulary in this order. If absent, discover new labels
        from each chunk and keep a sorted vocabulary. Useful for multiclass.
    positive_label : object, optional
        Class used for binary precision/recall/F1/AUC; defaults to the highest
        sorted observed class when exactly two classes have appeared.
    window_size : positive int, optional
        Number of latest evaluated samples for rolling classification metrics.

    ``update(y_true_chunk, y_pred_chunk, y_score_chunk=None)`` supports scalar
    confidence scores for the selected positive class. For exact AUC, only
    samples supplied with confidence scores are used; unscored chunks are not
    silently treated as having probability 0 or 1.
    """

    def __init__(self, labels=None, *, positive_label=None, window_size: int | None = None):
        if window_size is not None and (isinstance(window_size, (bool, np.bool_))
                                        or not isinstance(window_size, (int, np.integer))
                                        or window_size < 1):
            raise ValueError('window_size must be a positive integer or None.')
        if labels is not None:
            labels = np.asarray(labels)
            if labels.ndim != 1 or not len(labels) or len(np.unique(labels)) != len(labels):
                raise ValueError('labels must be a nonempty unique 1D sequence.')
        self._fixed_labels = None if labels is None else np.asarray(labels).copy()
        self.positive_label = positive_label
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.labels_ = (np.asarray([], dtype=object) if self._fixed_labels is None
                        else self._fixed_labels.copy())
        self.matrix_ = np.zeros((len(self.labels_), len(self.labels_)), dtype=np.int64)
        self.n_samples_seen_ = 0
        self._rolling = deque()
        self._rolling_count = 0
        self._score_history: list[tuple[np.ndarray, np.ndarray]] = []
        return self

    def update(self, y_true_chunk, y_pred_chunk, y_score_chunk=None):
        truth, predictions = _pair(y_true_chunk, y_pred_chunk)
        if self._fixed_labels is None:
            new_labels = np.unique(np.concatenate((truth, predictions, self.labels_)))
            if len(new_labels) != len(self.labels_) or not np.array_equal(new_labels, self.labels_):
                updated = np.zeros((len(new_labels), len(new_labels)), dtype=np.int64)
                if len(self.labels_):
                    idx = np.searchsorted(new_labels, self.labels_)
                    updated[np.ix_(idx, idx)] = self.matrix_
                self.matrix_, self.labels_ = updated, new_labels
        elif not (np.isin(truth, self.labels_).all() and np.isin(predictions, self.labels_).all()):
            raise ValueError('Unknown classification label; not in fixed labels.')
        scores = None
        if y_score_chunk is not None:
            scores = np.asarray(y_score_chunk, dtype=float)
            if scores.ndim != 1 or len(scores) != len(truth) or not np.isfinite(scores).all():
                raise ValueError('y_score_chunk must be finite 1D scores matching y_true_chunk.')
        self.matrix_ += _counts(truth, predictions, self.labels_)
        self.n_samples_seen_ += len(truth)
        if scores is not None:
            self._score_history.append((truth.copy(), scores.copy()))
        if self.window_size is not None:
            self._rolling.append((truth.copy(), predictions.copy()))
            self._rolling_count += len(truth)
            while self._rolling_count > self.window_size:
                remove = self._rolling_count - self.window_size
                t, p = self._rolling[0]
                if len(t) <= remove:
                    self._rolling.popleft(); self._rolling_count -= len(t)
                else:
                    self._rolling[0] = (t[remove:].copy(), p[remove:].copy())
                    self._rolling_count -= remove
        return self

    def result(self) -> dict:
        result = _stats(self.matrix_, self.labels_, self.positive_label)
        result['labels'] = self.labels_.copy()
        result['confusion_matrix'] = self.matrix_.copy()
        positive = self.positive_label
        if positive is None and len(self.labels_) == 2:
            positive = self.labels_[-1]
        result['auc'] = float('nan')
        if positive is not None and len(self.labels_) <= 2 and self._score_history:
            truth = np.concatenate([a for a, _ in self._score_history])
            scores = np.concatenate([b for _, b in self._score_history])
            result['auc'] = binary_auc(truth, scores, positive_label=positive)
        if self.window_size is not None:
            if self._rolling:
                t = np.concatenate([a for a, _ in self._rolling])
                p = np.concatenate([b for _, b in self._rolling])
                rolling = _stats(_counts(t, p, self.labels_), self.labels_, self.positive_label)
            else:
                rolling = _stats(np.zeros_like(self.matrix_), self.labels_, self.positive_label)
            result.update({f'rolling_{k}': v for k, v in rolling.items()})
        return result


class ConfusionMatrix:
    """Incremental confusion matrix: rows=true, columns=predicted."""
    def __init__(self, labels=None):
        self._metrics = StreamingClassificationMetrics(labels=labels)
    @property
    def labels_(self):
        return self._metrics.labels_.copy()
    def update(self, y_true_chunk, y_pred_chunk):
        self._metrics.update(y_true_chunk, y_pred_chunk)
        return self
    def reset(self):
        self._metrics.reset()
        return self
    def result(self):
        return self._metrics.result()['confusion_matrix']


class _MetricView:
    metric_name = 'accuracy'
    def __init__(self, *, positive_label=None, window_size=None):
        self._metrics = StreamingClassificationMetrics(positive_label=positive_label, window_size=window_size)
    def update(self, y_true_chunk, y_pred_chunk):
        self._metrics.update(y_true_chunk, y_pred_chunk)
        return self
    def reset(self):
        self._metrics.reset()
        return self
    def result(self):
        return self._metrics.result()[self.metric_name]


class Accuracy(_MetricView):
    metric_name = 'accuracy'


class Precision(_MetricView):
    metric_name = 'precision'


class Recall(_MetricView):
    metric_name = 'recall'


class F1Score(_MetricView):
    metric_name = 'f1'


class AUC:
    """Accumulate y_true and positive-class probability scores for exact ROC AUC."""
    def __init__(self, *, positive_label=1):
        self.positive_label = positive_label
        self.reset()
    def reset(self):
        self._chunks: list[tuple[np.ndarray, np.ndarray]] = []
        return self
    def update(self, y_true_chunk, y_score_chunk):
        y = np.asarray(y_true_chunk)
        s = np.asarray(y_score_chunk, dtype=float)
        binary_auc(y, s, positive_label=self.positive_label)  # validate
        self._chunks.append((y.copy(), s.copy()))
        return self
    def result(self):
        if not self._chunks:
            return float('nan')
        return binary_auc(np.concatenate([a for a, _ in self._chunks]),
                          np.concatenate([b for _, b in self._chunks]),
                          positive_label=self.positive_label)


__all__ = ['StreamingClassificationMetrics', 'ConfusionMatrix', 'Accuracy',
           'Precision', 'Recall', 'F1Score', 'AUC', 'binary_auc']
