"""Orchestrate prequential (test-then-train) classification streams.

A chunk is scored *before* its features or labels update the pipeline.
Chunks arriving before the first buffered fit are not scored (there is no
model yet). Logs mark them as warm-up; cumulative scores exclude them.
"""

from __future__ import annotations

import logging
import sys
from time import perf_counter

import numpy as np

from .metrics import StreamingClassificationMetrics
from .pipeline import Pipeline
from .tree import DecisionTreeClassifier

_LOG = logging.getLogger(__name__)


def estimate_memory_bytes(value, seen=None) -> int:
    """Estimate Python+NumPy object memory recursively (not process RSS)."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        total = sys.getsizeof(value)
        if value.dtype.hasobject:
            total += sum(estimate_memory_bytes(v, seen) for v in value.flat)
        return total
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        return total + sum(estimate_memory_bytes(k, seen) + estimate_memory_bytes(v, seen)
                           for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return total + sum(estimate_memory_bytes(v, seen) for v in value)
    if hasattr(value, '__dict__') and not isinstance(value, type):
        return total + estimate_memory_bytes(vars(value), seen)
    return total


class StreamTrainer:
    """Manage a buffered tree or ``Pipeline`` with prequential evaluation.

    ``fit_chunk(X, y)`` scores the chunk if a trained model already exists,
    then calls ``partial_fit`` exactly once. ``score_chunk`` scores without
    fitting, and updates the same cumulative metrics. It does *not* create a
    training event in ``history_``. ``update`` aliases ``fit_chunk``.

    ``history_`` is a list of per-chunk dictionaries, suitable for plotting.
    ``memory_bytes`` is a Python object-size estimate rather than process RSS.
    All accuracy fields exclude warm-up chunks without predictions.
    """

    def __init__(self, model: Pipeline | DecisionTreeClassifier, *,
                 window_size: int | None = 200, positive_label=None,
                 verbose: bool = False):
        if not isinstance(model, (Pipeline, DecisionTreeClassifier)):
            raise TypeError('StreamTrainer requires a Pipeline or DecisionTreeClassifier.')
        self.model = model
        self.metrics = StreamingClassificationMetrics(window_size=window_size,
                                                       positive_label=positive_label)
        self.verbose = bool(verbose)
        self.history_: list[dict] = []
        self.n_chunks_seen_ = 0
        self.n_samples_seen_ = 0
        self.n_warmup_samples_ = 0

    def _is_fitted(self):
        if isinstance(self.model, Pipeline):
            return self.model.is_fitted
        return self.model._tree is not None

    def _model_tree(self):
        return self.model.model if isinstance(self.model, Pipeline) else self.model

    def score_chunk(self, X_chunk, y_chunk) -> dict:
        """Score without fitting; cumulative metrics only include scored rows."""
        data, labels = Pipeline._validate(X_chunk, y_chunk)
        if not self._is_fitted():
            return {'evaluated': False, 'n_evaluated': 0, 'chunk_accuracy': float('nan'),
                    'chunk_error': float('nan')}
        predictions = self.model.predict(data)
        chunk_accuracy = float(np.mean(predictions == labels))
        scores = None
        try:
            probabilities = self.model.predict_proba(data)
            classes = self._model_tree().classes_
            # AUC is meaningful only for two-class models, with positive
            # score matched to the correct class index.
            if len(classes) == 2:
                pos = self.metrics.positive_label
                if pos is None:
                    # Pin the positive class so successive AUC chunks never
                    # accidentally mix incompatible score meanings.
                    pos = classes[-1]
                    self.metrics.positive_label = pos
                idx = np.flatnonzero(classes == pos)
                if len(idx):
                    scores = probabilities[:, int(idx[0])]
        except AttributeError:
            pass
        self.metrics.update(labels, predictions, y_score_chunk=scores)
        return {'evaluated': True, 'n_evaluated': len(labels),
                'chunk_accuracy': chunk_accuracy, 'chunk_error': 1 - chunk_accuracy}

    def fit_chunk(self, X_chunk, y_chunk) -> dict:
        """Test on the incoming chunk first, then update buffered training."""
        data, labels = Pipeline._validate(X_chunk, y_chunk)
        result = self.score_chunk(data, labels)
        if not result['evaluated']:
            self.n_warmup_samples_ += len(labels)
        start = perf_counter()
        self.model.partial_fit(data, labels)
        training_ms = (perf_counter() - start) * 1000.0
        self.n_chunks_seen_ += 1
        self.n_samples_seen_ += len(labels)
        metrics = self.metrics.result()
        row = {
            'chunk': self.n_chunks_seen_,
            'n_samples': len(labels),
            'n_evaluated': result['n_evaluated'],
            'evaluated': result['evaluated'],
            'chunk_accuracy': result['chunk_accuracy'],
            'chunk_error': result['chunk_error'],
            'cumulative_accuracy': metrics['accuracy'],
            'cumulative_error': metrics['error'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'rolling_accuracy': metrics.get('rolling_accuracy', float('nan')),
            'n_evaluated_total': metrics['support'],
            'n_retrains': self.model.n_retrains_,
            'buffered_samples': self.model.n_buffered_samples_,
            'update_ms': training_ms,
            'memory_bytes': estimate_memory_bytes(self.model) + estimate_memory_bytes(self.metrics),
        }
        self.history_.append(row)
        if self.verbose:
            _LOG.info('chunk=%d rows=%d evaluated=%s accuracy=%.3f cumulative=%.3f '
                      'buffered=%d retrains=%d update_ms=%.2f memory_bytes=%d',
                      row['chunk'], row['n_samples'], row['evaluated'],
                      row['chunk_accuracy'], row['cumulative_accuracy'],
                      row['buffered_samples'], row['n_retrains'], row['update_ms'],
                      row['memory_bytes'])
        return row.copy()

    update = fit_chunk

    def result(self) -> dict:
        """Return current cumulative classification results and stream counts."""
        data = self.metrics.result()
        data.update(n_chunks_seen=self.n_chunks_seen_,
                    n_samples_seen=self.n_samples_seen_,
                    n_warmup_samples=self.n_warmup_samples_,
                    n_retrains=self.model.n_retrains_,
                    buffered_samples=self.model.n_buffered_samples_)
        return data


__all__ = ['StreamTrainer', 'estimate_memory_bytes']
