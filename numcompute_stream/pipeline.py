"""Leakage-conscious incremental pipeline for one buffered classification tree.

Transformers update their *learning* state per arriving chunk. Predictions
always use a deep-copied *frozen* transform state from the most recent tree
retrain. At a threshold, all raw buffered rows are re-transformed under one
consistent transform state and a single tree is retrained on those rows.
This is essential for expanding one-hot vocabularies and changing scalers.
No earlier chunks are kept after a retrain; the next buffer starts empty.
"""

from __future__ import annotations

import copy
import numpy as np

from .tree import DecisionTreeClassifier


class Pipeline:
    """Sequence of ``(name, transformer)`` stages ending in a decision tree.

    Each transformer implements ``partial_fit(X)`` and ``transform(X)``;
    the last stage must be ``DecisionTreeClassifier``. The pipeline accepts
    numerical and categorical 1D/2D X, always treating axis=0 as samples.
    ``partial_fit`` returns self; ``predict``/``predict_proba`` are unavailable
    until at least ``buffer_size`` samples have been received (or ``fit``).

    The pipeline buffers raw X/y, not potentially incompatible transformed
    chunks. Output-width changes at a retrain cause an explicit full fit of
    the tree on the current buffer; the old prediction snapshot is replaced
    atomically after successful training. This maintains bounded training
    buffers, but cumulative transformer state remains across retrains.
    """

    def __init__(self, steps):
        if not isinstance(steps, (list, tuple)) or not steps:
            raise ValueError('steps must be a nonempty sequence of (name, object) pairs.')
        names = []
        for item in steps:
            if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str) or not item[0]:
                raise TypeError('Each pipeline step must be (nonempty_name, object).')
            names.append(item[0])
        if len(set(names)) != len(names):
            raise ValueError('Pipeline step names must be unique.')
        if not isinstance(steps[-1][1], DecisionTreeClassifier):
            raise TypeError('The final pipeline step must be a DecisionTreeClassifier.')
        for name, transform in steps[:-1]:
            if not callable(getattr(transform, 'partial_fit', None)) or not callable(getattr(transform, 'transform', None)):
                raise TypeError(f'Transformer {name!r} needs partial_fit and transform methods.')
        self.steps = list(steps)
        self.reset_buffer()
        self._prediction_transformers = None
        self.n_retrains_ = 0
        self.n_samples_seen_ = 0
        self.n_features_in_ = None

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def model(self):
        return self.steps[-1][1]

    @property
    def buffer_size(self):
        return self.model.buffer_size

    @property
    def is_fitted(self):
        return self._prediction_transformers is not None

    def reset_buffer(self):
        self._buffer_X: list[np.ndarray] = []
        self._buffer_y: list[np.ndarray] = []
        self.n_buffered_samples_ = 0

    @staticmethod
    def _validate(X, y=None):
        data = np.asarray(X)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(f'X must be nonempty 2D (n_samples, n_features); got {data.shape}.')
        if y is None:
            return data
        labels = DecisionTreeClassifier._validate_y(y, len(data))
        return data, labels

    @staticmethod
    def _apply_transforms(data, transformers):
        result = data
        for _, transformer in transformers:
            result = transformer.transform(result)
        return result

    def _check_feature_count(self, data):
        if self.n_features_in_ is not None and data.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} input features, received {data.shape[1]}.')

    def fit(self, X, y):
        """Immediately train on a complete batch, replacing current state."""
        data, labels = self._validate(X, y)
        fitted = []
        transformed = data
        for name, transformer in self.steps[:-1]:
            if not callable(getattr(transformer, 'fit', None)):
                raise TypeError(f'Transformer {name!r} requires fit(X) for Pipeline.fit.')
            transformer.fit(transformed)
            transformed = transformer.transform(transformed)
            fitted.append((name, transformer))
        self.model.fit(transformed, labels)
        self._prediction_transformers = copy.deepcopy(fitted)
        self.reset_buffer()
        self.n_features_in_ = data.shape[1]
        self.n_samples_seen_ = len(data)
        self.n_retrains_ = 0
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Update transformers; buffer raw rows; retrain only at threshold."""
        data, labels = self._validate(X_chunk, y_chunk)
        self._check_feature_count(data)
        # Update online transform state. The inference snapshot remains frozen.
        transformed = data
        for name, transformer in self.steps[:-1]:
            try:
                transformer.partial_fit(transformed)
                transformed = transformer.transform(transformed)
            except ValueError as exc:
                raise ValueError(f'Pipeline transformer {name!r} failed: {exc}') from exc
        self._buffer_X.append(data.copy())
        self._buffer_y.append(labels.copy())
        self.n_buffered_samples_ += len(data)
        self.n_samples_seen_ += len(data)
        if self.n_features_in_ is None:
            self.n_features_in_ = data.shape[1]
        if self.n_buffered_samples_ >= self.buffer_size:
            all_raw = np.concatenate(self._buffer_X, axis=0)
            all_y = np.concatenate(self._buffer_y, axis=0)
            features = self._apply_transforms(all_raw, self.steps[:-1])
            # The old tree cannot accept a new one-hot width. Here all
            # accumulated rows already exceed the retraining threshold, so
            # fitting on them now preserves the specified buffer semantics.
            if self.model.n_features_in_ is not None and features.shape[1] != self.model.n_features_in_:
                self.model.fit(features, all_y)
            else:
                self.model.partial_fit(features, all_y)
            self._prediction_transformers = copy.deepcopy(self.steps[:-1])
            self.n_retrains_ += 1
            self.reset_buffer()
        return self

    update = partial_fit

    def _prepare_predict(self, X):
        if not self.is_fitted:
            raise RuntimeError('Pipeline is not trained: collect buffer_size samples or call fit().')
        data = self._validate(X)
        self._check_feature_count(data)
        return self._apply_transforms(data, self._prediction_transformers)

    def predict(self, X):
        return self.model.predict(self._prepare_predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._prepare_predict(X))


__all__ = ['Pipeline']
