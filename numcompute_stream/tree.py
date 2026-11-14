"""A vectorised, buffered-retraining decision tree classifier.

This module implements CART-style binary splits using Gini or entropy.
Every candidate threshold of a feature is evaluated in one NumPy batch,
using sorted feature values and cumulative class counts. Recursion over
nodes is unavoidable, but individual training samples are not looped over.

For 0/1 features without missing values, splits use
``X[:, feature] == 1``: YES travels left and NO travels right.
For general numeric features, splits use ``X[:, feature] <= threshold``.
NaNs always travel right, including when a numeric threshold is chosen.
Infinite values are rejected; label ties favour the first sorted class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class _Node:
    """One node in the learned binary decision tree."""

    prediction: int
    counts: npt.NDArray[np.int64]
    feature: int | None = None
    threshold: float | None = None
    binary_split: bool = False
    left: _Node | None = None
    right: _Node | None = None


class DecisionTreeClassifier:
    """Train a classification tree and retrain it at buffer thresholds.

    Parameters
    ----------
    max_depth : int or None, default=5
        Root depth is 0. ``None`` permits growth until stopping conditions.
    min_samples_split : int, default=2
        Minimum number of observations to consider splitting a node.
    max_features : int, float, {'sqrt', 'log2'} or None, default=None
        Number/fraction of features considered per node. ``None`` uses all.
    buffer_size : int, default=500
        Minimum accumulated new observations that trigger a fresh fit.
        If the accumulated buffer exceeds this limit, the entire buffer
        is fitted once, then cleared (no leftover observations).
    criterion : {'gini', 'entropy'}, default='gini'
        Impurity measure used to select splits.
    random_state : int or None, default=0
        Seed for random feature sub-selection when max_features is restricted.

    Notes
    -----
    Features X must have shape (n_samples, n_features); finite floats and
    NaNs are accepted. Labels y have shape (n_samples,). Before the first
    trained tree exists, predict/predict_proba raise RuntimeError. A buffered
    retrain replaces, rather than incrementally modifies, the previous tree.
    """

    def __init__(
        self,
        max_depth: int | None = 5,
        min_samples_split: int = 2,
        max_features: int | float | str | None = None,
        buffer_size: int = 500,
        criterion: str = "gini",
        random_state: int | None = 0,
    ) -> None:
        if max_depth is not None and (
            not isinstance(max_depth, (int, np.integer))
            or isinstance(max_depth, (bool, np.bool_))
            or max_depth < 0
        ):
            raise ValueError("max_depth must be a non-negative integer or None.")
        if (
            not isinstance(min_samples_split, (int, np.integer))
            or isinstance(min_samples_split, (bool, np.bool_))
            or min_samples_split < 2
        ):
            raise ValueError("min_samples_split must be an integer >= 2.")
        if (
            not isinstance(buffer_size, (int, np.integer))
            or isinstance(buffer_size, (bool, np.bool_))
            or buffer_size < 1
        ):
            raise ValueError("buffer_size must be a positive integer.")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if max_features is not None:
            if isinstance(max_features, (bool, np.bool_)):
                raise ValueError("max_features cannot be boolean.")
            if isinstance(max_features, (int, np.integer)):
                if max_features < 1:
                    raise ValueError("max_features integer must be >= 1.")
            elif isinstance(max_features, (float, np.floating)):
                if not (0 < max_features <= 1):
                    raise ValueError("max_features fraction must be in (0, 1].")
            elif max_features not in ("sqrt", "log2"):
                raise ValueError(
                    "max_features must be None, positive integer, fraction, "
                    "'sqrt' or 'log2'."
                )
        if random_state is not None and (
            not isinstance(random_state, (int, np.integer))
            or isinstance(random_state, (bool, np.bool_))
            or random_state < 0
        ):
            raise ValueError("random_state must be a non-negative integer or None.")

        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.buffer_size = int(buffer_size)
        self.criterion = criterion
        self.random_state = random_state
        self._tree: _Node | None = None
        self._buffer_X: list[npt.NDArray[np.float64]] = []
        self._buffer_y: list[npt.NDArray[Any]] = []
        self.n_buffered_samples_ = 0
        self.n_samples_seen_ = 0
        self.n_retrains_ = 0
        self.n_features_in_: int | None = None

    @staticmethod
    def _validate_X(
        X: npt.ArrayLike, *, allow_empty: bool = False
    ) -> npt.NDArray[np.float64]:
        """Validate numerical, two-dimensional observations."""
        try:
            values = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("X must contain numeric values or NaNs.") from exc
        if values.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features); got {values.shape}.")
        if values.shape[1] == 0 or (not allow_empty and values.shape[0] == 0):
            raise ValueError("X must have at least one feature and one sample.")
        if np.isinf(values).any():
            raise ValueError("X may contain NaNs but must not contain infinity.")
        return values

    @staticmethod
    def _validate_y(y: npt.ArrayLike, n_samples: int) -> npt.NDArray[Any]:
        """Validate a nonempty vector of classification labels."""
        values = np.asarray(y)
        if values.ndim != 1:
            raise ValueError(f"y must be 1D (n_samples,); got {values.shape}.")
        if len(values) != n_samples:
            raise ValueError(
                f"X and y must contain the same number of samples; "
                f"got {n_samples} and {len(values)}."
            )
        if any(
            value is None
            or (isinstance(value, (float, np.floating)) and np.isnan(value))
            for value in values
        ):
            raise ValueError("y must not contain missing labels.")
        try:
            np.unique(values)
        except TypeError as exc:
            raise TypeError("y labels must be mutually comparable values.") from exc
        return values.copy()

    def _feature_count(self, n_features: int) -> int:
        """Resolve how many features are considered at one split."""
        feature_option = self.max_features
        if feature_option is None:
            return n_features
        if isinstance(feature_option, (int, np.integer)):
            if feature_option > n_features:
                raise ValueError("max_features cannot exceed the feature count.")
            return int(feature_option)
        if isinstance(feature_option, (float, np.floating)):
            return max(1, int(np.ceil(n_features * feature_option)))
        if feature_option == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        return max(1, int(np.log2(n_features)))

    def _impurity(self, class_counts: npt.NDArray[np.generic]) -> np.ndarray:
        """Return Gini/entropy for a batch of class-count vectors."""
        class_counts = np.asarray(class_counts, dtype=float)
        totals = class_counts.sum(axis=-1, keepdims=True)
        p = np.divide(
            class_counts, totals, out=np.zeros_like(class_counts), where=totals > 0
        )
        if self.criterion == "gini":
            return 1 - np.sum(p * p, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.sum(np.where(p > 0, p * np.log2(np.maximum(p, 1e-300)), 0), axis=-1)

    def _best_split(
        self, X: np.ndarray, encoded_y: np.ndarray, parent_impurity: float
    ) -> tuple[int, float, bool] | None:
        """Find the best split using vectorised cumulative class counts.

        The last return item marks a 0/1 question using ``== 1`` (YES
        left), using the binary-feature branch convention.
        Continuous features use a numeric threshold (``<=`` left).
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        selected = self._rng.choice(
            n_features, size=self._feature_count(n_features), replace=False
        )
        selected.sort()
        totals = np.bincount(encoded_y, minlength=n_classes)
        best_gain = 1e-12
        best: tuple[int, float, bool] | None = None

        for feature in selected:  # One loop per selected feature, not per sample.
            column = X[:, feature]
            non_missing = ~np.isnan(column)
            valid_vals = column[non_missing]
            if len(valid_vals) == 0:
                continue
            order = np.argsort(valid_vals, kind="mergesort")
            sorted_vals = valid_vals[order]
            sorted_y = encoded_y[non_missing][order]
            cumulative = np.cumsum(np.eye(n_classes, dtype=np.int64)[sorted_y], axis=0)
            positions = np.flatnonzero(np.r_[sorted_vals[1:] != sorted_vals[:-1], True])
            # The end position is useful only when there are missing values:
            # it explicitly splits observed values from NaNs.
            left_counts = cumulative[positions]
            n_left = positions + 1
            n_right = n_samples - n_left
            keep = n_right > 0
            if not keep.any():
                continue
            positions = positions[keep]
            left_counts = left_counts[keep]
            n_left, n_right = n_left[keep], n_right[keep]
            right_counts = totals - left_counts  # Includes missing-value labels.
            weighted = (
                n_left * self._impurity(left_counts)
                + n_right * self._impurity(right_counts)
            ) / n_samples
            gains = parent_impurity - weighted
            ix = int(np.argmax(gains))
            if gains[ix] > best_gain:
                start = int(positions[ix])
                end = start + 1
                if end < len(sorted_vals):
                    # Midpoint avoids overflow for large finite magnitudes.
                    threshold = float(
                        sorted_vals[start] * 0.5 + sorted_vals[end] * 0.5
                    )
                    # If the midpoint rounds to the higher value, the split
                    # still separates the observations because we use <=.
                    if threshold >= sorted_vals[end]:
                        threshold = float(sorted_vals[start])
                else:
                    threshold = float(sorted_vals[start])
                # Binary 0/1 features send YES rows to the left. This is
                # the same partition as
                # a 0.5 threshold but with reversed child ordering. If NaNs
                # exist, preserve the general threshold rule so NaNs stay
                # on the right and the evaluated gain remains accurate.
                binary_split = bool(
                    non_missing.all()
                    and len(sorted_vals) >= 2
                    and sorted_vals[0] == 0
                    and sorted_vals[-1] == 1
                    and np.all((sorted_vals == 0) | (sorted_vals == 1))
                )
                best_gain = float(gains[ix])
                best = (int(feature), threshold, binary_split)
        return best

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """Construct one subtree recursively."""
        counts = np.bincount(y, minlength=len(self.classes_)).astype(np.int64)
        node = _Node(prediction=int(np.argmax(counts)), counts=counts)
        impurity = float(self._impurity(counts))
        if (
            impurity <= 1e-12
            or len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node
        split = self._best_split(X, y, impurity)
        if split is None:
            return node
        feature, threshold, binary_split = split
        if binary_split:
            # Week 10: "feature == yes?"; yes rows go LEFT.
            left_mask = X[:, feature] == 1
        else:
            left_mask = X[:, feature] <= threshold  # NaNs go RIGHT.
        if not left_mask.any() or left_mask.all():
            return node
        node.feature = feature
        node.threshold = threshold
        node.binary_split = binary_split
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit one new tree on exactly the supplied observations."""
        try:
            self.classes_, encoded_y = np.unique(y, return_inverse=True)
        except TypeError as exc:
            raise TypeError("y labels must be mutually comparable values.") from exc
        self._feature_count(X.shape[1])
        self._rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = X.shape[1]
        self._tree = self._build_tree(X, encoded_y, depth=0)

    def _clear_buffer(self) -> None:
        """Release all buffered chunks after retraining."""
        self._buffer_X.clear()
        self._buffer_y.clear()
        self.n_buffered_samples_ = 0

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "DecisionTreeClassifier":
        """Fit a fresh tree immediately (without waiting for buffer_size)."""
        features = self._validate_X(X)
        labels = self._validate_y(y, len(features))
        self._train(features, labels)
        self._clear_buffer()
        return self

    def partial_fit(
        self, X_chunk: npt.ArrayLike, y_chunk: npt.ArrayLike
    ) -> "DecisionTreeClassifier":
        """Buffer one chunk; retrain on all buffered chunks at the threshold.

        Each incoming call is atomic with respect to training: a 600-row
        chunk with buffer_size=500 produces a single 600-row retrain.
        """
        features = self._validate_X(X_chunk)
        labels = self._validate_y(y_chunk, len(features))
        expected = self.n_features_in_
        if expected is None and self._buffer_X:
            expected = self._buffer_X[0].shape[1]
        if expected is not None and features.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} features; received {features.shape[1]}."
            )
        self._feature_count(features.shape[1])
        self._buffer_X.append(features.copy())
        self._buffer_y.append(labels)
        self.n_buffered_samples_ += len(labels)
        self.n_samples_seen_ += len(labels)
        if self.n_buffered_samples_ >= self.buffer_size:
            X_train = np.concatenate(self._buffer_X, axis=0)
            y_train = np.concatenate(self._buffer_y, axis=0)
            self._train(X_train, y_train)
            self._clear_buffer()
            self.n_retrains_ += 1
        return self

    def _require_tree(self) -> _Node:
        """Return the latest trained tree or explain why it is unavailable."""
        if self._tree is None:
            raise RuntimeError(
                "The tree is not trained: call fit(), or accumulate at "
                "least buffer_size samples with partial_fit()."
            )
        return self._tree

    def _predict_distribution(
        self, node: _Node, features: np.ndarray, rows: np.ndarray, result: np.ndarray
    ) -> None:
        """Route sample-index arrays recursively, without a sample loop."""
        if node.feature is None:
            result[rows] = node.counts / node.counts.sum()
            return
        assert node.left is not None and node.right is not None
        assert node.threshold is not None
        if node.binary_split:
            goes_left = features[rows, node.feature] == 1
        else:
            goes_left = features[rows, node.feature] <= node.threshold
        if goes_left.any():
            self._predict_distribution(node.left, features, rows[goes_left], result)
        if (~goes_left).any():
            self._predict_distribution(node.right, features, rows[~goes_left], result)

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return per-class probabilities, shape (n_samples, n_classes)."""
        tree = self._require_tree()
        features = self._validate_X(X, allow_empty=True)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features; "
                f"received {features.shape[1]}."
            )
        result = np.empty((len(features), len(self.classes_)), dtype=float)
        if len(features):
            self._predict_distribution(
                tree, features, np.arange(len(features)), result
            )
        return result

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Predict one class per row, shape (n_samples,)."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
