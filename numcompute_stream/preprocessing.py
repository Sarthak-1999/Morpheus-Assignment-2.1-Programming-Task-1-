"""Incremental transformers for numeric/categorical streaming features.

All operations treat rows as samples (axis=0) and columns as features
(axis=1). One-dimensional input is accepted as a single feature and
transform always returns a 2D output suitable for a decision tree.

Unlike ``numcompute.preprocessing`` (Assignment 2.1), these classes
maintain sufficient statistics or dynamically expand their vocabulary.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _matrix(X: npt.ArrayLike, *, numeric: bool = True, empty_ok: bool = False) -> np.ndarray:
    """Convert a chunk to (n_samples, n_features); reject invalid shapes."""
    try:
        values = np.asarray(X, dtype=float if numeric else object)
    except (TypeError, ValueError) as exc:
        raise TypeError("X must contain numeric values for this transformer.") from exc
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"X must be 1D or 2D; received shape {values.shape}.")
    if values.shape[1] == 0 or (not empty_ok and values.shape[0] == 0):
        raise ValueError("X must contain at least one feature and one sample.")
    if numeric and np.isinf(values).any():
        raise ValueError("X must not contain infinity; use NaN for missing values.")
    return values


def _fitted(obj: object) -> None:
    if not getattr(obj, "_fitted", False):
        raise RuntimeError(f"{type(obj).__name__} is not fitted. Call fit/partial_fit first.")


def _check_features(obj: object, X: np.ndarray) -> None:
    if X.shape[1] != obj.n_features_in_:  # type: ignore[attr-defined]
        raise ValueError(f"Expected {obj.n_features_in_} features, received {X.shape[1]}.")  # type: ignore[attr-defined]


class StandardScaler:
    """Per-column streaming z-score standardiser using batch-Welford merges.

    ``partial_fit(X_chunk)`` learns from rows along axis=0, ignoring NaNs.
    Features with zero variance or no observations use scale 1.0. Missing
    entries remain NaN after ``transform``. No past training chunks are kept.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> "StandardScaler":
        self._fitted = False
        return self

    def fit(self, X: npt.ArrayLike) -> "StandardScaler":
        return self.reset().partial_fit(X)

    def partial_fit(self, X: npt.ArrayLike) -> "StandardScaler":
        data = _matrix(X)
        if not self._fitted:
            self.n_features_in_ = data.shape[1]
            self.n_samples_seen_ = np.zeros(data.shape[1], dtype=np.int64)
            self.mean_ = np.zeros(data.shape[1], dtype=float)
            self._m2 = np.zeros(data.shape[1], dtype=float)
            self._fitted = True
        _check_features(self, data)
        valid = ~np.isnan(data)
        batch_count = valid.sum(axis=0)
        safe_count = np.maximum(batch_count, 1)
        batch_mean = np.sum(np.where(valid, data, 0.0), axis=0) / safe_count
        batch_m2 = np.sum(np.where(valid, (data - batch_mean) ** 2, 0.0), axis=0)
        old = self.n_samples_seen_
        total = old + batch_count
        delta = batch_mean - self.mean_
        ratio = np.divide(batch_count, total, out=np.zeros_like(batch_mean), where=total > 0)
        self.mean_ += delta * ratio
        cross = np.divide(old * batch_count, total, out=np.zeros_like(batch_mean), where=total > 0)
        self._m2 += batch_m2 + delta * delta * cross
        self.n_samples_seen_ = total
        self.var_ = np.divide(self._m2, total, out=np.zeros_like(self._m2), where=total > 0)
        self.scale_ = np.where(self.var_ > 0, np.sqrt(self.var_), 1.0)
        return self

    def transform(self, X: npt.ArrayLike) -> np.ndarray:
        _fitted(self)
        data = _matrix(X, empty_ok=True)
        _check_features(self, data)
        return (data - self.mean_) / self.scale_

    def fit_transform(self, X: npt.ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Cumulative, NaN-aware min/max normalisation per feature column."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        if len(feature_range) != 2 or not np.all(np.isfinite(feature_range)) or feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range must be a finite pair (minimum, maximum) with minimum < maximum.")
        self.feature_range = tuple(map(float, feature_range))
        self.reset()

    def reset(self) -> "MinMaxScaler":
        self._fitted = False
        return self

    def fit(self, X: npt.ArrayLike) -> "MinMaxScaler":
        return self.reset().partial_fit(X)

    def partial_fit(self, X: npt.ArrayLike) -> "MinMaxScaler":
        data = _matrix(X)
        if not self._fitted:
            self.n_features_in_ = data.shape[1]
            self.data_min_ = np.full(data.shape[1], np.inf)
            self.data_max_ = np.full(data.shape[1], -np.inf)
            self._fitted = True
        _check_features(self, data)
        valid = ~np.isnan(data)
        current_min = np.min(np.where(valid, data, np.inf), axis=0)
        current_max = np.max(np.where(valid, data, -np.inf), axis=0)
        self.data_min_ = np.minimum(self.data_min_, current_min)
        self.data_max_ = np.maximum(self.data_max_, current_max)
        seen = np.isfinite(self.data_min_)
        self.data_range_ = np.where(seen, self.data_max_ - self.data_min_, 0.0)
        self.scale_ = np.where(self.data_range_ > 0, self.data_range_, 1.0)
        return self

    def transform(self, X: npt.ArrayLike) -> np.ndarray:
        _fitted(self)
        data = _matrix(X, empty_ok=True)
        _check_features(self, data)
        lower, upper = self.feature_range
        origin = np.where(np.isfinite(self.data_min_), self.data_min_, 0.0)
        return ((data - origin) / self.scale_) * (upper - lower) + lower

    def fit_transform(self, X: npt.ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)


class Imputer:
    """Streaming per-feature NaN imputer.

    Strategies: ``mean`` (cumulative sum/count), ``median`` (exact; stores
    observed values), ``constant`` (no learning). When a feature has never
    had a valid observation, ``fill_value`` is used (default 0.0).
    Numeric missing data is represented by ``np.nan``.
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0) -> None:
        if strategy not in {"mean", "median", "constant"}:
            raise ValueError("strategy must be 'mean', 'median', or 'constant'.")
        if not isinstance(fill_value, (int, float, np.number)) or not np.isfinite(fill_value):
            raise ValueError("fill_value must be a finite number.")
        self.strategy = strategy
        self.fill_value = float(fill_value)
        self.reset()

    def reset(self) -> "Imputer":
        self._fitted = False
        self._observed: list[np.ndarray] = []
        return self

    def fit(self, X: npt.ArrayLike) -> "Imputer":
        return self.reset().partial_fit(X)

    def partial_fit(self, X: npt.ArrayLike) -> "Imputer":
        data = _matrix(X)
        if not self._fitted:
            self.n_features_in_ = data.shape[1]
            self.n_samples_seen_ = np.zeros(data.shape[1], dtype=np.int64)
            self._sum = np.zeros(data.shape[1], dtype=float)
            self.statistics_ = np.full(data.shape[1], self.fill_value, dtype=float)
            self._observed = [np.empty(0, dtype=float) for _ in range(data.shape[1])]
            self._fitted = True
        _check_features(self, data)
        if self.strategy == "constant":
            return self
        valid = ~np.isnan(data)
        count = valid.sum(axis=0)
        self.n_samples_seen_ += count
        if self.strategy == "mean":
            self._sum += np.where(valid, data, 0.0).sum(axis=0)
            self.statistics_ = np.divide(
                self._sum, self.n_samples_seen_,
                out=np.full_like(self._sum, self.fill_value),
                where=self.n_samples_seen_ > 0,
            )
        else:  # The exact median requires retaining observed values.
            for column in range(data.shape[1]):
                self._observed[column] = np.concatenate((self._observed[column], data[valid[:, column], column]))
                if self._observed[column].size:
                    self.statistics_[column] = float(np.median(self._observed[column]))
        return self

    def transform(self, X: npt.ArrayLike) -> np.ndarray:
        _fitted(self)
        data = _matrix(X, empty_ok=True)
        _check_features(self, data)
        return np.where(np.isnan(data), self.statistics_, data)

    def fit_transform(self, X: npt.ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)


class OneHotEncoder:
    """Incremental one-hot encoder with per-column expanding categories.

    Accepts 1D or 2D categorical arrays. Columns are independent and
    categories retain discovery order so adding a new category never changes
    *its own column's* prior category indices. The width of ``transform``
    changes when new categories appear. Unknown categories use zero vectors
    by default; set ``handle_unknown='error'`` to reject them.

    Important: the consuming pipeline must reconcile feature-count changes
    with an already trained decision tree; do not directly send a widened
    matrix to an old tree. The streaming Pipeline keeps an inference-time
    transformer snapshot until the next buffered tree retraining.
    """

    def __init__(self, handle_unknown: str = "ignore") -> None:
        if handle_unknown not in {"ignore", "error"}:
            raise ValueError("handle_unknown must be 'ignore' or 'error'.")
        self.handle_unknown = handle_unknown
        self.reset()

    def reset(self) -> "OneHotEncoder":
        self._fitted = False
        return self

    @staticmethod
    def _text_matrix(X: npt.ArrayLike, *, empty_ok: bool = False) -> np.ndarray:
        data = _matrix(X, numeric=False, empty_ok=empty_ok)
        # Use text tokens to make mixed numeric/string categories comparable.
        # None/NaN are normalised into one common missing-category token.
        # A string with exactly the reserved token is deliberately rejected.
        if data.size:
            missing = np.equal(data, None) | (data != data)  # noqa: E711
            text_values = np.asarray(data, dtype=str)
            if np.any((text_values == "<MISSING>") & ~missing):
                raise ValueError("'<MISSING>' is reserved for absent categories.")
            text_values[missing] = "<MISSING>"
            return text_values
        return data.astype(str)

    def fit(self, X: npt.ArrayLike) -> "OneHotEncoder":
        return self.reset().partial_fit(X)

    def partial_fit(self, X: npt.ArrayLike) -> "OneHotEncoder":
        data = self._text_matrix(X)
        if not self._fitted:
            self.n_features_in_ = data.shape[1]
            self.categories_: tuple[np.ndarray, ...] = tuple(np.empty(0, dtype=str) for _ in range(data.shape[1]))
            self._fitted = True
        _check_features(self, data)
        updated = []
        for column, existing in enumerate(self.categories_):
            unique = np.unique(data[:, column])
            unseen = unique[~np.isin(unique, existing)]
            updated.append(np.concatenate((existing, unseen)))
        self.categories_ = tuple(updated)
        self.n_features_out_ = sum(len(cat) for cat in self.categories_)
        return self

    def transform(self, X: npt.ArrayLike) -> np.ndarray:
        _fitted(self)
        data = self._text_matrix(X, empty_ok=True)
        _check_features(self, data)
        result = np.zeros((data.shape[0], self.n_features_out_), dtype=float)
        offset = 0
        for column, categories in enumerate(self.categories_):
            if data.shape[0]:
                matches = data[:, column, None] == categories[None, :]
                unknown = ~np.any(matches, axis=1)
                if unknown.any() and self.handle_unknown == "error":
                    raise ValueError(f"Unknown category in feature column {column}.")
                result[:, offset:offset + len(categories)] = matches
            offset += len(categories)
        return result

    def fit_transform(self, X: npt.ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)
