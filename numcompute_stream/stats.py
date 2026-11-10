"""Per-chunk numerical statistics using stable parallel variance merging.

All reductions use samples on axis=0 and features on axis=1. NaNs are
ignored per feature; infinities are rejected. Mean and variance are true
cumulative statistics. Quantiles and histograms need retained values:
by default they are exact across all chunks (memory grows with data).
Set ``window_size`` for bounded-memory *rolling* quantiles/histograms.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt

from .preprocessing import _check_features, _fitted, _matrix


class StreamingStats:
    """Track cumulative mean/variance and exact or sliding-window distributions.

    Parameters
    ----------
    window_size : int or None
        Number of most recent samples kept for quantiles and histograms.
        ``None`` stores all samples; mean/variance always use all samples.
    """

    def __init__(self, window_size: int | None = None) -> None:
        if window_size is not None and (
            not isinstance(window_size, (int, np.integer))
            or isinstance(window_size, (bool, np.bool_))
            or window_size <= 0
        ):
            raise ValueError("window_size must be a positive integer or None.")
        self.window_size = window_size
        self.reset()

    def reset(self) -> "StreamingStats":
        self._fitted = False
        self._history: deque[np.ndarray] = deque()
        self._history_count = 0
        return self

    def update_stats(self, X_chunk: npt.ArrayLike) -> "StreamingStats":
        """Accumulate one numeric chunk of shape (n_samples, n_features)."""
        data = _matrix(X_chunk)
        if not self._fitted:
            self.n_features_in_ = data.shape[1]
            self.count_ = np.zeros(data.shape[1], dtype=np.int64)
            self.mean_ = np.zeros(data.shape[1], dtype=float)
            self._m2 = np.zeros(data.shape[1], dtype=float)
            self.n_chunks_seen_ = 0
            self.n_samples_seen_ = 0
            self._fitted = True
        _check_features(self, data)
        valid = ~np.isnan(data)
        batch_count = valid.sum(axis=0)
        batch_mean = np.divide(
            np.where(valid, data, 0.0).sum(axis=0), batch_count,
            out=np.zeros(data.shape[1]), where=batch_count > 0,
        )
        batch_m2 = np.sum(np.where(valid, (data - batch_mean) ** 2, 0.0), axis=0)
        total = self.count_ + batch_count
        delta = batch_mean - self.mean_
        ratio = np.divide(batch_count, total, out=np.zeros_like(batch_mean), where=total > 0)
        self.mean_ += delta * ratio
        cross = np.divide(self.count_ * batch_count, total, out=np.zeros_like(batch_mean), where=total > 0)
        self._m2 += batch_m2 + delta**2 * cross
        self.count_ = total
        self.n_chunks_seen_ += 1
        self.n_samples_seen_ += data.shape[0]
        self._history.append(data.copy())
        self._history_count += len(data)
        if self.window_size is not None:
            while self._history_count > self.window_size:
                excess = self._history_count - self.window_size
                old = self._history[0]
                if len(old) <= excess:
                    self._history.popleft()
                    self._history_count -= len(old)
                else:
                    self._history[0] = old[excess:].copy()
                    self._history_count -= excess
        return self

    update = update_stats
    partial_fit = update_stats

    def fit(self, X: npt.ArrayLike) -> "StreamingStats":
        return self.reset().update_stats(X)

    def mean(self) -> np.ndarray:
        _fitted(self)
        return np.where(self.count_ > 0, self.mean_, np.nan).copy()

    def variance(self, ddof: int = 0) -> np.ndarray:
        """Per-column variance. Uses NaN if observed count <= ddof."""
        _fitted(self)
        if not isinstance(ddof, (int, np.integer)) or ddof < 0:
            raise ValueError("ddof must be a non-negative integer.")
        denominator = self.count_ - ddof
        return np.divide(
            self._m2, denominator,
            out=np.full(self.n_features_in_, np.nan),
            where=denominator > 0,
        ).clip(min=0, max=None)

    def std(self, ddof: int = 0) -> np.ndarray:
        return np.sqrt(self.variance(ddof=ddof))

    def _window_data(self) -> np.ndarray:
        _fitted(self)
        return np.concatenate(list(self._history), axis=0)

    def quantile(self, q: float | npt.ArrayLike) -> np.ndarray:
        """Exact quantiles across all stored/rolling rows; q in [0, 1].

        Scalar q returns (n_features,); an array of m q values returns
        (m, n_features). All-NaN columns return NaN.
        """
        queries = np.asarray(q, dtype=float)
        if queries.ndim > 1 or not np.isfinite(queries).all() or np.any((queries < 0) | (queries > 1)):
            raise ValueError("q must contain finite quantiles in [0, 1] and be scalar or 1D.")
        data = self._window_data()
        answer = np.full((queries.size, self.n_features_in_), np.nan)
        for feature in range(self.n_features_in_):
            finite = data[:, feature]
            finite = finite[~np.isnan(finite)]
            if len(finite):
                answer[:, feature] = np.quantile(finite, queries.reshape(-1))
        return answer.reshape(self.n_features_in_) if queries.ndim == 0 else answer

    def histogram(
        self, bins: int | npt.ArrayLike = 10,
        value_range: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-column histograms and matching bin edges.

        Output count shape (n_features, n_bins), edges (n_features, n_bins+1).
        For comparison across different calls/chunks, pass a fixed range or
        explicit bin edges; otherwise observed min/max can change.
        """
        data = self._window_data()
        counts = []
        edges = []
        for feature in range(self.n_features_in_):
            valid = data[:, feature]
            valid = valid[~np.isnan(valid)]
            c, e = np.histogram(valid, bins=bins, range=value_range)
            counts.append(c)
            edges.append(e)
        return np.stack(counts), np.stack(edges)

    def result(self) -> dict[str, np.ndarray | int]:
        _fitted(self)
        return {
            "mean": self.mean(),
            "variance": self.variance(),
            "count": self.count_.copy(),
            "n_samples_seen": self.n_samples_seen_,
            "n_chunks_seen": self.n_chunks_seen_,
        }
