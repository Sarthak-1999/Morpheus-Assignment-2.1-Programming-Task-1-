"""CSV streaming I/O: reuse NumCompute Assignment 2.1's custom CSV reader.

``iter_csv`` reads one chunk at a time and returns (header, 2D array).
``iter_supervised_csv`` extracts numeric features and one label column.
Only Python's standard library and NumPy are used by the I/O pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from numcompute.io import iter_csv, load_csv


def iter_supervised_csv(
    path: str | Path,
    *,
    target: str,
    features: Sequence[str],
    chunksize: int = 100,
    missing_strategy: str = "fill",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(X_chunk, y_chunk)`` from a CSV using the existing I/O reader.

    Features are converted to floats and may have NaNs. Labels are returned
    as strings to allow class names, with missing labels rejected. A feature
    may not duplicate the target. Duplicate/missing column names fail fast.
    """
    if not features or len(set(features)) != len(features) or target in features:
        raise ValueError("features must be nonempty, unique, and exclude the target.")
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer.")
    for header, chunk in iter_csv(
        path, chunksize=chunksize, missing_strategy=missing_strategy, dtype=str
    ):
        if len(set(header)) != len(header):
            raise ValueError("CSV header must contain unique column names.")
        names = list(features) + [target]
        absent = [name for name in names if name not in header]
        if absent:
            raise ValueError(f"CSV missing required columns: {absent}.")
        feature_idx = [header.index(name) for name in features]
        target_idx = header.index(target)
        try:
            X = chunk[:, feature_idx].astype(float)
        except ValueError as exc:
            raise ValueError("Selected CSV feature columns must be numeric or NaN.") from exc
        y = chunk[:, target_idx].copy()
        if np.isin(y, ["", "NaN", "nan", "None"]).any():
            raise ValueError("CSV target contains missing labels.")
        if np.isinf(X).any():
            raise ValueError("CSV selected features must not contain infinities.")
        yield X, y


__all__ = ["load_csv", "iter_csv", "iter_supervised_csv"]
