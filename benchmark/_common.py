"""Small standard-library helpers shared by the streaming benchmarks."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import numpy as np

from numcompute_stream.io import iter_supervised_csv

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "demo" / "data" / "titanic.csv"
DEFAULT_FEATURES = ("pclass", "age", "sibsp", "parch", "fare")


def positive_int(value: str) -> int:
    """CLI validator for strictly positive integers."""
    import argparse

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer >= 1") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return parsed


def write_csv(path: Path, records: list[dict], fieldnames: tuple[str, ...]) -> None:
    """Export pure-Python result records; never requires pandas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_environment(path: Path, *, chunk_size: int, split_buffer_sizes: tuple[int, ...],
                      model_buffer_sizes: tuple[int, ...], seed: int, repeats: int) -> None:
    """Keep the configuration with benchmark results for reproducibility."""
    import os
    from datetime import datetime, timezone

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "dataset": "demo/data/titanic.csv",
        "features": list(DEFAULT_FEATURES),
        "target": "survived",
        "chunk_size": chunk_size,
        "split_buffer_sizes": list(split_buffer_sizes),
        "model_buffer_sizes": list(model_buffer_sizes),
        "random_state": seed,
        "timing_repeats": repeats,
        "comparison_protocol": "same chronological chunks; test-before-train; shared post-warm-up cohort",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_chunks(path: Path = DEFAULT_DATA, *, chunk_size: int = 50):
    """Return ordered chunks using NumCompute's custom CSV reader."""
    return list(iter_supervised_csv(path, target="survived",
                                    features=DEFAULT_FEATURES,
                                    chunksize=chunk_size))


def buffer_snapshots(chunks, *, buffer_size: int):
    """Yield exactly one retraining batch each time a buffer reaches its limit.

    A threshold-crossing chunk is wholly included and the buffer is then
    cleared, precisely matching DecisionTreeClassifier.partial_fit().
    A final incomplete buffer is deliberately not trained.
    """
    if isinstance(buffer_size, bool) or not isinstance(buffer_size, int) or buffer_size < 1:
        raise ValueError("buffer_size must be a positive integer")
    x_pending, y_pending, n_pending = [], [], 0
    for x, y in chunks:
        x_pending.append(np.asarray(x))
        y_pending.append(np.asarray(y))
        n_pending += len(y)
        if n_pending >= buffer_size:
            yield np.concatenate(x_pending), np.concatenate(y_pending)
            x_pending, y_pending, n_pending = [], [], 0
