"""Python-loop vs. NumPy-vectorised DECISION-TREE SPLIT search in a stream.

This is an algorithmically matched root-split microbenchmark, NOT a claim
that the entire tree or end-to-end stream runs at the plotted speedup.
Both methods sort each feature, update cumulative class counts, and evaluate
all valid split positions, including splitting observed values from NaNs.
The NumPy path calls the production DecisionTreeClassifier._best_split;
the baseline uses Python loops for the same candidate/impurity calculation.

Run from the repository root:
    python -m benchmark.benchmark_vectorization
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numcompute.benchmarking import benchmark as measure
from numcompute_stream.tree import DecisionTreeClassifier

from ._common import (DEFAULT_DATA, buffer_snapshots, positive_int, read_chunks,
                      write_csv)


FIELDS = ("buffer_size", "retrain_event", "samples", "features", "criterion",
          "loop_ms", "vectorized_ms", "speedup", "same_split")
SUMMARY_FIELDS = ("buffer_size", "snapshots", "median_samples", "median_loop_ms",
                  "median_vectorized_ms", "median_speedup")


def _impurity_loop(counts: list[int], criterion: str) -> float:
    """Impurity using only Python numerical operations."""
    n = sum(counts)
    if n == 0:
        return 0.0
    if criterion == "gini":
        return 1.0 - sum((count / n) ** 2 for count in counts)
    if criterion == "entropy":
        return -sum((count / n) * math.log2(count / n)
                    for count in counts if count > 0)
    raise ValueError("criterion must be 'gini' or 'entropy'")


def best_split_loop(X: np.ndarray, y_encoded: np.ndarray,
                    *, n_classes: int, criterion: str = "gini"):
    """Reference split finder using Python sorted() and running class counts.

    X: (n_samples,n_features); y_encoded: 1D integer labels in [0,n_classes).
    Returns the production split tuple (feature, threshold, binary_split),
    or None if no positive gain exists. NaNs are routed RIGHT.
    """
    data = np.asarray(X, dtype=float)
    labels = np.asarray(y_encoded)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional feature matrix")
    if labels.ndim != 1 or len(labels) != len(data):
        raise ValueError("y_encoded must be one-dimensional and match X rows")
    if not isinstance(n_classes, int) or n_classes < 1:
        raise ValueError("n_classes must be a positive integer")
    if not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any() or (labels >= n_classes).any():
        raise ValueError("y_encoded must contain integer class indices")
    if np.isinf(data).any():
        raise ValueError("X must not contain infinity")
    totals = [0] * n_classes
    for label in labels:
        totals[int(label)] += 1
    before = _impurity_loop(totals, criterion)
    gain_best = 1e-12
    best = None
    n_samples = len(labels)

    for feature in range(data.shape[1]):
        observed = sorted(
            [(float(value), int(label)) for value, label in zip(data[:, feature], labels)
             if not math.isnan(float(value))],
            key=lambda pair: pair[0],
        )
        n_observed = len(observed)
        if n_observed == 0:
            continue
        binary_split = (n_observed == n_samples and observed[0][0] == 0.0
                        and observed[-1][0] == 1.0)
        left = [0] * n_classes
        for i, (value, label) in enumerate(observed):
            left[label] += 1
            if i + 1 < n_observed and value == observed[i + 1][0]:
                continue  # A threshold cannot split equal feature values.
            n_left = i + 1
            n_right = n_samples - n_left
            if n_right <= 0:
                continue
            right = [total - count for total, count in zip(totals, left)]
            after = (n_left * _impurity_loop(left, criterion)
                     + n_right * _impurity_loop(right, criterion)) / n_samples
            gain = before - after
            if gain > gain_best:
                threshold = (value * 0.5 + observed[i + 1][0] * 0.5
                             if i + 1 < n_observed else value)
                if i + 1 < n_observed and threshold >= observed[i + 1][0]:
                    threshold = value
                best = (feature, float(threshold), binary_split)
                gain_best = gain
    return best


def _production_splitter(n_classes: int, criterion: str):
    """Return the production vectorised split search with fixed model settings."""
    tree = DecisionTreeClassifier(criterion=criterion, random_state=0)
    tree.classes_ = np.arange(n_classes)
    tree._rng = np.random.default_rng(0)

    def split(X: np.ndarray, y_encoded: np.ndarray):
        counts = np.bincount(y_encoded, minlength=n_classes)
        parent = float(tree._impurity(counts))
        return tree._best_split(X, y_encoded, parent)

    return split


def _same_split(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    feature_a, threshold_a, binary_a = a
    feature_b, threshold_b, binary_b = b
    return (feature_a == feature_b and binary_a == binary_b
            and bool(np.isclose(threshold_a, threshold_b, rtol=1e-12, atol=1e-12)))


def split_comparison(chunks, *, buffer_sizes=(100, 200, 400),
                     repeats=5, warmup=1, criterion="gini"):
    """Compare both split implementations on identical streamed retrain batches."""
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >=1 and warmup >=0")
    if criterion not in ("gini", "entropy"):
        raise ValueError("criterion must be gini or entropy")
    rows: list[dict] = []
    for buffer_size in buffer_sizes:
        for event, (X, y) in enumerate(buffer_snapshots(chunks, buffer_size=int(buffer_size)), 1):
            classes, encoded = np.unique(y, return_inverse=True)
            encoded = encoded.astype(np.int64, copy=False)
            loop = lambda: best_split_loop(X, encoded, n_classes=len(classes), criterion=criterion)
            vector = _production_splitter(len(classes), criterion)
            fast = lambda: vector(X, encoded)
            if not _same_split(loop(), fast()):
                raise AssertionError(f"Split mismatch on buffer_size={buffer_size}, event={event}.")
            slow_ms = float(measure(loop, repeats=repeats, warmup=warmup)["median_seconds"]) * 1000
            fast_ms = float(measure(fast, repeats=repeats, warmup=warmup)["median_seconds"]) * 1000
            rows.append({
                "buffer_size": int(buffer_size), "retrain_event": event,
                "samples": len(y), "features": X.shape[1], "criterion": criterion,
                "loop_ms": round(slow_ms, 6), "vectorized_ms": round(fast_ms, 6),
                "speedup": round(slow_ms / fast_ms, 4) if fast_ms > 0 else float("nan"),
                "same_split": True,
            })
    if not rows:
        raise ValueError("There are no complete buffered retraining batches to compare")
    return rows


def summarise_split(rows):
    """Summarise per-retrain root split timings by buffer size."""
    result = []
    for size in sorted(set(row["buffer_size"] for row in rows)):
        selected = [row for row in rows if row["buffer_size"] == size]
        loop_ms = median(row["loop_ms"] for row in selected)
        vector_ms = median(row["vectorized_ms"] for row in selected)
        result.append({"buffer_size": size, "snapshots": len(selected),
                       "median_samples": median(row["samples"] for row in selected),
                       "median_loop_ms": round(loop_ms, 6),
                       "median_vectorized_ms": round(vector_ms, 6),
                       "median_speedup": round(loop_ms / vector_ms, 4)})
    return result


def save_split_plot(summary: list[dict], path: Path):
    """Save runtime chart; distinct point series mean loop and vectorised timings."""
    fig, ax = plt.subplots(figsize=(8, 4.7))
    sizes = np.array([row["buffer_size"] for row in summary], dtype=float)
    ax.plot(sizes, [row["median_loop_ms"] for row in summary], "o-", label="Python loops")
    ax.plot(sizes, [row["median_vectorized_ms"] for row in summary], "s-", label="NumPy vectorised")
    ax.set(xlabel="Buffered samples per retrain", ylabel="Median root split time (ms)",
           title="Split search: loops vs vectorised NumPy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run(*, csv_path: Path = DEFAULT_DATA, output_dir: Path = Path("benchmark/results"),
        chunk_size=50, buffer_sizes=(100, 200, 400), repeats=5, warmup=1,
        criterion="gini"):
    chunks = read_chunks(Path(csv_path), chunk_size=chunk_size)
    rows = split_comparison(chunks, buffer_sizes=buffer_sizes,
                            repeats=repeats, warmup=warmup, criterion=criterion)
    summary = summarise_split(rows)
    output = Path(output_dir)
    write_csv(output / "split_comparison.csv", rows, FIELDS)
    write_csv(output / "split_summary.csv", summary, SUMMARY_FIELDS)
    save_split_plot(summary, output / "split_search_runtime.png")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--chunk-size", type=positive_int, default=50)
    parser.add_argument("--buffer-sizes", nargs="+", type=positive_int, default=[100, 200, 400])
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--criterion", choices=("gini", "entropy"), default="gini")
    args = parser.parse_args(argv)
    summary = run(csv_path=args.csv, output_dir=args.output_dir,
                  chunk_size=args.chunk_size, buffer_sizes=tuple(args.buffer_sizes),
                  repeats=args.repeats, warmup=args.warmup, criterion=args.criterion)
    print("\nPython loops vs NumPy vectorised root split search (identical snapshots):")
    for row in summary:
        print(f"  buffer={row['buffer_size']:>4}, snapshots={row['snapshots']:>2}, "
              f"loop={row['median_loop_ms']:.3f}ms, NumPy={row['median_vectorized_ms']:.3f}ms, "
              f"speedup={row['median_speedup']:.2f}x")
    print(f"Saved CSVs and PNG to {args.output_dir}")


if __name__ == "__main__":
    main()
