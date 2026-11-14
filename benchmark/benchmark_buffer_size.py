"""Streamed tree accuracy / retraining-cost comparison for buffer_size values.

Each candidate sees exactly the same chronologically ordered 50-row chunks.
A chunk is predicted BEFORE training. All reported comparative accuracies
are computed on exactly the same suffix of chunks, beginning only after the
largest buffer has trained at least one tree. Smaller buffers' earlier scores
are not used for comparing accuracies, preventing warm-up selection bias.

`retrain_update_ms` times the threshold-crossing `.partial_fit()` calls:
preprocessing, buffered transformation and tree fit are included. It is
not the isolated decision-tree fit time. Each size has ONE decision tree.

Run from the repository root:
    python -m benchmark.benchmark_buffer_size
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier

from ._common import DEFAULT_DATA, positive_int, read_chunks, write_csv

SUMMARY_FIELDS = ("buffer_size", "chunk_size", "stream_samples", "shared_test_samples",
                  "shared_correct", "shared_accuracy", "prequential_accuracy_all_available",
                  "warmup_samples", "retrain_count", "buffered_samples_at_end",
                  "median_retrain_update_ms", "total_retrain_update_ms",
                  "total_update_ms", "peak_estimated_memory_kb", "repeats")
CHUNK_FIELDS = ("buffer_size", "chunk", "samples", "shared_evaluated",
                "shared_chunk_accuracy", "retrain_count", "buffered_samples")


def common_test_start(chunks, buffer_sizes) -> int:
    """Find the first chunk AFTER all buffer configurations first train.

    Return a zero-based chunk index. No model sees a test chunk before it
    has been fitted, and all compared models receive identical test rows.
    """
    if not chunks:
        raise ValueError("The streaming dataset must contain at least one chunk")
    if not buffer_sizes or any(isinstance(size, bool) or int(size) != size or size < 1
                               for size in buffer_sizes):
        raise ValueError("buffer_sizes must be positive integer thresholds")
    largest = max(buffer_sizes)
    seen = 0
    for index, (_, labels) in enumerate(chunks):
        if seen >= largest:
            return index
        seen += len(labels)
    raise ValueError("Not enough data to evaluate after the largest buffer has first trained")


def _make_trainer(size: int, *, depth: int, seed: int):
    tree = DecisionTreeClassifier(max_depth=depth, buffer_size=size, random_state=seed,
                                  criterion="gini")
    pipeline = Pipeline([
        ("impute", Imputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("tree", tree),
    ])
    return StreamTrainer(pipeline, positive_label="1", window_size=200)


def _one_run(chunks, *, buffer_size: int, first_shared_chunk: int,
             depth: int = 5, seed: int = 42) -> dict:
    trainer = _make_trainer(buffer_size, depth=depth, seed=seed)
    correct = 0
    n_eval = 0
    event_times = []
    chunk_history = []
    old_retrain_count = 0
    for index, (x, y) in enumerate(chunks):
        shared = index >= first_shared_chunk
        accuracy = float("nan")
        if shared:
            if not trainer.model.is_fitted:
                raise AssertionError("Shared evaluation started before this tree was fitted")
            # Score before the incoming chunk can influence this model.
            pred = trainer.model.predict(x)
            chunk_correct = int(np.count_nonzero(pred == y))
            correct += chunk_correct
            n_eval += len(y)
            accuracy = chunk_correct / len(y)
        row = trainer.fit_chunk(x, y)
        if row["n_retrains"] > old_retrain_count:
            # Includes imputation, scaling, buffered assembly and tree fit.
            event_times.append(float(row["update_ms"]))
            old_retrain_count = row["n_retrains"]
        chunk_history.append({
            "buffer_size": buffer_size, "chunk": index + 1, "samples": len(y),
            "shared_evaluated": bool(shared), "shared_chunk_accuracy": accuracy,
            "retrain_count": row["n_retrains"], "buffered_samples": row["buffered_samples"],
        })
    if n_eval == 0 or not event_times:
        raise AssertionError("Experiment requires fitted tree and post-warmup test rows")
    history = trainer.history_
    return {
        "shared_test_samples": n_eval,
        "shared_correct": correct,
        "shared_accuracy": correct / n_eval,
        "prequential_accuracy_all_available": trainer.result()["accuracy"],
        "warmup_samples": trainer.n_warmup_samples_,
        "retrain_count": trainer.model.n_retrains_,
        "buffered_samples_at_end": trainer.model.n_buffered_samples_,
        "median_retrain_update_ms": median(event_times),
        "total_retrain_update_ms": sum(event_times),
        "total_update_ms": sum(float(row["update_ms"]) for row in history),
        "peak_estimated_memory_kb": max(row["memory_bytes"] for row in history) / 1024,
        "chunk_history": chunk_history,
    }


def compare_buffers(chunks, *, buffer_sizes=(50, 100, 200, 500),
                    repeats=3, depth=5, seed=42):
    """Run independent fixed-seed trials for each buffer size, mediate timings.

    Timing columns are medians of per-trial quantities. Accuracy is checked
    for exact repeatability (no random data splits and fixed random seed).
    """
    if repeats < 1 or depth < 0:
        raise ValueError("repeats must be >=1 and depth >=0")
    sizes = tuple(buffer_sizes)
    if len(set(sizes)) != len(sizes):
        raise ValueError("buffer_sizes must be unique")
    first_shared_chunk = common_test_start(chunks, sizes)
    total = sum(len(y) for _, y in chunks)
    chunk_size = len(chunks[0][1])
    results, chunk_records = [], []
    for size in sizes:
        trials = [_one_run(chunks, buffer_size=size, first_shared_chunk=first_shared_chunk,
                           depth=depth, seed=seed) for _ in range(repeats)]
        baseline = trials[0]
        if any(trial["shared_correct"] != baseline["shared_correct"]
               or trial["shared_test_samples"] != baseline["shared_test_samples"]
               or trial["retrain_count"] != baseline["retrain_count"]
               for trial in trials[1:]):
            raise AssertionError("Fixed-seed trials differed in accuracy or retrain count")
        fields = ("median_retrain_update_ms", "total_retrain_update_ms",
                  "total_update_ms", "peak_estimated_memory_kb")
        aggregated = {key: round(median(trial[key] for trial in trials), 4)
                      for key in fields}
        aggregated.update({
            "buffer_size": size,
            "chunk_size": chunk_size,
            "stream_samples": total,
            "shared_test_samples": baseline["shared_test_samples"],
            "shared_correct": baseline["shared_correct"],
            "shared_accuracy": round(baseline["shared_accuracy"], 6),
            "prequential_accuracy_all_available": round(baseline["prequential_accuracy_all_available"], 6),
            "warmup_samples": baseline["warmup_samples"],
            "retrain_count": baseline["retrain_count"],
            "buffered_samples_at_end": baseline["buffered_samples_at_end"],
            "repeats": repeats,
        })
        results.append(aggregated)
        chunk_records.extend(baseline["chunk_history"])
    return results, chunk_records


def save_accuracy_plot(summary, output: Path):
    """Accuracy comparison across buffer sizes, all evaluated on one cohort."""
    summary = sorted(summary, key=lambda row: row["buffer_size"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    positions = [str(r["buffer_size"]) for r in summary]
    accuracies = [100 * r["shared_accuracy"] for r in summary]
    bars = ax.bar(positions, accuracies)
    for bar, value in zip(bars, accuracies):
        ax.annotate(f"{value:.1f}%", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 4), ha="center")
    ax.set(xlabel="Buffer size (samples)", ylabel="Accuracy on common test chunks (%)",
           title="Streaming prequential accuracy on matched future samples")
    ax.set_ylim(0, max(100, max(accuracies) + 8))
    ax.grid(axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_retrain_plot(summary, output: Path):
    """Retrain frequency plus measured threshold-crossing update cost."""
    summary = sorted(summary, key=lambda row: row["buffer_size"])
    sizes = [row["buffer_size"] for row in summary]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(sizes, [row["retrain_count"] for row in summary], "o-",
            label="Retraining events")
    ax.set(xlabel="Buffer size (samples)", ylabel="Retraining events",
           title="Streaming retrain frequency vs buffer size")
    ax.grid(True, alpha=0.25)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_retrain_cost_plot(summary, output: Path):
    """Total measured retraining-event update time over the same stream."""
    summary = sorted(summary, key=lambda row: row["buffer_size"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot([row["buffer_size"] for row in summary],
            [row["total_retrain_update_ms"] for row in summary], "s-")
    ax.set(xlabel="Buffer size (samples)",
           ylabel="Total threshold-crossing update time (ms)",
           title="Streaming buffered retraining work")
    ax.grid(True, alpha=0.25)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run(*, csv_path: Path = DEFAULT_DATA, output_dir: Path = Path("benchmark/results"),
        chunk_size=50, buffer_sizes=(50, 100, 200, 500), repeats=3,
        depth=5, seed=42):
    chunks = read_chunks(Path(csv_path), chunk_size=chunk_size)
    summary, traces = compare_buffers(chunks, buffer_sizes=buffer_sizes,
                                      repeats=repeats, depth=depth, seed=seed)
    output = Path(output_dir)
    write_csv(output / "buffer_summary.csv", summary, SUMMARY_FIELDS)
    write_csv(output / "buffer_chunk_history.csv", traces, CHUNK_FIELDS)
    save_accuracy_plot(summary, output / "buffer_accuracy.png")
    save_retrain_plot(summary, output / "buffer_retrain_count.png")
    save_retrain_cost_plot(summary, output / "buffer_retrain_cost.png")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--chunk-size", type=positive_int, default=50)
    parser.add_argument("--buffer-sizes", nargs="+", type=positive_int,
                        default=[50, 100, 200, 500])
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    summary = run(csv_path=args.csv, output_dir=args.output_dir,
                  chunk_size=args.chunk_size, buffer_sizes=tuple(args.buffer_sizes),
                  repeats=args.repeats, depth=args.max_depth, seed=args.seed)
    print("\nBuffer size vs matched future-sample accuracy and retraining:")
    print("  size  shared_n  accuracy  retrains  last_buffer  median_event_ms  total_event_ms")
    for row in summary:
        print(f"  {row['buffer_size']:>4}  {row['shared_test_samples']:>8}  "
              f"{row['shared_accuracy']:>8.3f}  {row['retrain_count']:>8}  "
              f"{row['buffered_samples_at_end']:>11}  "
              f"{row['median_retrain_update_ms']:>15.2f}  "
              f"{row['total_retrain_update_ms']:>14.2f}")
    print(f"Saved CSVs and PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
