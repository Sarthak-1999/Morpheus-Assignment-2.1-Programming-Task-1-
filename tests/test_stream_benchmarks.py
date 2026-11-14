"""Benchmark correctness checks; performance thresholds are intentionally absent.

Wall time is hardware-dependent: tests compare algorithms and experimental
protocols, never assert that NumPy must be faster on a particular computer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from benchmark._common import buffer_snapshots, positive_int, read_chunks, write_csv, write_environment
from benchmark.benchmark_vectorization import (
    _impurity_loop, _production_splitter, _same_split,
    best_split_loop, split_comparison, summarise_split, save_split_plot, run as run_split,
)
from benchmark.benchmark_buffer_size import (
    common_test_start, compare_buffers, _one_run, save_accuracy_plot,
    save_retrain_plot, save_retrain_cost_plot, run as run_buffers,
)
from benchmark.run_benchmarks import main as run_all


def toy_chunks(n: int = 16, chunk_size: int = 2):
    x = np.arange(n, dtype=float)
    features = np.column_stack([x, (x % 3) / 3.0])
    y = np.where(x >= n // 2, "1", "0")
    return [(features[i:i+chunk_size], y[i:i+chunk_size])
            for i in range(0, n, chunk_size)]


def check_splits_match(X, y, criterion="gini"):
    classes, yi = np.unique(np.asarray(y), return_inverse=True)
    yi = yi.astype(int)
    loop = best_split_loop(np.asarray(X, float), yi, n_classes=len(classes), criterion=criterion)
    vectorized = _production_splitter(len(classes), criterion)(np.asarray(X, float), yi)
    assert _same_split(loop, vectorized)
    return loop


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_exact_binary_lecture_data(criterion):
    x = np.array([[1,1,1],[1,1,1],[1,1,0],[1,1,0],[1,0,1],[1,0,0],
                  [1,0,0],[0,1,1],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])
    y = np.array([1,1,1,1,1,0,0,0,0,0,0,0])
    split = check_splits_match(x, y, criterion)
    assert split[0] == 0
    assert split[2] is True


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_split_with_nan_values(criterion):
    X = [[0.0, 0.0], [1.0, np.nan], [2.0, 1.0], [np.nan, 0.0], [3.0, 1.0]]
    y = ["0", "1", "1", "0", "1"]
    assert check_splits_match(X, y, criterion) is not None


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_zero_variance_all_features(criterion):
    assert check_splits_match(np.ones((9, 3)), [0,1,0,1,0,1,0,1,0], criterion) is None


def test_uninformative_labels_stay_leaf():
    assert check_splits_match([[0],[1],[2],[3]], [1,1,1,1]) is None


def test_multiclass_consistency():
    assert check_splits_match([[0, 1],[1, 1],[2, 0],[3, 0],[4, 1],[5, 0]],
                              ["a","a","b","b","c","c"]) is not None


def test_all_missing_column_is_skipped():
    assert check_splits_match([[np.nan,0],[np.nan,1],[np.nan,2],[np.nan,3]],
                              [0,0,1,1]) is not None


def test_missing_vs_observed_split_supported():
    split = check_splits_match([[1.0],[1.0],[np.nan],[np.nan]], [0,0,1,1])
    assert split == (0, 1.0, False)


def test_invalid_labels_fail():
    with pytest.raises(ValueError, match="integer class indices"):
        best_split_loop(np.array([[0],[1]], float), np.array([0,4]), n_classes=2)


def test_invalid_criterion_fails():
    with pytest.raises(ValueError, match="criterion"):
        best_split_loop(np.array([[0],[1]], float), np.array([0,1]), n_classes=2,
                        criterion="unexpected")


def test_invalid_data_dimensions_fail():
    with pytest.raises(ValueError, match="two-dimensional"):
        best_split_loop(np.array([0,1]), np.array([0,1]), n_classes=2)


def test_split_mismatch_detector():
    assert _same_split(None, None)
    assert not _same_split(None, (0, 0.5, False))
    assert not _same_split((0, 0.5, False), (1, 0.5, False))
    assert not _same_split((0, 0.5, False), (0, 0.5, True))


def test_gini_entropy_purity_and_balanced():
    assert _impurity_loop([3, 0], "gini") == pytest.approx(0.0)
    assert _impurity_loop([2, 2], "gini") == pytest.approx(0.5)
    assert _impurity_loop([2, 2], "entropy") == pytest.approx(1.0)
    assert _impurity_loop([0, 0], "entropy") == 0.0


def test_stream_snapshot_clearing_overshoot():
    chunks = toy_chunks(n=9, chunk_size=2)
    batches = list(buffer_snapshots(chunks, buffer_size=3))
    assert [len(y) for x, y in batches] == [4, 4]
    assert np.array_equal(batches[0][1], ["0", "0", "0", "0"])
    # Last unmatched sample is intentionally left buffered, not trained.


def test_stream_snapshot_one_larger_chunk():
    chunks = toy_chunks(n=8, chunk_size=4)
    batches = list(buffer_snapshots(chunks, buffer_size=3))
    assert len(batches) == 2
    assert [len(y) for x, y in batches] == [4,4]


def test_stream_snapshot_rejects_non_positive_buffer():
    with pytest.raises(ValueError):
        list(buffer_snapshots(toy_chunks(), buffer_size=0))


def test_common_cohort_starts_after_largest_fit():
    assert common_test_start(toy_chunks(n=12, chunk_size=2), [2, 4, 6]) == 3


def test_common_cohort_invalid_horizon_fails():
    with pytest.raises(ValueError, match="Not enough data"):
        common_test_start(toy_chunks(n=4, chunk_size=2), [10])


def test_common_cohort_empty_fails():
    with pytest.raises(ValueError, match="at least one"):
        common_test_start([], [2])


def test_common_cohort_bool_buffer_fails():
    with pytest.raises(ValueError, match="positive integer"):
        common_test_start(toy_chunks(), [True])


def test_compare_buffers_same_shared_samples():
    summary, traces = compare_buffers(toy_chunks(n=16, chunk_size=2),
                                      buffer_sizes=(2,4,6), repeats=2, seed=8)
    assert {r["shared_test_samples"] for r in summary} == {10}
    assert [r["retrain_count"] for r in summary] == [8,4,2]
    assert len(traces) == 3 * 8
    assert all(0 <= r["shared_accuracy"] <= 1 for r in summary)
    assert all(r["buffered_samples_at_end"] < r["buffer_size"] for r in summary)
    assert all(r["shared_correct"] <= r["shared_test_samples"] for r in summary)


def test_compare_buffers_repeated_determinism():
    chunks = toy_chunks(n=16, chunk_size=2)
    a, _ = compare_buffers(chunks, buffer_sizes=(2, 4), repeats=1)
    b, _ = compare_buffers(chunks, buffer_sizes=(2, 4), repeats=1)
    assert [r["shared_accuracy"] for r in a] == [r["shared_accuracy"] for r in b]


def test_buffer_comparison_rejects_duplicate_sizes():
    with pytest.raises(ValueError, match="unique"):
        compare_buffers(toy_chunks(), buffer_sizes=(2,2))


def test_buffer_comparison_rejects_negative_depth():
    with pytest.raises(ValueError, match="depth"):
        compare_buffers(toy_chunks(), buffer_sizes=(2,4), depth=-1)


def test_buffer_result_reports_correct_end_buffer():
    trial = _one_run(toy_chunks(n=11, chunk_size=2), buffer_size=4,
                     first_shared_chunk=2)
    assert trial["retrain_count"] == 2
    assert trial["buffered_samples_at_end"] == 3
    assert trial["shared_test_samples"] == 7


def test_split_benchmark_records_equal_splits():
    rows = split_comparison(toy_chunks(n=16, chunk_size=2),
                            buffer_sizes=(4,8), repeats=1, warmup=0)
    assert len(rows) == 6
    assert all(row["same_split"] for row in rows)
    assert all(row["loop_ms"] > 0 and row["vectorized_ms"] > 0 for row in rows)


def test_split_benchmark_no_complete_buffer():
    with pytest.raises(ValueError, match="no complete"):
        split_comparison(toy_chunks(n=8), buffer_sizes=(100,),
                         repeats=1, warmup=0)


def test_split_summary_uses_median():
    fake = [dict(buffer_size=100, samples=100, loop_ms=3., vectorized_ms=1.),
            dict(buffer_size=100, samples=100, loop_ms=9., vectorized_ms=3.)]
    out = summarise_split(fake)
    assert len(out) == 1
    assert out[0]["median_loop_ms"] == 6
    assert out[0]["median_vectorized_ms"] == 2
    assert out[0]["median_speedup"] == 3


def test_csv_writer_preserves_header(tmp_path):
    path = tmp_path / "rows.csv"
    write_csv(path, [{"count": 1, "value": 2.5}], ("count", "value"))
    with path.open(newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == [{"count": "1", "value": "2.5"}]


def test_positive_integer_parser():
    from argparse import ArgumentTypeError
    assert positive_int("7") == 7
    with pytest.raises(ArgumentTypeError):
        positive_int("0")


def test_environment_json_written(tmp_path):
    path = tmp_path / "env.json"
    write_environment(path, chunk_size=50, split_buffer_sizes=(100,200),
                      model_buffer_sizes=(50,100), seed=42, repeats=2)
    obj = json.loads(path.read_text())
    assert obj["chunk_size"] == 50
    assert obj["python_version"]
    assert obj["comparison_protocol"].startswith("same chronological")


def test_plot_helpers_save_non_empty_pngs(tmp_path):
    summaries = [{"buffer_size": 2, "shared_accuracy": .8, "retrain_count": 5,
                  "total_retrain_update_ms": 4.3},
                 {"buffer_size": 4, "shared_accuracy": .7, "retrain_count": 3,
                  "total_retrain_update_ms": 3.0}]
    save_accuracy_plot(summaries, tmp_path / "a.png")
    save_retrain_plot(summaries, tmp_path / "b.png")
    save_retrain_cost_plot(summaries, tmp_path / "c.png")
    assert all((tmp_path / p).stat().st_size > 500 for p in ("a.png","b.png","c.png"))


def test_split_plot_saves_png(tmp_path):
    save_split_plot([{"buffer_size": 100, "median_loop_ms": 2,
                      "median_vectorized_ms": 1}], tmp_path / "split.png")
    assert (tmp_path / "split.png").stat().st_size > 500


def test_read_default_csv_chunks():
    chunks = read_chunks(chunk_size=200)
    assert sum(len(y) for _, y in chunks) == 891
    assert chunks[0][0].shape[1] == 5


def test_end_to_end_benchmark_creates_results(tmp_path):
    # Exercise the same command a student will run in VS Code.
    run_all(["--repeats", "1", "--output-dir", str(tmp_path), "--chunk-size", "50"])
    expected = ("split_comparison.csv", "split_summary.csv", "split_search_runtime.png",
                "buffer_summary.csv", "buffer_chunk_history.csv", "buffer_accuracy.png",
                "buffer_retrain_count.png", "buffer_retrain_cost.png", "environment.json")
    assert all((tmp_path / name).stat().st_size > 0 for name in expected)
    with (tmp_path / "buffer_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["buffer_size"]) for r in rows] == [50,100,200,500]
    assert {int(r["shared_test_samples"]) for r in rows} == {391}
