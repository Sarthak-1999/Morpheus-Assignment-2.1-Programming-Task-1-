from __future__ import annotations

import argparse
from pathlib import Path

from . import benchmark_buffer_size, benchmark_vectorization
from ._common import DEFAULT_DATA, positive_int, write_environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--chunk-size", type=positive_int, default=50)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print("Experiment A: root split search on buffered CSV snapshots")
    splits = benchmark_vectorization.run(
        csv_path=args.csv, output_dir=args.output_dir, chunk_size=args.chunk_size,
        repeats=args.repeats, buffer_sizes=(100, 200, 400),
    )
    for item in splits:
        print(f"  n={item['buffer_size']}: "
              f"loop={item['median_loop_ms']:.3f}ms, "
              f"NumPy={item['median_vectorized_ms']:.3f}ms, "
              f"speedup={item['median_speedup']:.2f}x")
    print("\nExperiment B: complete streaming pipeline with different buffers")
    comparisons = benchmark_buffer_size.run(
        csv_path=args.csv, output_dir=args.output_dir, chunk_size=args.chunk_size,
        repeats=args.repeats, buffer_sizes=(50, 100, 200, 500), seed=args.seed,
    )
    for item in comparisons:
        print(f"  buffer={item['buffer_size']}, "
              f"matched accuracy={item['shared_accuracy']:.3f}, "
              f"retrain count={item['retrain_count']}, "
              f"median event update={item['median_retrain_update_ms']:.2f}ms")
    if args.csv.resolve() == DEFAULT_DATA.resolve():
        write_environment(args.output_dir / "environment.json", chunk_size=args.chunk_size,
                          split_buffer_sizes=(100, 200, 400),
                          model_buffer_sizes=(50, 100, 200, 500),
                          seed=args.seed, repeats=args.repeats)
    print(f"\nOutputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
