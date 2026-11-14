"""Buffered decision-tree classification on streamed Titanic CSV rows.

Run from the repository root:
    python -m demo.tree_demo

Predictions are evaluated before training on each arriving chunk. The first
chunk(s) cannot be scored until the buffer has produced an initial tree.
"""

from pathlib import Path
import sys

import numpy as np

# Also works when running the script directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "NumCompute"))

from numcompute.io import iter_csv
from numcompute_stream.tree import DecisionTreeClassifier


def main() -> None:
    """Print one prediction accuracy per stream chunk after initial training."""
    dataset = ROOT / "demo" / "data" / "titanic.csv"
    tree = DecisionTreeClassifier(max_depth=5, buffer_size=200)
    correct = 0
    scored = 0
    for chunk_id, (header, rows) in enumerate(iter_csv(dataset, chunksize=100), 1):
        target_index = header.index("survived")
        feature_indices = [header.index(col) for col in ("pclass", "age", "sibsp", "parch", "fare")]
        X = np.asarray(rows[:, feature_indices], dtype=float)
        y = np.asarray(rows[:, target_index], dtype=int)
        if tree.n_retrains_:
            predictions = tree.predict(X)
            chunk_accuracy = float(np.mean(predictions == y))
            correct += int(np.sum(predictions == y))
            scored += len(y)
            metric = f"accuracy={chunk_accuracy:.3f}"
        else:
            metric = "accuracy=N/A (tree not trained yet)"
        before_retrains = tree.n_retrains_
        tree.partial_fit(X, y)
        retrained = tree.n_retrains_ > before_retrains
        print(
            f"chunk={chunk_id:02} count={len(y):3} {metric} "
            f"buffer={tree.n_buffered_samples_:3} retrained={retrained}"
        )
    print(f"Cumulative prequential accuracy: {correct / scored:.3f} ({scored} scored rows)")
    print(f"Total buffered retrains: {tree.n_retrains_}")


if __name__ == "__main__":
    main()
