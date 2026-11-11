"""
Run: python demo/lecturer_classification_tree_demo.py

This code implements and explains the learning rules
"""

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


FEATURE_NAMES = ("submitted", "attended", "study_group")
STUDENTS = np.array(
    [
        [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 0, 1],
        [1, 0, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0],
        [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0],
    ],
    dtype=int,
)
X, y = STUDENTS[:, :3], STUDENTS[:, 3]


def display_tree(node, level=0):
    indent = "  " * level
    if node.feature is None:
        print(f"{indent}Predict {'PASS' if node.prediction else 'FAIL'} "
              f"(counts={node.counts.tolist()})")
        return
    question = (
        f"{FEATURE_NAMES[node.feature]} == YES?"
        if node.binary_split
        else f"{FEATURE_NAMES[node.feature]} <= {node.threshold:g}?"
    )
    print(f"{indent}{question}")
    print(f"{indent}  YES / left:")
    display_tree(node.left, level + 2)
    print(f"{indent}  NO / right:")
    display_tree(node.right, level + 2)


def main():
    print("Lecturer's Week 10 classification tree: twelve students")
    print("Features: submitted, attended, study_group; target: passed")
    for criterion in ("gini", "entropy"):
        model = DecisionTreeClassifier(max_depth=2, criterion=criterion).fit(X, y)
        correct = int(np.count_nonzero(model.predict(X) == y))
        print(f"\n{criterion.upper()} — training accuracy {correct}/12")
        display_tree(model._tree)

    deep = DecisionTreeClassifier(max_depth=3).fit(X, y)
    print(f"\nDepth=3 learns the deliberate exception: "
          f"{int(np.sum(deep.predict(X) == y))}/12 training matches")
    print("Training matches are NOT a held-out evaluation measure.")

    print("\nAssignment 2.2 buffered partial_fit with buffer_size=12:")
    stream_model = DecisionTreeClassifier(max_depth=2, buffer_size=12)
    for chunk_i, start in enumerate(range(0, len(y), 4), start=1):
        stream_model.partial_fit(X[start:start+4], y[start:start+4])
        print(f"  chunk={chunk_i}, buffered={stream_model.n_buffered_samples_}, "
              f"retrains={stream_model.n_retrains_}")
    print("  final predictions:", stream_model.predict(X).tolist())


if __name__ == "__main__":
    main()
