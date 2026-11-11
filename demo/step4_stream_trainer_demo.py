"""
Run from the repository root
    python -m demo.step4_stream_trainer_demo

The data is read by NumCompute's original custom CSV iterator through
numcompute_stream.io, not pandas. No external ML libraries are used.
"""

from pathlib import Path
import numpy as np

from numcompute_stream.io import iter_supervised_csv
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer


DATA = Path(__file__).resolve().parent / 'data' / 'titanic.csv'
FEATURES = ['pclass', 'age', 'sibsp', 'parch', 'fare']


def fmt(value: float) -> str:
    return 'N/A' if not np.isfinite(value) else f'{value:0.3f}'


def main() -> None:
    pipeline = Pipeline([
        ('impute', Imputer(strategy='mean')),
        ('scale', StandardScaler()),
        ('tree', DecisionTreeClassifier(max_depth=5, criterion='gini', buffer_size=200)),
    ])
    trainer = StreamTrainer(pipeline, positive_label='1', window_size=200)

    print('Assignment 2.2 / Step 4 / Titanic streaming demo')
    print('Evaluation: score BEFORE fitting each incoming chunk (no test leakage).')
    print('A buffer of 200 new samples triggers one replacement-tree retrain.\n')
    print('chunk  rows  acc    cumulative  rolling  AUC    buffered  retrains  update_ms')
    print('-' * 81)

    for X, y in iter_supervised_csv(DATA, target='survived',
                                    features=FEATURES, chunksize=100):
        row = trainer.fit_chunk(X, y)
        print(f"{row['chunk']:>5} {row['n_samples']:>5}  "
              f"{fmt(row['chunk_accuracy']):>5}   {fmt(row['cumulative_accuracy']):>8}   "
              f"{fmt(row['rolling_accuracy']):>6}  {fmt(row['auc']):>5}  "
              f"{row['buffered_samples']:>8}  {row['n_retrains']:>8}  "
              f"{row['update_ms']:>9.2f}")

    summary = trainer.result()
    print('\nSummary:')
    print('  all incoming samples:', summary['n_samples_seen'])
    print('  evaluated samples:', summary['support'])
    print('  warm-up samples (excluded from evaluation):', summary['n_warmup_samples'])
    print('  retraining events:', summary['n_retrains'])
    print('  final buffered samples (no forced final retrain):', summary['buffered_samples'])
    print('  cumulative accuracy:', fmt(summary['accuracy']))
    print('  precision / recall / F1:', *(fmt(summary[k]) for k in ('precision', 'recall', 'f1')))
    print('  AUC:', fmt(summary['auc']))
    print('  confusion matrix, rows=true columns=predicted:')
    print(summary['confusion_matrix'])
    print('  class order:', summary['labels'].tolist())


if __name__ == '__main__':
    main()
