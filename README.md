# NumCompute — Streaming Decision Tree Framework

A streaming classification framework built from scratch with **Python, NumPy, and Matplotlib**. This project extends the original NumCompute numerical-computing package with incremental preprocessing, a buffered decision tree classifier, streaming evaluation, visualisation, and reproducible performance benchmarks.

The original group package remains in `NumCompute/numcompute/`; the streaming extension is implemented separately in `numcompute_stream/`.

## Features

- **Decision tree classifier:** Gini or entropy-based splitting, configurable maximum depth, minimum split size, feature selection, and class probabilities.
- **Buffered retraining:** `partial_fit(X_chunk, y_chunk)` collects incoming samples and replaces the trained tree whenever the buffer reaches `buffer_size`.
- **Incremental preprocessing:** Running statistics, scaling, missing-value imputation, and categorical one-hot encoding.
- **Streaming pipeline:** Chained preprocessing and model training with a consistent feature representation between retraining events.
- **Evaluation:** Per-chunk and cumulative accuracy, error, precision, recall, F1-score, confusion matrix, ROC AUC, and rolling-window metrics.
- **Visualisation:** Reusable Matplotlib plots for metrics over time, model comparisons, predictions versus actual labels, and a confusion matrix heatmap.
- **Testing and benchmarking:** Automated unit/integration tests and controlled performance experiments.

## Project structure

```text
NumCompute/numcompute/       Original group package
numcompute_stream/
    tree.py                 DecisionTreeClassifier
    preprocessing.py        Incremental transformers
    stats.py                Streaming statistics
    io.py                   CSV input and chunk iteration
    metrics.py              Classification metrics
    pipeline.py             Incremental model pipeline
    stream.py               StreamTrainer and logging
    visualise.py            Plotting utilities, including confusion matrix
tests/                      Unit and integration tests
demo/
    data/titanic.csv        Example classification dataset
    stream_demo.ipynb       Main streaming demonstration
    stream_trainer_demo.py    CSV streaming and metric logging
    classification_tree_demo.py  Binary-feature tree example
    streaming_data_demo.py   Incremental preprocessing example
    tree_demo.py             Buffered tree training example
benchmark/
    benchmark_vectorization.py
    benchmark_buffer_size.py
    run_benchmarks.py
    results/                 Benchmark CSV files and figures
pyproject.toml
requirements.txt
requirements-dev.txt
README.md
```

## Installation

Use Python **3.10 or later**. From the repository root (the folder containing `pyproject.toml`), run:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

NumCompute's algorithm implementations use standard Python, NumPy, and Matplotlib; `pytest` is used for testing. Jupyter tooling is needed only to run the notebook.

## Streaming classification example

```python
from numcompute_stream import (
    DecisionTreeClassifier,
    Imputer,
    Pipeline,
    StandardScaler,
    StreamTrainer,
)
from numcompute_stream.io import iter_supervised_csv
pipeline = Pipeline([
    ("impute", Imputer(strategy="mean")),
    ("scale", StandardScaler()),
    ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=200)),
])
trainer = StreamTrainer(pipeline, positive_label="1", window_size=200)
for X, y in iter_supervised_csv(
    "demo/data/titanic.csv",
    target="survived",
    features=["pclass", "age", "sibsp", "parch", "fare"],
    chunksize=100,
):
    log = trainer.fit_chunk(X, y)
    print(log["chunk"], log["chunk_accuracy"], log["n_retrains"])
results = trainer.result()
print(results["confusion_matrix"])
```

### Buffered retraining and evaluation

For `buffer_size=200`, two 100-sample chunks trigger one retraining event. Between retraining events, predictions use the most recently trained tree; once retraining occurs, the data buffer is cleared. If the stream ends with fewer than 200 buffered samples, they remain buffered rather than triggering a forced fit.

`StreamTrainer.fit_chunk(X, y)` follows **test-before-train (prequential) evaluation**: it scores an incoming chunk before using the chunk to update the pipeline. Before the initial model fit, predictions are unavailable, so warm-up chunks are excluded from evaluation. The pipeline keeps a fixed inference-time preprocessing representation until the next retrain, preventing online scaler or encoder updates from changing inputs to an older tree.

For evaluation without updating the model, use `StreamTrainer.score_chunk(X, y)`.

## Jupyter demonstration and visualisation

Open `demo/stream_demo.ipynb` in a Jupyter-compatible environment or VS Code with the Python and Jupyter extensions, select the project interpreter, and choose **Run All**.

The notebook reads the Titanic CSV through the custom I/O module, simulates incoming chunks, trains the streaming pipeline, logs evaluation metrics, and compares separate single-tree models with different buffer sizes. It includes accuracy/error curves, AUC, a streaming dashboard, predictions versus ground truth, and a coloured cumulative confusion matrix heatmap.

Reusable plotting functions are available in `numcompute_stream.visualise`:

```python
from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
    plot_stream_dashboard,
    plot_confusion_matrix,
)
```

Plots can be shown in a notebook or saved as image files.

## Testing

Run the complete test suite from the repository root:

```bash
python -m pytest -q
```

The latest version passed **534 tests** in the verified local test run, covering both original NumCompute functionality and the streaming extension. Test cases include empty or malformed inputs, missing values, constant features, class ties, buffer thresholds and oversized chunks, evolving one-hot categories, metrics accumulation, confusion matrix plotting, and CSV-to-model integration.

The tree demonstrations can also be run directly:

```bash
python -m demo.classification_tree_demo
python -m demo.stream_trainer_demo
```

## Performance benchmarks

Reproduce both experiments with:

```bash
python -m benchmark.run_benchmarks --repeats 5
```

The benchmark scripts use the Titanic CSV in chronological 50-row chunks and save their tables, runtime metadata, and figures under `benchmark/results/`.

### Loop-based versus NumPy-vectorised split search

Both implementations calculate the same root-node split on matching buffered data. The timing covers **split search**, not a complete tree fit. Medians from a five-repeat run on Windows 11:

| Buffered samples | Python loops | NumPy vectorised | Speedup |
| ---------------: | -----------: | ---------------: | ------: |
|              100 |     0.750 ms |         0.915 ms |   0.82× |
|              200 |     1.750 ms |         0.984 ms |   1.78× |
|              400 |     3.697 ms |         1.157 ms |   3.19× |

NumPy was faster at larger buffer sizes, while array-processing overhead outweighed the benefit at 100 samples.

### Buffer size versus streaming performance

Each configuration contains one decision tree (`max_depth=5`). Accuracy is measured on the same **391 post-warm-up observations**, with each arriving chunk evaluated before training.

| Buffer size | Matched accuracy | Retrain events | Median retraining-update time |
| ----------: | ---------------: | -------------: | ----------------------------: |
|          50 |            66.8% |             17 |                       5.58 ms |
|         100 |            71.6% |              8 |                       7.91 ms |
|         200 |            68.8% |              4 |                       9.43 ms |
|         500 |            68.8% |              1 |                      12.42 ms |

In this experiment, a buffer size of 100 gave the highest matched accuracy, while 500 required the fewest retraining events. The retraining-update timing includes the entire threshold-crossing pipeline update, not just tree fitting. Timings are hardware-dependent.

## Design decisions and limitations

- **Single-tree model:** Each pipeline contains one classification tree; no external machine-learning library or ensemble method is used.
- **Numerical stability:** Split evaluation and preprocessing handle missing values, zero-variance features, invalid shapes, and ties through explicit rules and validation.
- **Replacement retraining:** The trained tree is rebuilt using the latest completed buffer, rather than continuously growing or retaining all earlier training samples.
- **Feature consistency:** Prediction-time preprocessing is frozen between retrains, including when new one-hot categories appear.
- **Warm-up:** Metrics cannot be computed before the first tree is trained.
- **Memory:** Exact cumulative ROC AUC retains labels and scores, so its storage grows with evaluated observations. Logged memory estimates are approximate Python object sizes rather than process memory usage.
- **Pipeline constraints:** Width-changing categorical encoders are supported immediately before the tree; more complex arrangements of stateful transformers may need feature-schema handling.
- **Benchmark interpretation:** Results are specific to the dataset, sample order, model settings, and execution environment; they do not imply a universally optimal buffer size.

## Dependencies

- Python 3.10+
- NumPy
- Matplotlib
- pytest (development/tests)
- Jupyter / ipykernel (notebook execution only)

The streaming model and numerical algorithms do **not** use scikit-learn, pandas, PyTorch, or other external machine-learning/data-processing libraries.
