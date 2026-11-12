"""Streaming, single-decision-tree extension for NumCompute Assignment 2.2."""

from .tree import DecisionTreeClassifier
from .preprocessing import StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
from .stats import StreamingStats
from .pipeline import Pipeline
from .stream import StreamTrainer
from .visualise import (plot_metric_over_time, compare_models,
                        plot_predictions_vs_ground_truth, plot_stream_dashboard,
                        LiveMetricPlot)
from .metrics import (StreamingClassificationMetrics, ConfusionMatrix, Accuracy,
                      Precision, Recall, F1Score, AUC, binary_auc)

__all__ = [
    "DecisionTreeClassifier",
    "StandardScaler", "MinMaxScaler", "Imputer", "OneHotEncoder",
    "StreamingStats", "Pipeline", "StreamTrainer",
    "StreamingClassificationMetrics", "ConfusionMatrix", "Accuracy",
    "Precision", "Recall", "F1Score", "AUC", "binary_auc",
    "plot_metric_over_time", "compare_models", "plot_predictions_vs_ground_truth",
    "plot_stream_dashboard", "LiveMetricPlot",
]
