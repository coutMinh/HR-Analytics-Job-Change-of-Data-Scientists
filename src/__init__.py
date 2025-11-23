"""
Core package for the HR Analytics project.

The modules expose NumPy-first utilities for data ingestion, preprocessing,
visualization helpers (Matplotlib/Seaborn only), and scratch-built models.
"""

from .data_processing import (  # noqa: F401
    preprocess_train_dataset,
    preprocess_test_dataset,
    load_raw_dataset,
    summarize_categorical,
    summarize_numeric,
)
from .models import (  # noqa: F401
    LogisticRegressionGD,
    k_fold_cross_validate,
    train_test_split,
    classification_report,
)
from .visualization import (  # noqa: F401
    plot_numeric_distribution,
    plot_categorical_counts,
    plot_feature_scatter,
    plot_correlation_heatmap,
)

__all__ = [
    "preprocess_train_dataset",
    "preprocess_test_dataset",
    "load_raw_dataset",
    "summarize_categorical",
    "summarize_numeric",
    "LogisticRegressionGD",
    "k_fold_cross_validate",
    "train_test_split",
    "classification_report",
    "plot_numeric_distribution",
    "plot_categorical_counts",
    "plot_feature_scatter",
    "plot_correlation_heatmap",
]

