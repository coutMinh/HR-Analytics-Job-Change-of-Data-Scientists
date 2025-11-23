"""
Scratch implementations of models and evaluation utilities using NumPy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
    split_idx = int(n_samples * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


@dataclass
class TrainingHistory:
    losses: np.ndarray
    accuracies: np.ndarray


class LogisticRegressionGD:
    """
    Binary logistic regression trained with vectorized gradient descent.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 1000,
        l2_reg: float = 0.0,
        tolerance: float = 1e-6,
        class_weight: Optional[Dict[float, float]] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.tolerance = tolerance
        self.class_weight = class_weight or {}
        self.weights: np.ndarray | None = None
        self.history: TrainingHistory | None = None

    def _augment(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        X_aug = self._augment(X)
        n_samples, n_features = X_aug.shape
        sample_weights = np.ones_like(y, dtype=float)
        for cls_val, cls_weight in self.class_weight.items():
            sample_weights = np.where(y == cls_val, cls_weight, sample_weights)
        normalization = np.sum(sample_weights)

        weights = np.zeros(n_features)
        losses = np.zeros(self.epochs)
        accuracies = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            logits = np.einsum("ij,j->i", X_aug, weights)
            preds = _sigmoid(logits)
            error = (preds - y) * sample_weights
            gradient = np.einsum("ij,i->j", X_aug, error) / normalization
            gradient[1:] += self.l2_reg * weights[1:] / n_samples
            weights -= self.learning_rate * gradient
            weighted_loss = -np.sum(
                sample_weights * (y * np.log(preds + 1e-12) + (1 - y) * np.log(1 - preds + 1e-12))
            ) / normalization
            reg_term = (self.l2_reg / (2 * n_samples)) * np.sum(weights[1:] ** 2)
            losses[epoch] = weighted_loss + reg_term
            accuracies[epoch] = np.mean((preds >= 0.5) == y)
            if epoch > 0 and abs(losses[epoch] - losses[epoch - 1]) < self.tolerance:
                losses = losses[: epoch + 1]
                accuracies = accuracies[: epoch + 1]
                break
        self.weights = weights
        self.history = TrainingHistory(losses=losses, accuracies=accuracies)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fitted.")
        X_aug = self._augment(X)
        logits = np.einsum("ij,j->i", X_aug, self.weights)
        return _sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(float)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / max(y_true.size, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    report = classification_metrics(y_true, y_pred)
    if y_prob is not None:
        y_prob = np.clip(y_prob, 1e-9, 1 - 1e-9)
        log_loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        report["log_loss"] = float(log_loss)
    return report


