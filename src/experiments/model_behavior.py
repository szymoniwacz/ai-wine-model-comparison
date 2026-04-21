"""
Experiment: compare Logistic Regression vs SVM not only by accuracy,
but by where their predictions actually differ.
"""

import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from src.data_loader import load_data
from src.model_factory import create_model

_MODELS = ["logistic_regression", "svm"]
_DEFAULT_ARTIFACTS_DIR = "artifacts"


def _save_individual_confusion_matrices(
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    class_names: list[str],
    artifacts_dir: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
            ax=ax, colorbar=False
        )
        ax.set_title(f"Confusion Matrix: {name.replace('_', ' ').title()}")
        path = os.path.join(artifacts_dir, f"cm_{name}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths[f"cm_{name}"] = path
    return paths


def _save_combined_confusion_matrix(
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    class_names: list[str],
    artifacts_dir: str,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
            ax=ax, colorbar=False
        )
        ax.set_title(name.replace("_", " ").title())
    fig.suptitle("Confusion Matrices: Logistic Regression vs SVM", fontsize=14)
    path = os.path.join(artifacts_dir, "cm_combined.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_agreement_chart(
    agreements: int,
    disagreements: int,
    artifacts_dir: str,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = [agreements, disagreements]
    bars = ax.bar(["Agreement", "Disagreement"], counts, color=["steelblue", "tomato"])
    ax.set_ylabel("Number of samples")
    ax.set_title("Model Agreement: Logistic Regression vs SVM")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )
    path = os.path.join(artifacts_dir, "model_agreement.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_accuracy_chart(
    accuracies: dict[str, float],
    artifacts_dir: str,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    names = [n.replace("_", " ").title() for n in accuracies]
    values = list(accuracies.values())
    bars = ax.bar(names, values, color="skyblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison: Logistic Regression vs SVM")
    for bar, acc in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{acc:.3f}",
            ha="center",
            va="bottom",
        )
    path = os.path.join(artifacts_dir, "accuracy_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run(artifacts_dir: str = _DEFAULT_ARTIFACTS_DIR) -> dict[str, Any]:
    """
    Run the model behavior experiment.

    Trains Logistic Regression and SVM on the same train/test split,
    computes accuracies, and finds samples where the models disagree.

    Returns a dict with:
        accuracies          – per-model accuracy on the test set
        disagreement_count  – number of test samples where predictions differ
        disagreement_indices – test-set indices where predictions differ
        artifact_paths      – paths to all saved plot files
    """
    wine = load_wine()
    class_names: list[str] = wine.target_names.tolist()

    X_train, X_test, y_train, y_test = load_data()

    predictions: dict[str, np.ndarray] = {}
    accuracies: dict[str, float] = {}
    for name in _MODELS:
        model = create_model(name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        accuracies[name] = float(accuracy_score(y_test, y_pred))

    lr_preds = predictions["logistic_regression"]
    svm_preds = predictions["svm"]
    disagreement_mask = lr_preds != svm_preds
    disagreement_indices: list[int] = np.where(disagreement_mask)[0].tolist()
    disagreement_count = len(disagreement_indices)
    agreements = len(lr_preds) - disagreement_count

    os.makedirs(artifacts_dir, exist_ok=True)

    artifact_paths: dict[str, str] = {}
    artifact_paths.update(
        _save_individual_confusion_matrices(
            predictions, y_test, class_names, artifacts_dir
        )
    )
    artifact_paths["cm_combined"] = _save_combined_confusion_matrix(
        predictions, y_test, class_names, artifacts_dir
    )
    artifact_paths["agreement_chart"] = _save_agreement_chart(
        agreements, disagreement_count, artifacts_dir
    )
    artifact_paths["accuracy_chart"] = _save_accuracy_chart(accuracies, artifacts_dir)

    return {
        "accuracies": accuracies,
        "disagreement_count": disagreement_count,
        "disagreement_indices": disagreement_indices,
        "artifact_paths": artifact_paths,
    }
