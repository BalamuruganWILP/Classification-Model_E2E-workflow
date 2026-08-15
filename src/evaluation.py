"""
Common evaluation functions for all five models.
"""

import json
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)


def evaluate_model(
    y_test,
    y_pred
):
    """
    Calculate common classification metrics.
    """

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            y_pred
        ),

        "precision_macro": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "mcc": matthews_corrcoef(
            y_test,
            y_pred
        ),
    }

    return metrics


def print_evaluation(
    y_test,
    y_pred,
    model_name
):
    """
    Print detailed model evaluation.
    """

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} - TEST RESULTS")
    print("=" * 70)

    metrics = evaluate_model(
        y_test,
        y_pred
    )

    for metric, value in metrics.items():
        print(
            f"{metric:25s}: "
            f"{value:.4f}"
        )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    return metrics


def save_metrics(
    metrics,
    model_name,
    output_directory="results"
):
    """
    Save model metrics as JSON.
    """

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_directory
        / f"{model_name}_metrics.json"
    )

    with open(
        output_file,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"\nMetrics saved to: "
        f"{output_file}"
    )