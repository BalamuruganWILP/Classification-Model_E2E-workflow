"""
Utilities for handling class imbalance.

Methods:
1. Class weights
2. Random undersampling

SMOTE is kept in smote.py.
"""

import pandas as pd

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# Class weights
# ============================================================

def calculate_class_weights(y):
    """
    Calculate balanced class weights.

    Parameters
    ----------
    y : pandas.Series

    Returns
    -------
    dict
        Mapping of class -> weight
    """

    classes = sorted(y.unique())

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    class_weights = dict(
        zip(classes, weights)
    )

    return class_weights


# ============================================================
# Random undersampling
# ============================================================

def undersample_training_data(
    X_train,
    y_train,
    random_state=42
):
    """
    Randomly undersample the majority class.

    IMPORTANT:
    This function should ONLY be applied to training data.

    The test dataset must remain untouched.
    """

    train_data = X_train.copy()

    train_data["_target_"] = y_train.values

    min_class_count = (
        train_data["_target_"]
        .value_counts()
        .min()
    )

    balanced_parts = []

    for class_value in sorted(
        train_data["_target_"].unique()
    ):

        class_data = train_data[
            train_data["_target_"] == class_value
        ]

        sampled_class = class_data.sample(
            n=min_class_count,
            random_state=random_state
        )

        balanced_parts.append(sampled_class)

    balanced_data = pd.concat(
        balanced_parts
    ).sample(
        frac=1,
        random_state=random_state
    )

    y_balanced = balanced_data["_target_"]

    X_balanced = balanced_data.drop(
        columns=["_target_"]
    )

    return X_balanced, y_balanced