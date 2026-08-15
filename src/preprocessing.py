"""
1. Load training CSV file
2. Validate expected columns
3. Separate features and target
4. Remove non-predictive columns
5. Check missing values
6. Visualize the distribution of the three target classes
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Configuration
TARGET_COLUMN = "NSP"

# Non-predictive columns
NON_PREDICTIVE_COLUMNS = [
    "CLASS",
]

# Expected CTG input features
FEATURE_COLUMNS = [
    "LB",
    "AC",
    "FM",
    "UC",
    "DL",
    "DS",
    "DP",
    "ASTV",
    "MSTV",
    "ALTV",
    "MLTV",
    "Width",
    "Min",
    "Max",
    "Nmax",
    "Nzeros",
    "Mode",
    "Mean",
    "Median",
    "Variance",
    "Tendency",
]



# 1. Load training data

def load_training_data(file_path):
    """
    Load the training CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to train_data.csv

    Returns
    -------
    pandas.DataFrame
        Loaded training dataset
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print("=" * 70)
    print("TRAINING DATA LOADED")
    print("=" * 70)

    print(f"File   : {file_path}")
    print(f"Rows   : {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


# 2. Validate expected columns

def validate_columns(df):
    """
    Validate that the training dataset contains all
    expected columns.
    """

    expected_columns = (
        FEATURE_COLUMNS
        + ["CLASS", TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following expected columns are missing:\n"
            f"{missing_columns}"
        )

    print("\n" + "=" * 70)
    print("COLUMN VALIDATION")
    print("=" * 70)

    print("✓ All expected columns are present.")

    return True


# 3 & 4. Separate features and target
#         Remove non-predictive columns


def prepare_features_and_target(df):
    """
    Separate the dataset into:

    X = 21 CTG predictive features
    y = NSP target

    Non-predictive columns:
        - CLASS

    are excluded from X.
    """

    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET_COLUMN].copy()

    print("\n" + "=" * 70)
    print("FEATURE / TARGET SEPARATION")
    print("=" * 70)

    print(f"Number of features : {X.shape[1]}")
    print(f"Number of samples  : {X.shape[0]}")
    print(f"Target variable    : {TARGET_COLUMN}")

    print("\nRemoved non-predictive columns:")
    for column in NON_PREDICTIVE_COLUMNS:
        print(f"  - {column}")

    print("\nFeature columns:")
    for i, column in enumerate(X.columns, start=1):
        print(f"  {i:2}. {column}")

    return X, y



# 5. Check missing values

def check_missing_values(df):
    """
    Check for missing values in the dataset.
    """

    missing_counts = df.isnull().sum()

    total_missing = missing_counts.sum()

    print("\n" + "=" * 70)
    print("MISSING VALUE CHECK")
    print("=" * 70)

    if total_missing == 0:
        print("✓ No missing values found.")
    else:
        print(f"Total missing values: {total_missing}")
        print("\nMissing values by column:")

        print(
            missing_counts[missing_counts > 0]
            .sort_values(ascending=False)
        )

    return missing_counts


# 6. Class distribution

def get_class_distribution(y):
    """
    Calculate count and percentage of each NSP class.
    """

    class_counts = (
        y.value_counts()
        .sort_index()
    )

    class_percentages = (
        y.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    distribution = pd.DataFrame({
        "Count": class_counts,
        "Percentage": class_percentages
    })

    return distribution


def plot_class_distribution(y):
    """
    Plot the distribution of the three NSP classes.
    """

    distribution = get_class_distribution(y)

    print("\n" + "=" * 70)
    print("TARGET CLASS DISTRIBUTION")
    print("=" * 70)

    print(distribution)

    # Create labels for the plot
    class_labels = {
        1: "NSP 1 - Normal",
        2: "NSP 2 - Suspect",
        3: "NSP 3 - Pathologic",
    }

    labels = [
        class_labels.get(cls, f"NSP {cls}")
        for cls in distribution.index
    ]

    # Plot
    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        labels,
        distribution["Count"]
    )

    plt.title("Training Data - NSP Class Distribution")
    plt.xlabel("NSP Class")
    plt.ylabel("Number of Samples")

    plt.xticks(rotation=0)

    # Display count and percentage above each bar
    for bar, count, percentage in zip(
        bars,
        distribution["Count"],
        distribution["Percentage"]
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({percentage:.1f}%)",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()


# Main preprocessing workflow

def preprocess_training_data(file_path):
    """
    Execute the complete initial preprocessing workflow.

    Returns
    -------
    X : pandas.DataFrame
        21 CTG input features

    y : pandas.Series
        NSP target
    """

    # 1. Load data
    df = load_training_data(file_path)

    print(df.head())
    #print some basic info about the dataframe
    print("\nDataframe info:")
    print(df.info())

    # 2. Validate columns
    validate_columns(df)

    # 3 & 4. Separate features and target
    X, y = prepare_features_and_target(df)

    # 5. Check missing values
    check_missing_values(df)

    # Validate target classes
    valid_classes = {1, 2, 3}

    actual_classes = set(y.unique())

    if not actual_classes.issubset(valid_classes):
        raise ValueError(
            f"Unexpected NSP classes found: {actual_classes}"
        )

    # 6. Display class distribution
    plot_class_distribution(y)

    return X, y


if __name__ == "__main__":


    # Data:
    # data/train_data.csv

    project_root = Path(__file__).resolve().parent.parent

    train_file = (
        project_root
            / "train_data.csv"
    )

    print(f"Training file: {train_file}")

    X_train, y_train = preprocess_training_data(train_file)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)

    print(f"X shape: {X_train.shape}")
    print(f"y shape: {y_train.shape}")