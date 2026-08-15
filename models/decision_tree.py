"""

This script:
1. Loads training and test data
2. Separates features and target
3. Trains a Decision Tree
4. Makes predictions on the untouched test set
5. Evaluates the model
6. Saves the trained model
7. Saves evaluation metrics

IMPORTANT:
This is the BASELINE Decision Tree model.

No scaling, SMOTE, undersampling, or class weighting
is applied in this baseline.
"""

from pathlib import Path
import sys
import joblib


# ============================================================
# PROJECT PATHS
# ============================================================

# models/decision_tree.py
#
# parent        -> models
# parent.parent -> project root

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_PATH = PROJECT_ROOT / "src"

# Allow Python to find modules inside src/
sys.path.insert(0, str(SRC_PATH))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from preprocessing import (
    load_training_data,
    prepare_features_and_target,
)

from evaluation import (
    print_evaluation,
    save_metrics,
)


# ============================================================
# IMPORT MACHINE LEARNING MODEL
# ============================================================

from sklearn.tree import DecisionTreeClassifier


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FILE = PROJECT_ROOT / "train_data.csv"

TEST_FILE = PROJECT_ROOT / "test_data.csv"

MODEL_DIRECTORY = PROJECT_ROOT / "model"

RESULTS_DIRECTORY = PROJECT_ROOT / "results"

MODEL_FILE = (
    MODEL_DIRECTORY /
    "decision_tree.pkl"
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("DECISION TREE - BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check files
    # --------------------------------------------------------

    print("\n[1/6] Checking data files...")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found:\n{TEST_FILE}"
        )

    print(f"Training file : {TRAIN_FILE}")
    print(f"Test file     : {TEST_FILE}")

    # --------------------------------------------------------
    # 2. Load datasets
    # --------------------------------------------------------

    print("\n[2/6] Loading datasets...")

    train_df = load_training_data(
        TRAIN_FILE
    )

    test_df = load_training_data(
        TEST_FILE
    )

    print(
        f"Training shape : {train_df.shape}"
    )

    print(
        f"Test shape     : {test_df.shape}"
    )

    # --------------------------------------------------------
    # 3. Separate features and target
    # --------------------------------------------------------

    print(
        "\n[3/6] Separating features and target..."
    )

    X_train, y_train = (
        prepare_features_and_target(
            train_df
        )
    )

    X_test, y_test = (
        prepare_features_and_target(
            test_df
        )
    )

    print(
        f"X_train shape : {X_train.shape}"
    )

    print(
        f"y_train shape : {y_train.shape}"
    )

    print(
        f"X_test shape  : {X_test.shape}"
    )

    print(
        f"y_test shape  : {y_test.shape}"
    )

    print(
        f"Features      : {X_train.shape[1]}"
    )

    print(
        "Target        : NSP"
    )

    # --------------------------------------------------------
    # 4. Create and train model
    # --------------------------------------------------------

    print(
        "\n[4/6] Training Decision Tree..."
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "✓ Model training completed."
    )

    print(
        f"Tree depth    : {model.get_depth()}"
    )

    print(
        f"Number leaves : {model.get_n_leaves()}"
    )

    # --------------------------------------------------------
    # 5. Test and evaluate model
    # --------------------------------------------------------

    print(
        "\n[5/6] Evaluating on untouched test data..."
    )

    y_pred = model.predict(
        X_test
    )

    metrics = print_evaluation(
        y_test,
        y_pred,
        "Decision Tree - Baseline"
    )

    # --------------------------------------------------------
    # 6. Save model and results
    # --------------------------------------------------------

    print(
        "\n[6/6] Saving model and results..."
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save trained model
    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"✓ Model saved to:\n  {MODEL_FILE}"
    )

    # Save metrics
    save_metrics(
        metrics,
        "decision_tree_baseline",
        RESULTS_DIRECTORY
    )

    print("\n")
    print("=" * 70)
    print("DECISION TREE COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()