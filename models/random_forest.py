"""

This script:
1. Loads training and test data
2. Separates features and target
3. Trains Random Forest
4. Makes predictions on the untouched test set
5. Evaluates the model
6. Displays feature importance
7. Saves the trained model
8. Saves evaluation metrics

IMPORTANT:
This is the BASELINE Random Forest model.

No scaling, SMOTE, undersampling, or class weighting
is applied in this baseline.
"""

from pathlib import Path
import sys
import joblib


# ============================================================
# PROJECT PATHS
# ============================================================

# models/random_forest.py
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

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FILE = PROJECT_ROOT / "train_data.csv"

TEST_FILE = PROJECT_ROOT / "test_data.csv"

MODEL_DIRECTORY = PROJECT_ROOT / "model"

RESULTS_DIRECTORY = PROJECT_ROOT / "results"

MODEL_FILE = MODEL_DIRECTORY / "random_forest.pkl"


# ============================================================
# RANDOM FOREST CONFIGURATION
# ============================================================

# Baseline configuration.
#
# We will tune these parameters later.
N_ESTIMATORS = 100

CRITERION = "gini"

RANDOM_STATE = 42

N_JOBS = -1


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("RANDOM FOREST - BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check files
    # --------------------------------------------------------

    print("\n[1/7] Checking data files...")

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

    print("\n[2/7] Loading datasets...")

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
        "\n[3/7] Separating features and target..."
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
    # 4. Create and train Random Forest
    # --------------------------------------------------------

    print(
        "\n[4/7] Training Random Forest..."
    )

    print(
        f"Number of trees : {N_ESTIMATORS}"
    )

    print(
        f"Criterion       : {CRITERION}"
    )

    print(
        f"Random state    : {RANDOM_STATE}"
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        criterion=CRITERION,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "✓ Model training completed."
    )

    # --------------------------------------------------------
    # 5. Test and evaluate
    # --------------------------------------------------------

    print(
        "\n[5/7] Evaluating on untouched test data..."
    )

    y_pred = model.predict(
        X_test
    )

    metrics = print_evaluation(
        y_test,
        y_pred,
        "Random Forest - Baseline"
    )

    # --------------------------------------------------------
    # 6. Display feature importance
    # --------------------------------------------------------

    print(
        "\n[6/7] Feature importance..."
    )

    feature_importance = model.feature_importances_

    importance_pairs = list(
        zip(
            X_train.columns,
            feature_importance
        )
    )

    importance_pairs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop 10 features:")

    for rank, (feature, importance) in enumerate(
        importance_pairs[:10],
        start=1
    ):
        print(
            f"{rank:2d}. "
            f"{feature:<12} "
            f"{importance:.4f}"
        )

    # --------------------------------------------------------
    # 7. Save model and results
    # --------------------------------------------------------

    print(
        "\n[7/7] Saving model and results..."
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
        "random_forest_baseline",
        RESULTS_DIRECTORY
    )

    print("\n")
    print("=" * 70)
    print("RANDOM FOREST COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()