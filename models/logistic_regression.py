"""
Logistic Regression Model
=========================

This script:
1. Loads training and test data
2. Separates features and target
3. Scales the numerical features
4. Trains Logistic Regression
5. Makes predictions on the untouched test set
6. Evaluates the model
7. Saves the trained model
8. Saves evaluation metrics

IMPORTANT:
This is the BASELINE Logistic Regression model.

No SMOTE, undersampling, or class weighting is applied here.
"""

from pathlib import Path
import sys
import joblib


# ============================================================
# PROJECT PATHS
# ============================================================

# models/logistic_regression.py
#
# parent       -> models
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

from scaling import scale_train_test

from evaluation import (
    print_evaluation,
    save_metrics,
)


# ============================================================
# IMPORT MACHINE LEARNING MODEL
# ============================================================

from sklearn.linear_model import LogisticRegression


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FILE = PROJECT_ROOT / "train_data.csv"

TEST_FILE = PROJECT_ROOT / "test_data.csv"

MODEL_DIRECTORY = PROJECT_ROOT / "model"

RESULTS_DIRECTORY = PROJECT_ROOT / "results"

MODEL_FILE = (
    MODEL_DIRECTORY /
    "logistic_regression.pkl"
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("LOGISTIC REGRESSION - BASELINE")
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
        f"Target        : NSP"
    )

    # --------------------------------------------------------
    # 4. Scale features
    # --------------------------------------------------------

    print(
        "\n[4/7] Scaling features..."
    )

    (
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = scale_train_test(
        X_train,
        X_test
    )

    print(
        "✓ StandardScaler fitted on training data."
    )

    print(
        "✓ Training data transformed."
    )

    print(
        "✓ Test data transformed using the same scaler."
    )

    # --------------------------------------------------------
    # 5. Create and train model
    # --------------------------------------------------------

    print(
        "\n[5/7] Training Logistic Regression..."
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    print(
        "✓ Model training completed."
    )

    # --------------------------------------------------------
    # 6. Test model
    # --------------------------------------------------------

    print(
        "\n[6/7] Evaluating on untouched test data..."
    )

    y_pred = model.predict(
        X_test_scaled
    )

    # Evaluation
    metrics = print_evaluation(
        y_test,
        y_pred,
        "Logistic Regression - Baseline"
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

    # Save model
    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"✓ Model saved to:\n  {MODEL_FILE}"
    )

    # Save scaler
    SCALER_FILE = (
        MODEL_DIRECTORY /
        "logistic_regression_scaler.pkl"
    )

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    print(
        f"✓ Scaler saved to:\n  {SCALER_FILE}"
    )

    # Save metrics
    save_metrics(
        metrics,
        "logistic_regression_baseline",
        RESULTS_DIRECTORY
    )

    print("\n")
    print("=" * 70)
    print("LOGISTIC REGRESSION COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()