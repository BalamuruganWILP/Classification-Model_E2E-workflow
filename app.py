"""

This application:
1. Loads the untouched test dataset
2. Loads five pre-trained baseline models
3. Applies the required preprocessing
4. Applies saved scalers where required
5. Generates predictions
6. Evaluates all five models
7. Displays a comparison of model performance

IMPORTANT:
- Models are NOT retrained in this application.
- The test dataset is NOT modified.
- No SMOTE or balancing is applied.
- This dashboard evaluates the BASELINE models.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CTG Model Evaluation",
    page_icon="🫀",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_FILE = PROJECT_ROOT / "test_data.csv"

MODEL_DIRECTORY = PROJECT_ROOT / "model"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "Logistic Regression": {
        "model_file": "logistic_regression.pkl",
        "scaler_file": "logistic_regression_scaler.pkl",
        "requires_scaling": True,
    },

    "Decision Tree": {
        "model_file": "decision_tree.pkl",
        "scaler_file": None,
        "requires_scaling": False,
    },

    "KNN": {
        "model_file": "knn.pkl",
        "scaler_file": "knn_scaler.pkl",
        "requires_scaling": True,
    },

    "Gaussian Naive Bayes": {
        "model_file": "naive_bayes.pkl",
        "scaler_file": None,
        "requires_scaling": False,
    },

    "Random Forest": {
        "model_file": "random_forest.pkl",
        "scaler_file": None,
        "requires_scaling": False,
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_test_data():

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_FILE}"
        )

    return pd.read_csv(TEST_FILE)


@st.cache_resource
def load_model(model_path):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    return joblib.load(model_path)


@st.cache_resource
def load_scaler(scaler_path):

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler file not found:\n{scaler_path}"
        )

    return joblib.load(scaler_path)


def prepare_test_data(df):

    """
    Prepare the test data using the same feature/target
    structure used during model training.
    """

    if "NSP" not in df.columns:
        raise ValueError(
            "Target column 'NSP' was not found in test_data.csv."
        )

    # Target
    y = df["NSP"].copy()

    # Remove target
    X = df.drop(columns=["NSP"])

    # Remove non-predictive CLASS column if present
    if "CLASS" in X.columns:
        X = X.drop(columns=["CLASS"])

    return X, y


def calculate_metrics(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),

        "MCC": matthews_corrcoef(
            y_true,
            y_pred
        ),
    }


# ============================================================
# TITLE
# ============================================================

st.title("🫀 Cardiotocography Classification")
st.subheader("Baseline Model Evaluation Dashboard")

st.markdown(
    """
This dashboard evaluates five pre-trained machine learning models
on the **untouched test dataset**.

No model is retrained and no SMOTE or class balancing is applied
in this baseline evaluation.
"""
)


# ============================================================
# LOAD TEST DATA
# ============================================================

try:

    test_df = load_test_data()

except Exception as e:

    st.error(
        f"Unable to load test dataset: {e}"
    )

    st.stop()


# ============================================================
# PREPARE TEST DATA
# ============================================================

try:

    X_test, y_test = prepare_test_data(
        test_df
    )

except Exception as e:

    st.error(
        f"Unable to prepare test dataset: {e}"
    )

    st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.header("Test Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Test Samples",
        len(test_df)
    )

with col2:
    st.metric(
        "Features",
        X_test.shape[1]
    )

with col3:
    st.metric(
        "Classes",
        y_test.nunique()
    )

with col4:
    st.metric(
        "Target",
        "NSP"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

st.subheader("Test Set Class Distribution")

class_counts = (
    y_test
    .value_counts()
    .sort_index()
)

class_percentages = (
    y_test
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

distribution_df = pd.DataFrame({
    "Class": class_counts.index,
    "Count": class_counts.values,
    "Percentage": class_percentages.values
})

st.dataframe(
    distribution_df,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    class_counts
)


# ============================================================
# MODEL EVALUATION
# ============================================================

st.header("Model Evaluation")

results = {}

predictions = {}

confusion_matrices = {}

errors = {}


for model_name, config in MODEL_CONFIG.items():

    try:

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model_path = (
            MODEL_DIRECTORY /
            config["model_file"]
        )

        model = load_model(
            model_path
        )

        # ----------------------------------------------------
        # Prepare input
        # ----------------------------------------------------

        X_input = X_test.copy()

        # ----------------------------------------------------
        # Apply saved scaler if required
        # ----------------------------------------------------

        if config["requires_scaling"]:

            scaler_path = (
                MODEL_DIRECTORY /
                config["scaler_file"]
            )

            scaler = load_scaler(
                scaler_path
            )

            X_input = scaler.transform(
                X_input
            )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_pred = model.predict(
            X_input
        )

        predictions[model_name] = y_pred

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            y_test,
            y_pred
        )

        results[model_name] = metrics

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        confusion_matrices[model_name] = (
            confusion_matrix(
                y_test,
                y_pred,
                labels=sorted(
                    y_test.unique()
                )
            )
        )

    except Exception as e:

        errors[model_name] = str(e)


# ============================================================
# DISPLAY ERRORS
# ============================================================

if errors:

    st.warning(
        "Some models could not be evaluated."
    )

    for model_name, error in errors.items():

        st.error(
            f"{model_name}: {error}"
        )


# ============================================================
# RESULTS TABLE
# ============================================================

if results:

    results_df = (
        pd.DataFrame(results)
        .T
    )

    results_df = (
        results_df
        .sort_values(
            "Recall",
            ascending=False
        )
    )

    st.subheader(
        "Model Performance Comparison"
    )

    st.dataframe(
        results_df.style.format(
            "{:.4f}"
        ),
        use_container_width=True
    )


# ============================================================
# PRIMARY METRIC
# ============================================================

if results:

    st.subheader(
        "Primary Metric — Recall"
    )

    st.markdown(
        """
    **Recall is particularly important for this medical
    classification problem because false negatives can be
    clinically significant.**

    A higher recall means the model identifies a larger
    proportion of the actual cases belonging to each class.
    """
    )

    recall_df = (
        results_df[["Recall"]]
        .sort_values(
            "Recall",
            ascending=False
        )
    )

    st.bar_chart(
        recall_df
    )


# ============================================================
# METRIC COMPARISON
# ============================================================

if results:

    st.subheader(
        "Overall Metric Comparison"
    )

    chart_df = results_df[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Balanced Accuracy",
        ]
    ]

    st.bar_chart(
        chart_df
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

if confusion_matrices:

    st.header(
        "Confusion Matrices"
    )

    st.markdown(
        """
        Rows represent the **actual NSP class** and columns
        represent the **predicted NSP class**.
        """
    )

    for model_name, cm in confusion_matrices.items():

        st.subheader(
            model_name
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                f"Actual {c}"
                for c in sorted(
                    y_test.unique()
                )
            ],
            columns=[
                f"Predicted {c}"
                for c in sorted(
                    y_test.unique()
                )
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )


# ============================================================
# PER-CLASS PERFORMANCE
# ============================================================

st.header(
    "Per-Class Performance"
)

selected_model = st.selectbox(
    "Select model",
    list(predictions.keys())
)

if selected_model:

    y_pred_selected = (
        predictions[selected_model]
    )

    report = classification_report(
        y_test,
        y_pred_selected,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).T

    # Keep actual classes + aggregate rows
    st.dataframe(
        report_df.style.format(
            "{:.4f}"
        ),
        use_container_width=True
    )


# ============================================================
# BEST MODEL
# ============================================================

if results:

    best_model = (
        results_df["Recall"]
        .idxmax()
    )

    best_recall = (
        results_df.loc[
            best_model,
            "Recall"
        ]
    )

    st.header(
        "Baseline Model Summary"
    )

    st.success(
        f"Highest baseline Recall: "
        f"{best_model} "
        f"({best_recall:.4f})"
    )

    st.info(
        """
        This identifies the best model only according to
        weighted Recall on the untouched test set.

        Final model selection should consider per-class
        Recall, F1 Score, confusion matrices, and the
        impact of class imbalance rather than relying on
        a single metric.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Cardiotocography ML Assignment | "
    "Baseline Evaluation | "
    "Test set remains untouched"
)