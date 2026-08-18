"""
Cardiotocography ML Assignment
==============================

Streamlit Model Evaluation Dashboard

Features
--------
1. Upload a CSV test dataset
2. Use the bundled test_data.csv
3. Download the bundled test dataset
4. Select any trained model
5. Display selected-model metrics prominently
6. Display selected-model confusion matrix
7. Display selected-model classification report
8. Compare all available models
9. Display confusion matrices and reports for all models
10. Display class distribution
11. Display weighted vs macro metrics
12. Display Random Forest feature importance

Note: This application ONLY evaluates already-trained models.


"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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



# PAGE CONFIGURATION

st.set_page_config(
    page_title="CTG Model Evaluation",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_FILE = PROJECT_ROOT / "test_data.csv"

MODEL_DIRECTORY = PROJECT_ROOT / "model"


# ============================================================
# CONSTANTS
# ============================================================

TARGET_COLUMN = "NSP"

NON_PREDICTIVE_COLUMNS = [
    "CLASS"
]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {

    "Logistic Regression": {
        "model_file": "logistic_regression.pkl",
        "scaler_file": "logistic_regression_scaler.pkl",
        "requires_scaling": True,
        "description": (
            "Linear classification model using logistic regression."
        ),
    },

    "Decision Tree": {
        "model_file": "decision_tree.pkl",
        "scaler_file": None,
        "requires_scaling": False,
        "description": (
            "Tree-based classification model using recursive feature splitting."
        ),
    },

    "KNN": {
        "model_file": "knn.pkl",
        "scaler_file": "knn_scaler.pkl",
        "requires_scaling": True,
        "description": (
            "Distance-based classifier using nearest observations."
        ),
    },

    "Gaussian Naive Bayes": {
        "model_file": "naive_bayes.pkl",
        "scaler_file": None,
        "requires_scaling": False,
        "description": (
            "Probabilistic classifier assuming Gaussian feature distributions."
        ),
    },

    "Random Forest": {
        "model_file": "random_forest.pkl",
        "scaler_file": None,
        "requires_scaling": False,
        "description": (
            "Ensemble model combining predictions from multiple decision trees."
        ),
    },
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 650;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    return joblib.load(model_path)


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler(scaler_path):

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler file not found:\n{scaler_path}"
        )

    return joblib.load(scaler_path)


# ============================================================
# LOAD BUNDLED TEST DATA
# ============================================================

@st.cache_data
def load_bundled_test_data():

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Bundled test dataset not found:\n{TEST_FILE}"
        )

    return pd.read_csv(TEST_FILE)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_test_data(df):

    """
    Prepare uploaded or bundled dataset.

    Required:
        NSP target column

    Optional:
        CLASS is removed because it is non-predictive.

    Returns:
        X, y
    """

    df = df.copy()

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' "
            "was not found in the uploaded CSV."
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = df[TARGET_COLUMN].copy()

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    # --------------------------------------------------------
    # Remove non-predictive columns
    # --------------------------------------------------------

    for column in NON_PREDICTIVE_COLUMNS:

        if column in X.columns:

            X = X.drop(
                columns=[column]
            )

    # --------------------------------------------------------
    # Validate numeric features
    # --------------------------------------------------------

    non_numeric_columns = (
        X.select_dtypes(
            exclude=np.number
        ).columns.tolist()
    )

    if non_numeric_columns:

        raise ValueError(
            "The following feature columns are not numeric:\n"
            + ", ".join(non_numeric_columns)
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_count = int(
        X.isna().sum().sum()
    )

    if missing_count > 0:

        raise ValueError(
            f"The dataset contains {missing_count} missing "
            "feature values. Please provide a cleaned dataset."
        )

    return X, y


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(X, expected_features):

    """
    Ensure uploaded dataset contains exactly the features
    expected by the trained models.
    """

    missing_features = [
        feature
        for feature in expected_features
        if feature not in X.columns
    ]

    extra_features = [
        feature
        for feature in X.columns
        if feature not in expected_features
    ]

    if missing_features:

        raise ValueError(
            "The uploaded dataset is missing required "
            f"feature columns:\n{missing_features}"
        )

    # Keep only the features used during training
    X = X[expected_features].copy()

    return X, extra_features


# ============================================================
# GET EXPECTED FEATURES
# ============================================================

def get_expected_features():

    """
    Determine feature names from the bundled test dataset.
    """

    bundled_df = load_bundled_test_data()

    X, _ = prepare_test_data(
        bundled_df
    )

    return X.columns.tolist()


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

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

        "Macro Precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Macro Recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Macro F1": f1_score(
            y_true,
            y_pred,
            average="macro",
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
# CLASSIFICATION REPORT
# ============================================================

def get_classification_report_df(
    y_true,
    y_pred
):

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    rows = []

    for key, value in report.items():

        if key in [
            "accuracy",
            "macro avg",
            "weighted avg"
        ]:
            continue

        rows.append(
            {
                "Class": f"NSP {key}",
                "Precision": value["precision"],
                "Recall": value["recall"],
                "F1 Score": value["f1-score"],
                "Support": int(
                    value["support"]
                ),
            }
        )

    # Add summary rows

    rows.append(
        {
            "Class": "Macro Average",
            "Precision": report[
                "macro avg"
            ]["precision"],
            "Recall": report[
                "macro avg"
            ]["recall"],
            "F1 Score": report[
                "macro avg"
            ]["f1-score"],
            "Support": int(
                report[
                    "macro avg"
                ]["support"]
            ),
        }
    )

    rows.append(
        {
            "Class": "Weighted Average",
            "Precision": report[
                "weighted avg"
            ]["precision"],
            "Recall": report[
                "weighted avg"
            ]["recall"],
            "F1 Score": report[
                "weighted avg"
            ]["f1-score"],
            "Support": int(
                report[
                    "weighted avg"
                ]["support"]
            ),
        }
    )

    return pd.DataFrame(rows)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name,
    labels
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 4.5)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
        aspect="auto"
    )

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set(
        xticks=np.arange(
            len(labels)
        ),
        yticks=np.arange(
            len(labels)
        ),
        xticklabels=[
            f"NSP {label}"
            for label in labels
        ],
        yticklabels=[
            f"NSP {label}"
            for label in labels
        ],
        xlabel="Predicted Class",
        ylabel="Actual Class",
        title=f"{model_name} — Confusion Matrix",
    )

    threshold = cm.max() / 2

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
                fontsize=12,
                fontweight="bold",
            )

    fig.tight_layout()

    return fig


# ============================================================
# METRIC COMPARISON CHART
# ============================================================

def plot_metric_comparison(
    results_df
):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    plot_df = results_df[
        metrics
    ]

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    x = np.arange(
        len(plot_df.index)
    )

    width = 0.18

    for i, metric in enumerate(
        metrics
    ):

        ax.bar(
            x + (
                i - 1.5
            ) * width,
            plot_df[metric],
            width,
            label=metric
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_df.index,
        rotation=20,
        ha="right"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_title(
        "Performance Comparison — All Models"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    return fig


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(
    model,
    feature_names,
    top_n=10
):

    if not hasattr(
        model,
        "feature_importances_"
    ):
        return None

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(top_n)
        .sort_values(
            "Importance",
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_title(
        "Random Forest — Top 10 Feature Importance"
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    fig.tight_layout()

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🫀 Cardiotocography Classification'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Model Evaluation Dashboard'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Evaluation Settings"
    )

    st.subheader(
        "1. Select Dataset"
    )

    dataset_source = st.radio(
        "Dataset source",
        [
            "Use bundled test data",
            "Upload CSV"
        ],
        index=0
    )

    st.divider()

    st.subheader(
        "2. Select Model"
    )

    selected_model = st.selectbox(
        "Model",
        list(
            MODEL_CONFIG.keys()
        )
    )

    st.divider()

    st.subheader(
        "Baseline Configuration"
    )

    st.caption(
        """
        Models are evaluated using saved `.pkl`files.

        """
    )


# ============================================================
# DATASET SELECTION
# ============================================================

uploaded_file = None

if dataset_source == "Upload CSV":

    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type=["csv"],
        help=(
            "CSV must contain NSP and the 21 "
            "predictor variables."
        )
    )

    if uploaded_file is None:

        st.info(
            "Please upload a CSV file from the sidebar "
            "to begin evaluation."
        )

        st.stop()

    try:

        test_df = pd.read_csv(
            uploaded_file
        )

        dataset_name = uploaded_file.name

    except Exception as error:

        st.error(
            f"Unable to read uploaded CSV:\n{error}"
        )

        st.stop()

else:

    try:

        test_df = load_bundled_test_data()

        dataset_name = TEST_FILE.name

    except Exception as error:

        st.error(
            f"Unable to load bundled test data:\n{error}"
        )

        st.stop()


# ============================================================
# DOWNLOAD BUNDLED TEST DATA
# ============================================================

if TEST_FILE.exists():

    try:

        bundled_test_df = load_bundled_test_data()

        csv_data = bundled_test_df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.sidebar.download_button(
            label="⬇️ Download Bundled Test Data",
            data=csv_data,
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception:
        pass


# ============================================================
# PREPARE DATASET
# ============================================================

try:

    X_test, y_test = prepare_test_data(
        test_df
    )

    expected_features = (
        get_expected_features()
    )

    X_test, extra_features = validate_features(
        X_test,
        expected_features
    )

except Exception as error:

    st.error(
        f"Dataset validation failed:\n\n{error}"
    )

    st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.info(
    f"""
    **Dataset in use:** `{dataset_name}`

    **Samples:** {len(test_df):,}  
    **Features:** {X_test.shape[1]}  
    **Target:** `{TARGET_COLUMN}`  
    **Classes:** {y_test.nunique()}
    """
)

if extra_features:

    st.warning(
        "The following extra columns were ignored because "
        "they were not used during model training:\n\n"
        + ", ".join(extra_features)
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Dataset Overview'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Samples",
        f"{len(test_df):,}"
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
        TARGET_COLUMN
    )


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

distribution_df = pd.DataFrame(
    {
        "NSP Class": class_counts.index,
        "Count": class_counts.values,
        "Percentage": class_percentages.values,
    }
)

distribution_df[
    "Percentage"
] = distribution_df[
    "Percentage"
].round(2)


with st.expander(
    "View Class Distribution"
):

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        st.dataframe(
            distribution_df,
            use_container_width=True,
            hide_index=True
        )

    with col2:

        majority_class = (
            class_percentages.idxmax()
        )

        minority_class = (
            class_percentages.idxmin()
        )

        st.warning(
            f"""
            **Class imbalance**

            Majority class:

            **NSP {majority_class}**
            ({class_percentages.max():.2f}%)

            Minority class:

            **NSP {minority_class}**
            ({class_percentages.min():.2f}%)

            Therefore, weighted metrics should not be
            considered alone.
            """
        )


# ============================================================
# EVALUATE SELECTED MODEL
# ============================================================

selected_config = MODEL_CONFIG[
    selected_model
]

selected_model_path = (
    MODEL_DIRECTORY /
    selected_config["model_file"]
)

try:

    selected_model_object = load_model(
        selected_model_path
    )

except Exception as error:

    st.error(
        f"Unable to load {selected_model}:\n{error}"
    )

    st.stop()


# ------------------------------------------------------------
# Apply scaler
# ------------------------------------------------------------

X_selected = X_test.copy()

if selected_config[
    "requires_scaling"
]:

    scaler_path = (
        MODEL_DIRECTORY /
        selected_config["scaler_file"]
    )

    try:

        scaler = load_scaler(
            scaler_path
        )

        X_selected = scaler.transform(
            X_selected
        )

    except Exception as error:

        st.error(
            f"Unable to load scaler for "
            f"{selected_model}:\n{error}"
        )

        st.stop()


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

try:

    selected_predictions = (
        selected_model_object.predict(
            X_selected
        )
    )

except Exception as error:

    st.error(
        f"Prediction failed for {selected_model}:\n{error}"
    )

    st.stop()


# ============================================================
# SELECTED MODEL — TOP METRICS
# ============================================================

selected_metrics = calculate_metrics(
    y_test,
    selected_predictions
)

st.divider()

st.header(
    f"🎯 Selected Model: {selected_model}"
)

st.caption(
    MODEL_CONFIG[
        selected_model
    ]["description"]
)


# ------------------------------------------------------------
# Primary metrics
# ------------------------------------------------------------

metric_cols = st.columns(5)

primary_metrics = [
    ("Accuracy", "Accuracy"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("F1 Score", "F1 Score"),
    ("Balanced Accuracy", "Balanced Accuracy"),
]

for column, (
    label,
    key
) in zip(
    metric_cols,
    primary_metrics
):

    with column:

        st.metric(
            label,
            f"{selected_metrics[key]:.4f}"
        )


# ------------------------------------------------------------
# Additional metrics
# ------------------------------------------------------------

with st.expander(
    "View Additional Metrics"
):

    additional_metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
                "MCC",
            ],
            "Score": [
                selected_metrics[
                    "Macro Precision"
                ],
                selected_metrics[
                    "Macro Recall"
                ],
                selected_metrics[
                    "Macro F1"
                ],
                selected_metrics[
                    "MCC"
                ],
            ],
        }
    )

    st.dataframe(
        additional_metrics_df.style.format(
            {
                "Score": "{:.4f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SELECTED MODEL — CONFUSION MATRIX + REPORT
# ============================================================

st.subheader(
    "Selected Model Evaluation"
)

selected_col1, selected_col2 = st.columns(
    [1, 1]
)

labels = sorted(
    y_test.unique()
)

with selected_col1:

    st.markdown(
        "### Confusion Matrix"
    )

    cm_fig = plot_confusion_matrix(
        y_test,
        selected_predictions,
        selected_model,
        labels
    )

    st.pyplot(
        cm_fig,
        use_container_width=True
    )

    plt.close(
        cm_fig
    )


with selected_col2:

    st.markdown(
        "### Classification Report"
    )

    selected_report_df = (
        get_classification_report_df(
            y_test,
            selected_predictions
        )
    )

    st.dataframe(
        selected_report_df.style.format(
            {
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ALL MODEL EVALUATION
# ============================================================

st.divider()

st.header(
    "📊 All Model Evaluation"
)

st.caption(
    "All available trained models are evaluated on the same "
    "dataset selected above."
)


all_results = {}

all_predictions = {}

all_models = {}

model_errors = {}


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

        all_models[
            model_name
        ] = model

        # ----------------------------------------------------
        # Prepare features
        # ----------------------------------------------------

        X_input = X_test.copy()

        # ----------------------------------------------------
        # Scaling
        # ----------------------------------------------------

        if config[
            "requires_scaling"
        ]:

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
        # Predict
        # ----------------------------------------------------

        y_pred = model.predict(
            X_input
        )

        all_predictions[
            model_name
        ] = y_pred

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        all_results[
            model_name
        ] = calculate_metrics(
            y_test,
            y_pred
        )

    except Exception as error:

        model_errors[
            model_name
        ] = str(error)


# ============================================================
# DISPLAY MODEL ERRORS
# ============================================================

if model_errors:

    st.warning(
        "Some models could not be evaluated."
    )

    for model_name, error in model_errors.items():

        st.error(
            f"**{model_name}**: {error}"
        )


if not all_results:

    st.error(
        "No models were successfully evaluated."
    )

    st.stop()


# ============================================================
# ALL MODEL METRICS TABLE
# ============================================================

all_results_df = pd.DataFrame(
    all_results
).T

all_results_df = all_results_df.sort_values(
    "Recall",
    ascending=False
)

st.subheader(
    "Evaluation Metrics — All Models"
)

display_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "Macro Recall",
    "Balanced Accuracy",
    "MCC",
]

st.dataframe(
    all_results_df[
        display_columns
    ].style.format(
        "{:.4f}"
    ),
    use_container_width=True
)


# ============================================================
# ALL MODEL METRIC CHART
# ============================================================

st.subheader(
    "Metric Comparison"
)

metric_fig = plot_metric_comparison(
    all_results_df
)

st.pyplot(
    metric_fig,
    use_container_width=True
)

plt.close(
    metric_fig
)


# ============================================================
# ALL MODEL CONFUSION MATRICES
# ============================================================

st.divider()

st.header(
    "🔲 Confusion Matrices — All Models"
)

for model_name, y_pred in all_predictions.items():

    with st.expander(
        f"{model_name} — Confusion Matrix & Classification Report",
        expanded=False
    ):

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            fig = plot_confusion_matrix(
                y_test,
                y_pred,
                model_name,
                labels
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        with col2:

            report_df = (
                get_classification_report_df(
                    y_test,
                    y_pred
                )
            )

            st.dataframe(
                report_df.style.format(
                    {
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1 Score": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

st.divider()

st.header(
    "🌲 Random Forest Feature Importance"
)

rf_model = all_models.get(
    "Random Forest"
)

if rf_model is not None:

    rf_fig = plot_feature_importance(
        rf_model,
        X_test.columns,
        top_n=10
    )

    if rf_fig is not None:

        st.pyplot(
            rf_fig,
            use_container_width=True
        )

        plt.close(
            rf_fig
        )

        st.caption(
            "Feature importance reflects model behavior and "
            "should not be interpreted as clinical causality."
        )


# ============================================================
# BASELINE INTERPRETATION
# ============================================================

st.divider()

st.header(
    "📌 Baseline Interpretation"
)

best_model = all_results_df[
    "Recall"
].idxmax()

best_recall = all_results_df.loc[
    best_model,
    "Recall"
]

best_macro_model = all_results_df[
    "Macro Recall"
].idxmax()

best_macro_recall = all_results_df.loc[
    best_macro_model,
    "Macro Recall"
]

st.success(
    f"""
    **Best model by weighted Recall: {best_model}**

    Weighted Recall: **{best_recall:.4f}**

    Best model by Macro Recall:
    **{best_macro_model}**

    Macro Recall: **{best_macro_recall:.4f}**
    """
)

st.markdown(
    """
    ### Why both metrics matter

    **Weighted Recall** accounts for the number of observations
    in each class. Because NSP 1 is the majority class, it can
    have a strong influence on this metric.

    **Macro Recall** calculates Recall independently for each
    class and then gives every class equal importance.

    Therefore, for this imbalanced dataset, the final model
    should not be selected using weighted Recall alone.


    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Cardiotocography ML Assignment | "
    "Baseline Evaluation | "
    "Models loaded from .pkl files "
)