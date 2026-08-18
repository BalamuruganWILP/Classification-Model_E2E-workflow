import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CTG Classification Model Evaluation",
    page_icon="🫀",
    layout="wide"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")

TEST_DATA_FILE = os.path.join(
    BASE_DIR,
    "test_data.csv"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

TARGET_COLUMN = "NSP"

NON_PREDICTIVE_COLUMNS = [
    "CLASS"
]

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
    "Tendency"
]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {

    "Logistic Regression": {
        "model_file": "logistic_regression.pkl",
        "scaler_file": "logistic_regression_scaler.pkl",
        "requires_scaling": True
    },

    "Decision Tree": {
        "model_file": "decision_tree.pkl",
        "scaler_file": None,
        "requires_scaling": False
    },

    "KNN": {
        "model_file": "knn.pkl",
        "scaler_file": "knn_scaler.pkl",
        "requires_scaling": True
    },

    "Naive Bayes": {
        "model_file": "naive_bayes.pkl",
        "scaler_file": None,
        "requires_scaling": False
    },

    "Random Forest": {
        "model_file": "random_forest.pkl",
        "scaler_file": None,
        "requires_scaling": False
    }
}


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🫀 Cardiotocography Classification")
st.subheader("Machine Learning Model Evaluation Dashboard")

st.markdown(
    """
    This application evaluates five classification models trained on
    the UCI Cardiotocography dataset.

    **Target variable:** `NSP`

    - **1:** Normal
    - **2:** Suspect
    - **3:** Pathologic
    """
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_csv(file):

    return pd.read_csv(file)


@st.cache_resource
def load_model(model_path):

    return joblib.load(model_path)


def prepare_dataset(df):

    """
    Validate and prepare dataset for model prediction.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Check target column
    # --------------------------------------------------------

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from the dataset."
        )

    # --------------------------------------------------------
    # Remove non-predictive columns
    # --------------------------------------------------------

    for column in NON_PREDICTIVE_COLUMNS:

        if column in df.columns:

            df = df.drop(columns=[column])

    # --------------------------------------------------------
    # Check expected feature columns
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "The following required feature columns are missing:\n\n"
            + ", ".join(missing_features)
        )

    # --------------------------------------------------------
    # Select features in correct order
    # --------------------------------------------------------

    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET_COLUMN].copy()

    # --------------------------------------------------------
    # Missing value check
    # --------------------------------------------------------

    if X.isnull().sum().sum() > 0:

        missing_count = X.isnull().sum().sum()

        raise ValueError(
            f"The dataset contains {missing_count} missing feature values."
        )

    if y.isnull().sum() > 0:

        raise ValueError(
            "The target column contains missing values."
        )

    return X, y


def evaluate_model(
    model_name,
    X_test,
    y_test
):

    """
    Load model, apply required preprocessing,
    generate predictions and calculate evaluation metrics.
    """

    config = MODEL_CONFIG[model_name]

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        config["model_file"]
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    model = load_model(model_path)

    # --------------------------------------------------------
    # Apply scaling where required
    # --------------------------------------------------------

    X_input = X_test.copy()

    if config["requires_scaling"]:

        scaler_path = os.path.join(
            MODEL_DIR,
            config["scaler_file"]
        )

        if not os.path.exists(scaler_path):

            raise FileNotFoundError(
                f"Scaler file not found:\n{scaler_path}"
            )

        scaler = load_model(scaler_path)

        X_input = scaler.transform(X_test)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(X_input)

    # Convert predictions to numpy array
    y_pred = np.asarray(y_pred)

    # --------------------------------------------------------
    # Probability predictions for AUC
    # --------------------------------------------------------

    y_proba = None
    auc = np.nan

    if hasattr(model, "predict_proba"):

        try:

            y_proba = model.predict_proba(X_input)

            # ------------------------------------------------
            # Multiclass ROC-AUC
            #
            # One-vs-Rest strategy
            # Macro averaging across classes
            # ------------------------------------------------

            auc = roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="macro"
            )

        except Exception as e:

            st.warning(
                f"AUC could not be calculated for "
                f"{model_name}: {e}"
            )

            auc = np.nan

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[1, 2, 3]
    )

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=[1, 2, 3],
        target_names=[
            "Normal",
            "Suspect",
            "Pathologic"
        ],
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    # --------------------------------------------------------
    # Return everything
    # --------------------------------------------------------

    return {

        "model": model,

        "y_pred": y_pred,

        "y_proba": y_proba,

        "accuracy": accuracy,

        "auc": auc,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "mcc": mcc,

        "balanced_accuracy": balanced_accuracy,

        "confusion_matrix": cm,

        "classification_report": report_df
    }


def display_metrics(results):

    """
    Display the six primary assignment metrics.
    """

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    with col1:

        st.metric(
            "Accuracy",
            f"{results['accuracy']:.4f}"
        )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    with col2:

        if pd.notna(results["auc"]):

            st.metric(
                "AUC",
                f"{results['auc']:.4f}"
            )

        else:

            st.metric(
                "AUC",
                "N/A"
            )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    with col3:

        st.metric(
            "Precision",
            f"{results['precision']:.4f}"
        )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    with col4:

        st.metric(
            "Recall",
            f"{results['recall']:.4f}"
        )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    with col5:

        st.metric(
            "F1",
            f"{results['f1']:.4f}"
        )

    # --------------------------------------------------------
    # MCC
    # --------------------------------------------------------

    with col6:

        st.metric(
            "MCC",
            f"{results['mcc']:.4f}"
        )


def display_confusion_matrix(
    cm,
    title
):

    """
    Display confusion matrix.
    """

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Normal",
            "Suspect",
            "Pathologic"
        ],
        yticklabels=[
            "Normal",
            "Suspect",
            "Pathologic"
        ],
        ax=ax
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_title(
        title
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Dataset Options"
)


# ============================================================
# DATASET SELECTION
# ============================================================

dataset_option = st.sidebar.radio(
    "Choose dataset",
    [
        "Use bundled test data",
        "Upload CSV"
    ]
)


# ============================================================
# LOAD BUNDLED TEST DATA
# ============================================================

if dataset_option == "Use bundled test data":

    if not os.path.exists(TEST_DATA_FILE):

        st.error(
            f"Bundled test data not found:\n{TEST_DATA_FILE}"
        )

        st.stop()

    df = load_csv(
        TEST_DATA_FILE
    )

    st.sidebar.success(
        "Bundled test data loaded"
    )

    # --------------------------------------------------------
    # Download button
    # --------------------------------------------------------

    with open(
        TEST_DATA_FILE,
        "rb"
    ) as file:

        st.sidebar.download_button(
            label="⬇️ Download Test Data",
            data=file,
            file_name="test_data.csv",
            mime="text/csv"
        )


# ============================================================
# UPLOAD CSV
# ============================================================

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload test CSV",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "Please upload a CSV file to continue."
        )

        st.stop()

    try:

        df = load_csv(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"Unable to read CSV file: {e}"
        )

        st.stop()


# ============================================================
# DATASET PREVIEW
# ============================================================

st.header(
    "1. Dataset Information"
)

try:

    X_test, y_test = prepare_dataset(
        df
    )

except Exception as e:

    st.error(
        str(e)
    )

    st.stop()


# ------------------------------------------------------------
# Dataset information
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Test Samples",
        len(X_test)
    )

with col2:

    st.metric(
        "Features",
        X_test.shape[1]
    )

with col3:

    st.metric(
        "Target",
        TARGET_COLUMN
    )

with col4:

    st.metric(
        "Number of Classes",
        y_test.nunique()
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

with st.expander(
    "View Target Class Distribution"
):

    class_distribution = (
        y_test
        .value_counts()
        .sort_index()
        .rename_axis("NSP")
        .reset_index(name="Count")
    )

    class_distribution["Percentage"] = (
        class_distribution["Count"]
        / len(y_test)
        * 100
    )

    st.dataframe(
        class_distribution,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    "View Dataset Preview"
):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# MODEL SELECTION
# ============================================================

st.header(
    "2. Model Selection"
)

selected_model = st.selectbox(
    "Select a model to evaluate",
    list(MODEL_CONFIG.keys())
)


# ============================================================
# EVALUATE SELECTED MODEL
# ============================================================

st.header(
    "3. Selected Model Evaluation"
)

with st.spinner(
    f"Evaluating {selected_model}..."
):

    try:

        selected_results = evaluate_model(
            selected_model,
            X_test,
            y_test
        )

    except Exception as e:

        st.error(
            f"Error evaluating {selected_model}: {e}"
        )

        st.stop()


st.subheader(
    f"{selected_model}"
)


# ============================================================
# DISPLAY SELECTED MODEL METRICS
# ============================================================

display_metrics(
    selected_results
)


# ============================================================
# ADDITIONAL METRICS
# ============================================================

with st.expander(
    "View Additional Metrics"
):

    additional_metrics = pd.DataFrame(
        {
            "Metric": [
                "Balanced Accuracy",
                "Weighted Precision",
                "Weighted Recall",
                "Weighted F1",
                "MCC"
            ],

            "Value": [
                selected_results[
                    "balanced_accuracy"
                ],

                selected_results[
                    "precision"
                ],

                selected_results[
                    "recall"
                ],

                selected_results[
                    "f1"
                ],

                selected_results[
                    "mcc"
                ]
            ]
        }
    )

    additional_metrics["Value"] = (
        additional_metrics["Value"]
        .map(lambda x: f"{x:.4f}")
    )

    st.dataframe(
        additional_metrics,
        use_container_width=True,
        hide_index=True
    )





# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader(
    "Confusion Matrix"
)

display_confusion_matrix(
    selected_results["confusion_matrix"],
    f"{selected_model} - Confusion Matrix"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.subheader(
    "Classification Report"
)

st.dataframe(
    selected_results[
        "classification_report"
    ].style.format(
        {
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }
    ),
    use_container_width=True
)


# ============================================================
# ALL MODEL EVALUATION
# ============================================================

st.header(
    "4. All Model Comparison"
)

all_results = {}

progress_bar = st.progress(
    0
)

model_names = list(
    MODEL_CONFIG.keys()
)

for index, model_name in enumerate(
    model_names
):

    try:

        all_results[model_name] = evaluate_model(
            model_name,
            X_test,
            y_test
        )

    except Exception as e:

        st.error(
            f"Error evaluating {model_name}: {e}"
        )

    progress_bar.progress(
        (index + 1)
        / len(model_names)
    )

progress_bar.empty()


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison_data = []

for model_name, results in all_results.items():

    comparison_data.append(
        {
            "ML Model Name": model_name,

            "Accuracy":
                results["accuracy"],

            "AUC":
                results["auc"],

            "Precision":
                results["precision"],

            "Recall":
                results["recall"],

            "F1":
                results["f1"],

            "MCC":
                results["mcc"]
        }
    )


comparison_df = pd.DataFrame(
    comparison_data
)


st.subheader(
    "Model Performance Comparison"
)


# ------------------------------------------------------------
# Format display
# ------------------------------------------------------------

display_df = comparison_df.copy()

for column in [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]:

    display_df[column] = display_df[column].apply(
        lambda x:
        f"{x:.4f}"
        if pd.notna(x)
        else "N/A"
    )


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# METRIC COMPARISON CHART
# ============================================================

st.subheader(
    "Model Performance Comparison"
)

# ------------------------------------------------------------
# Metrics required by the assignment
# ------------------------------------------------------------

metrics = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]

# ------------------------------------------------------------
# Prepare comparison data
# ------------------------------------------------------------

plot_df = comparison_df[
    [
        "ML Model Name",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
].copy()

plot_df = plot_df.set_index(
    "ML Model Name"
)

# ------------------------------------------------------------
# Create chart
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(13, 6)
)

# X-axis positions
x = np.arange(
    len(plot_df.index)
)

# Six metrics -> narrower bars
width = 0.13

# ------------------------------------------------------------
# Plot each metric
# ------------------------------------------------------------

for i, metric in enumerate(metrics):

    values = plot_df[metric].values

    bars = ax.bar(
        x + (
            i - 2.5
        ) * width,
        values,
        width,
        label=metric
    )

    # --------------------------------------------------------
    # Add values above bars
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        values
    ):

        if pd.notna(value):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,

                value + 0.015,

                f"{value:.3f}",

                ha="center",
                va="bottom",

                fontsize=8,

                rotation=90
            )

# ------------------------------------------------------------
# X-axis
# ------------------------------------------------------------

ax.set_xticks(
    x
)

ax.set_xticklabels(
    plot_df.index,
    rotation=20,
    ha="right"
)

# ------------------------------------------------------------
# Y-axis
# ------------------------------------------------------------

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1.10
)

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

ax.set_title(
    "Performance Comparison — All Models",
    fontsize=15,
    fontweight="bold"
)

# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

ax.grid(
    axis="y",
    alpha=0.25
)

ax.set_axisbelow(
    True
)

# ------------------------------------------------------------
# Remove unnecessary borders
# ------------------------------------------------------------

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.12
    ),
    ncol=3,
    frameon=False
)

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

fig.tight_layout()

# ------------------------------------------------------------
# Display in Streamlit
# ------------------------------------------------------------

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)

# ============================================================
# BEST MODEL
# ============================================================

if not comparison_df.empty:

    best_model_row = comparison_df.loc[
        comparison_df["F1"].idxmax()
    ]

    st.success(
        f"🏆 Best model based on weighted F1: "
        f"**{best_model_row['ML Model Name']}** "
        f"({best_model_row['F1']:.4f})"
    )



# ============================================================
# METRIC COMPARISON CHART
# ============================================================

def plot_metric_comparison(
    results_df
):

    metrics = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
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
# MODEL PERFORMANCE HEATMAP
# ============================================================

st.subheader(
    "Model Performance Comparison"
)

heatmap_df = comparison_df.set_index(
    "ML Model Name"
)[
    [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
].copy()


# ------------------------------------------------------------
# Create heatmap
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 5)
)

sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".4f",
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={
        "label": "Score"
    },
    ax=ax
)

ax.set_title(
    "Classification Model Performance",
    fontsize=16,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "Evaluation Metric",
    fontsize=11
)

ax.set_ylabel(
    "Machine Learning Model",
    fontsize=11
)

plt.xticks(
    rotation=0
)

plt.yticks(
    rotation=0
)

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)

# ============================================================
# ALL MODEL CONFUSION MATRICES
# ============================================================

st.header(
    "5. Confusion Matrices – All Models"
)

for model_name, results in all_results.items():

    with st.expander(
        f"{model_name} – Confusion Matrix"
    ):

        display_confusion_matrix(
            results["confusion_matrix"],
            f"{model_name} - Confusion Matrix"
        )


# ============================================================
# ALL MODEL CLASSIFICATION REPORTS
# ============================================================

st.header(
    "6. Classification Reports – All Models"
)

for model_name, results in all_results.items():

    with st.expander(
        f"{model_name} – Classification Report"
    ):

        st.dataframe(
            results[
                "classification_report"
            ].style.format(
                {
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:.0f}"
                }
            ),
            use_container_width=True
        )


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

if "Random Forest" in all_results:

    st.header(
        "7. Random Forest Feature Importance"
    )

    rf_model = all_results[
        "Random Forest"
    ]["model"]

    if hasattr(
        rf_model,
        "feature_importances_"
    ):

        importance_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,

                "Importance":
                    rf_model.feature_importances_
            }
        )

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Top features table
        # ----------------------------------------------------

        st.subheader(
            "Feature Importance Ranking"
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Feature importance chart
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            ax=ax
        )

        ax.set_title(
            "Random Forest Feature Importance"
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.set_ylabel(
            "Feature"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "Cardiotocography Classification | "
    "Machine Learning Assignment"
)

st.caption(
    "AUC is calculated using multiclass "
    "One-vs-Rest (OvR) ROC-AUC with macro averaging."
)