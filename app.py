"""
Cardiotocography ML Assignment
==============================

Streamlit Evaluation Dashboard

This application evaluates five pre-trained baseline models:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Random Forest

IMPORTANT
---------
- Models are NOT retrained in this application.
- The test dataset remains untouched.
- No SMOTE is applied.
- No undersampling is applied.
- No class balancing is applied.
- Saved scalers are used where required.
- This dashboard represents BASELINE model evaluation.

Project structure expected:

project/
│
├── app.py
├── test_data.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── logistic_regression_scaler.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── knn_scaler.pkl
│   ├── naive_bayes.pkl
│   └── random_forest.pkl
│
└── ...
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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

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
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {

    "Logistic Regression": {
        "model_file": "logistic_regression.pkl",
        "scaler_file": "logistic_regression_scaler.pkl",
        "requires_scaling": True,
        "description": (
            "Linear classification model using logistic "
            "regression."
        ),
    },

    "Decision Tree": {
        "model_file": "decision_tree.pkl",
        "scaler_file": None,
        "requires_scaling": False,
        "description": (
            "Tree-based model using recursive feature "
            "splitting."
        ),
    },

    "KNN": {
        "model_file": "knn.pkl",
        "scaler_file": "knn_scaler.pkl",
        "requires_scaling": True,
        "description": (
            "Distance-based classifier using the nearest "
            "training observations."
        ),
    },

    "Gaussian Naive Bayes": {
        "model_file": "naive_bayes.pkl",
        "scaler_file": None,
        "requires_scaling": False,
        "description": (
            "Probabilistic classifier using Gaussian "
            "feature distributions."
        ),
    },

    "Random Forest": {
        "model_file": "random_forest.pkl",
        "scaler_file": None,
        "requires_scaling": False,
        "description": (
            "Ensemble of decision trees using aggregated "
            "tree predictions."
        ),
    },
}


# ============================================================
# GENERAL CONSTANTS
# ============================================================

TARGET_COLUMN = "NSP"

NON_PREDICTIVE_COLUMNS = [
    "CLASS"
]

RANDOM_STATE = 42


# ============================================================
# CUSTOM CSS
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

    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dddddd;
        background-color: #f8f9fa;
    }

    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0c060;
        background-color: #fff8dc;
    }

    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #7bbf7b;
        background-color: #f0fff0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_test_data():

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_FILE}"
        )

    return pd.read_csv(TEST_FILE)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(model_path):

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    return joblib.load(model_path)


# ============================================================
# SCALER LOADING
# ============================================================

@st.cache_resource
def load_scaler(scaler_path):

    if not scaler_path.exists():

        raise FileNotFoundError(
            f"Scaler file not found:\n{scaler_path}"
        )

    return joblib.load(scaler_path)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_test_data(df):

    """
    Separate target and predictor variables.

    Target:
        NSP

    Non-predictive:
        CLASS
    """

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found."
        )

    y = df[TARGET_COLUMN].copy()

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    # Remove non-predictive column if present
    for column in NON_PREDICTIVE_COLUMNS:

        if column in X.columns:

            X = X.drop(
                columns=[column]
            )

    return X, y


# ============================================================
# METRIC CALCULATION
# ============================================================

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
# PER CLASS METRICS
# ============================================================

def calculate_per_class_metrics(
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

    classes = sorted(
        y_true.unique()
    )

    for class_value in classes:

        class_string = str(class_value)

        if class_string in report:

            rows.append(
                {
                    "Class": f"NSP {class_value}",
                    "Precision": report[
                        class_string
                    ]["precision"],
                    "Recall": report[
                        class_string
                    ]["recall"],
                    "F1 Score": report[
                        class_string
                    ]["f1-score"],
                    "Support": int(
                        report[
                            class_string
                        ]["support"]
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# CONFUSION MATRIX PLOT
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

    # Write values inside cells
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
# GENERIC BAR CHART
# ============================================================

def plot_metric_comparison(
    results_df,
    metrics
):

    plot_df = results_df[
        metrics
    ].copy()

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    x = np.arange(
        len(plot_df.index)
    )

    width = 0.8 / len(metrics)

    for i, metric in enumerate(
        metrics
    ):

        positions = (
            x
            + i * width
            - (
                len(metrics) - 1
            ) * width / 2
        )

        ax.bar(
            positions,
            plot_df[metric],
            width=width,
            label=metric,
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
        "Baseline Model Performance"
    )

    ax.legend(
        loc="upper right"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    return fig


# ============================================================
# RECALL COMPARISON
# ============================================================

def plot_recall_comparison(
    results_df
):

    plot_df = results_df[
        [
            "Macro Recall",
            "Recall"
        ]
    ].copy()

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    x = np.arange(
        len(plot_df.index)
    )

    width = 0.35

    ax.bar(
        x - width / 2,
        plot_df["Recall"],
        width,
        label="Weighted Recall"
    )

    ax.bar(
        x + width / 2,
        plot_df["Macro Recall"],
        width,
        label="Macro Recall"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_df.index,
        rotation=20,
        ha="right"
    )

    ax.set_ylabel(
        "Recall"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_title(
        "Weighted Recall vs Macro Recall"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    return fig


# ============================================================
# PER CLASS RECALL CHART
# ============================================================

def plot_per_class_recall(
    per_class_results
):

    pivot_df = (
        per_class_results
        .pivot(
            index="Model",
            columns="Class",
            values="Recall"
        )
    )

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    x = np.arange(
        len(pivot_df.index)
    )

    number_of_classes = len(
        pivot_df.columns
    )

    width = (
        0.8 /
        number_of_classes
    )

    for i, class_name in enumerate(
        pivot_df.columns
    ):

        positions = (
            x
            + i * width
            - (
                number_of_classes - 1
            ) * width / 2
        )

        ax.bar(
            positions,
            pivot_df[class_name],
            width=width,
            label=class_name,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        pivot_df.index,
        rotation=20,
        ha="right"
    )

    ax.set_ylabel(
        "Recall"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_title(
        "Per-Class Recall Across Models"
    )

    ax.legend(
        title="Class"
    )

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
        "Feature Importance"
    )

    ax.set_title(
        "Random Forest — Top Feature Importance"
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    fig.tight_layout()

    return fig


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🫀 Cardiotocography Classification'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Baseline Machine Learning Model Evaluation Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    """
    This dashboard evaluates five pre-trained machine learning
    models using the **untouched test dataset**.

    Models are loaded from saved `.pkl` files. No model is
    retrained and no SMOTE, undersampling, or class balancing
    is applied during this baseline evaluation.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    test_df = load_test_data()

    X_test, y_test = prepare_test_data(
        test_df
    )

except Exception as error:

    st.error(
        f"Unable to load or prepare the test dataset:\n\n{error}"
    )

    st.stop()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Test Dataset Overview'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Test Samples",
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


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Class Distribution'
    '</div>',
    unsafe_allow_html=True
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

col_distribution, col_warning = st.columns(
    [1.3, 1]
)

with col_distribution:

    st.dataframe(
        distribution_df,
        use_container_width=True,
        hide_index=True
    )

with col_warning:

    majority_percentage = (
        class_percentages.max()
    )

    minority_percentage = (
        class_percentages.min()
    )

    majority_class = (
        class_percentages.idxmax()
    )

    minority_class = (
        class_percentages.idxmin()
    )

    st.warning(
        f"""
        **Class imbalance detected**

        Majority class: **NSP {majority_class}**
        ({majority_percentage:.2f}%)

        Minority class: **NSP {minority_class}**
        ({minority_percentage:.2f}%)

        Weighted metrics may therefore be strongly
        influenced by the majority class.

        Minority-class recall should be examined
        before final model selection.
        """
    )


# ============================================================
# CLASS DISTRIBUTION CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 4)
)

ax.bar(
    [
        f"NSP {x}"
        for x in class_counts.index
    ],
    class_counts.values
)

ax.set_xlabel(
    "NSP Class"
)

ax.set_ylabel(
    "Number of Samples"
)

ax.set_title(
    "Test Set Class Distribution"
)

ax.grid(
    axis="y",
    alpha=0.25
)

for i, value in enumerate(
    class_counts.values
):

    ax.text(
        i,
        value,
        f"{value}",
        ha="center",
        va="bottom"
    )

fig.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# MODEL EVALUATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Model Evaluation'
    '</div>',
    unsafe_allow_html=True
)

results = {}

predictions = {}

per_class_results = {}

confusion_matrices = {}

loaded_models = {}

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

        loaded_models[
            model_name
        ] = model

        # ----------------------------------------------------
        # Prepare test features
        # ----------------------------------------------------

        X_input = X_test.copy()

        # ----------------------------------------------------
        # Apply saved scaler where required
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
        # Prediction
        # ----------------------------------------------------

        y_pred = model.predict(
            X_input
        )

        predictions[
            model_name
        ] = y_pred

        # ----------------------------------------------------
        # Overall metrics
        # ----------------------------------------------------

        results[
            model_name
        ] = calculate_metrics(
            y_test,
            y_pred
        )

        # ----------------------------------------------------
        # Per-class metrics
        # ----------------------------------------------------

        class_df = (
            calculate_per_class_metrics(
                y_test,
                y_pred
            )
        )

        class_df[
            "Model"
        ] = model_name

        per_class_results[
            model_name
        ] = class_df

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        labels = sorted(
            y_test.unique()
        )

        confusion_matrices[
            model_name
        ] = confusion_matrix(
            y_test,
            y_pred,
            labels=labels
        )

    except Exception as error:

        model_errors[
            model_name
        ] = str(error)


# ============================================================
# MODEL LOADING STATUS
# ============================================================

if model_errors:

    st.warning(
        "One or more models could not be evaluated."
    )

    for model_name, error in model_errors.items():

        st.error(
            f"**{model_name}**: {error}"
        )


if not results:

    st.error(
        "No models could be evaluated."
    )

    st.stop()


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
).T


# ============================================================
# MODEL RANKING
# ============================================================

ranking_df = results_df[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Macro Recall",
        "Balanced Accuracy",
    ]
].copy()

ranking_df = ranking_df.sort_values(
    "Recall",
    ascending=False
)


# ============================================================
# BEST MODEL SUMMARY
# ============================================================

best_weighted_recall_model = (
    results_df[
        "Recall"
    ].idxmax()
)

best_weighted_recall = (
    results_df.loc[
        best_weighted_recall_model,
        "Recall"
    ]
)

best_macro_recall_model = (
    results_df[
        "Macro Recall"
    ].idxmax()
)

best_macro_recall = (
    results_df.loc[
        best_macro_recall_model,
        "Macro Recall"
    ]
)


st.markdown(
    '<div class="section-title">'
    'Baseline Model Summary'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Best Weighted Recall",
        best_weighted_recall_model,
        f"{best_weighted_recall:.4f}"
    )

with col2:

    st.metric(
        "Best Macro Recall",
        best_macro_recall_model,
        f"{best_macro_recall:.4f}"
    )

with col3:

    st.metric(
        "Models Evaluated",
        len(results)
    )


st.success(
    f"""
    **Current baseline leader: {best_weighted_recall_model}**

    It achieved the highest weighted Recall on the untouched
    test set: **{best_weighted_recall:.4f}**.

    However, weighted Recall alone should not determine the
    final model because the dataset is imbalanced. Macro Recall
    and per-class Recall are also considered below.
    """
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Model Comparison",
        "🎯 Per-Class Analysis",
        "🔲 Confusion Matrices",
        "🔍 Model Details",
        "🌲 Feature Importance",
    ]
)


# ============================================================
# TAB 1 — MODEL COMPARISON
# ============================================================

with tab1:

    st.subheader(
        "Overall Model Performance"
    )

    display_metrics = ranking_df[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Macro Recall",
            "Balanced Accuracy",
        ]
    ].copy()

    st.dataframe(
        display_metrics.style.format(
            "{:.4f}"
        ),
        use_container_width=True
    )

    st.caption(
        "Recall is the primary metric for this assignment. "
        "Macro Recall gives equal importance to each class, "
        "while weighted Recall accounts for class frequency."
    )

    st.subheader(
        "Metric Comparison"
    )

    metrics_to_plot = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    metric_fig = plot_metric_comparison(
        ranking_df,
        metrics_to_plot
    )

    st.pyplot(
        metric_fig,
        use_container_width=True
    )

    plt.close(metric_fig)

    st.subheader(
        "Weighted Recall vs Macro Recall"
    )

    recall_fig = plot_recall_comparison(
        ranking_df
    )

    st.pyplot(
        recall_fig,
        use_container_width=True
    )

    plt.close(recall_fig)

    st.subheader(
        "Model Ranking by Weighted Recall"
    )

    ranking_display = ranking_df[
        [
            "Recall",
            "Macro Recall",
            "F1 Score",
            "Balanced Accuracy",
        ]
    ].copy()

    ranking_display.insert(
        0,
        "Rank",
        range(
            1,
            len(ranking_display) + 1
        )
    )

    st.dataframe(
        ranking_display.style.format(
            {
                "Recall": "{:.4f}",
                "Macro Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "Balanced Accuracy": "{:.4f}",
            }
        ),
        use_container_width=True
    )


# ============================================================
# TAB 2 — PER CLASS ANALYSIS
# ============================================================

with tab2:

    st.subheader(
        "Per-Class Performance"
    )

    st.markdown(
        """
        Because the dataset is imbalanced, overall weighted metrics
        can hide poor performance on minority classes.

        The table below shows Precision, Recall and F1 Score
        separately for each NSP class.
        """
    )

    all_per_class = pd.concat(
        per_class_results.values(),
        ignore_index=True
    )

    all_per_class_display = all_per_class[
        [
            "Model",
            "Class",
            "Precision",
            "Recall",
            "F1 Score",
            "Support",
        ]
    ]

    st.dataframe(
        all_per_class_display.style.format(
            {
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Per-Class Recall Comparison"
    )

    per_class_recall_fig = plot_per_class_recall(
        all_per_class
    )

    st.pyplot(
        per_class_recall_fig,
        use_container_width=True
    )

    plt.close(
        per_class_recall_fig
    )

    st.subheader(
        "Detailed Model Analysis"
    )

    selected_model = st.selectbox(
        "Select a model",
        list(predictions.keys())
    )

    selected_class_df = (
        per_class_results[
            selected_model
        ]
        .copy()
    )

    selected_class_df = (
        selected_class_df[
            [
                "Class",
                "Precision",
                "Recall",
                "F1 Score",
                "Support",
            ]
        ]
    )

    st.dataframe(
        selected_class_df.style.format(
            {
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    # Identify weakest class for selected model
    weakest_class_row = (
        selected_class_df
        .loc[
            selected_class_df[
                "Recall"
            ].idxmin()
        ]
    )

    st.warning(
        f"""
        For **{selected_model}**, the lowest class-specific
        Recall is for **{weakest_class_row["Class"]}**:

        **Recall = {weakest_class_row["Recall"]:.4f}**

        This is important when evaluating the effect of class
        balancing and SMOTE later.
        """
    )


# ============================================================
# TAB 3 — CONFUSION MATRICES
# ============================================================

with tab3:

    st.subheader(
        "Confusion Matrices"
    )

    st.markdown(
        """
        Rows represent the **actual NSP class** and columns
        represent the **predicted NSP class**.

        Values on the diagonal represent correct predictions.
        Off-diagonal values represent classification errors.
        """
    )

    selected_cm_model = st.selectbox(
        "Select model",
        list(confusion_matrices.keys()),
        key="confusion_model"
    )

    labels = sorted(
        y_test.unique()
    )

    cm_fig = plot_confusion_matrix(
        y_test,
        predictions[
            selected_cm_model
        ],
        selected_cm_model,
        labels
    )

    st.pyplot(
        cm_fig,
        use_container_width=False
    )

    plt.close(cm_fig)

    st.subheader(
        "All Model Confusion Matrices"
    )

    for model_name in confusion_matrices:

        with st.expander(
            model_name
        ):

            fig = plot_confusion_matrix(
                y_test,
                predictions[
                    model_name
                ],
                model_name,
                labels
            )

            st.pyplot(
                fig,
                use_container_width=False
            )

            plt.close(fig)


# ============================================================
# TAB 4 — MODEL DETAILS
# ============================================================

with tab4:

    st.subheader(
        "Model Configuration"
    )

    selected_detail_model = st.selectbox(
        "Select model",
        list(MODEL_CONFIG.keys()),
        key="details_model"
    )

    config = MODEL_CONFIG[
        selected_detail_model
    ]

    model_object = loaded_models.get(
        selected_detail_model
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Configuration"
        )

        st.write(
            f"**Model:** {selected_detail_model}"
        )

        st.write(
            f"**Scaling required:** "
            f"{'Yes' if config['requires_scaling'] else 'No'}"
        )

        if config["requires_scaling"]:

            st.write(
                f"**Scaler:** "
                f"{config['scaler_file']}"
            )

        st.write(
            f"**Model file:** "
            f"{config['model_file']}"
        )

        st.write(
            f"**Test samples:** "
            f"{len(X_test)}"
        )

        st.write(
            f"**Features:** "
            f"{X_test.shape[1]}"
        )

    with col2:

        st.markdown(
            "### Description"
        )

        st.write(
            config["description"]
        )

        st.markdown(
            "### Baseline Metrics"
        )

        model_metrics = results[
            selected_detail_model
        ]

        for metric_name in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Macro Recall",
            "Balanced Accuracy",
        ]:

            st.write(
                f"**{metric_name}:** "
                f"{model_metrics[metric_name]:.4f}"
            )

    # Show common model attributes where available
    if model_object is not None:

        st.markdown(
            "### Model Parameters"
        )

        try:

            params = model_object.get_params()

            parameter_df = pd.DataFrame(
                {
                    "Parameter": params.keys(),
                    "Value": [
                        str(value)
                        for value in params.values()
                    ],
                }
            )

            st.dataframe(
                parameter_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception:

            st.info(
                "Model parameters could not be displayed."
            )


# ============================================================
# TAB 5 — FEATURE IMPORTANCE
# ============================================================

with tab5:

    st.subheader(
        "Random Forest Feature Importance"
    )

    st.markdown(
        """
        Random Forest provides model-derived feature importance
        values based on the contribution of features to tree
        splitting.

        **Important:** These values indicate model behavior.
        They should not be interpreted as clinical causality.
        """
    )

    rf_model = loaded_models.get(
        "Random Forest"
    )

    if rf_model is None:

        st.warning(
            "Random Forest model is not available."
        )

    elif not hasattr(
        rf_model,
        "feature_importances_"
    ):

        st.warning(
            "The loaded Random Forest model does not provide "
            "feature importance."
        )

    else:

        feature_importance_df = pd.DataFrame(
            {
                "Feature": X_test.columns,
                "Importance": (
                    rf_model.feature_importances_
                ),
            }
        )

        feature_importance_df = (
            feature_importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        feature_importance_df.insert(
            0,
            "Rank",
            range(
                1,
                len(feature_importance_df) + 1
            )
        )

        st.dataframe(
            feature_importance_df.style.format(
                {
                    "Importance": "{:.4f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Top 10 Features"
        )

        importance_fig = plot_feature_importance(
            rf_model,
            X_test.columns,
            top_n=10
        )

        if importance_fig is not None:

            st.pyplot(
                importance_fig,
                use_container_width=True
            )

            plt.close(
                importance_fig
            )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

st.divider()

st.header(
    "Baseline Interpretation"
)

st.markdown(
    f"""
### Current finding

**{best_weighted_recall_model}** is the strongest baseline
model according to weighted Recall, with a score of
**{best_weighted_recall:.4f}**.

However, the dataset is imbalanced. Therefore, the difference
between **weighted Recall** and **Macro Recall** is important.

### What we should investigate next

The baseline results should now be used to answer:

1. How well does each model detect the majority class?
2. How well does each model detect NSP 2?
3. How well does each model detect NSP 3?
4. Does class imbalance reduce minority-class Recall?
5. Does SMOTE improve minority-class performance?
6. Does improving minority Recall negatively affect overall
   performance?

The next experimental stage should therefore compare the
baseline models against models trained using appropriate
class-balancing techniques.
"""
)


# ============================================================
# BASELINE STATUS
# ============================================================

st.info(
    """
    **Baseline status**

    ✓ Test set remains untouched  
    ✓ Models are loaded from saved `.pkl` files  
    ✓ Saved scalers are used where required  
    ✓ No SMOTE  
    ✓ No undersampling  
    ✓ No class weighting  
    ✓ No retraining  

    This provides a clean baseline for the next
    class-imbalance experiments.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Cardiotocography ML Assignment | "
    "Baseline Evaluation Dashboard | "
    "Test Set Remains Untouched"
)