# Classification-Model_E2E-workflow


# a. Problem Statement

Cardiotocography (CTG) is used during pregnancy and labor to monitor fetal well-being through measurements related to fetal heart rate and uterine contractions.

The UCI Cardiotocography dataset contains measurements extracted from cardiotocograms and expert-assigned fetal state classifications.

The objective of this project is to develop a **multi-class classification system** that predicts the fetal state represented by the `NSP` target:

| NSP Class | Meaning |
|---|---|
| `1` | Normal |
| `2` | Suspect |
| `3` | Pathologic |

This is treated as a three-class classification problem.

Because the classes are imbalanced, evaluation is not based on accuracy alone. Macro-level metrics such as Macro Recall and Macro F1 are also considered to understand how well the models perform across all three classes.

---

# b. Dataset Description

## 3.1 Dataset Source

The project uses the **Cardiotocography dataset** from the UCI Machine Learning Repository.

**Dataset:** Cardiotocography  
**UCI Dataset ID:** 193

Official source:

https://archive.ics.uci.edu/dataset/193/cardiotocography

The UCI repository describes the dataset as containing measurements of fetal heart rate (FHR) and uterine contraction (UC) features from cardiotocograms classified by expert obstetricians.

The original dataset contains:

- **2,126 instances**
- **21 predictive features**
- **3-class fetal-state classification**
- No missing values reported by UCI

The dataset can also be used for a 10-class morphological classification task, but this project focuses on the three-class `NSP` fetal-state classification.

## Dataset Variables

The project uses the following 21 predictive features:

| # | Feature | Description |
|---:|---|---|
| 1 | `LB` | FHR baseline |
| 2 | `AC` | Number of accelerations |
| 3 | `FM` | Number of fetal movements |
| 4 | `UC` | Number of uterine contractions |
| 5 | `DL` | Number/rate of light decelerations |
| 6 | `DS` | Number/rate of severe decelerations |
| 7 | `DP` | Number/rate of prolonged decelerations |
| 8 | `ASTV` | Percentage of abnormal short-term variability |
| 9 | `MSTV` | Mean value of short-term variability |
| 10 | `ALTV` | Percentage of abnormal long-term variability |
| 11 | `MLTV` | Mean value of long-term variability |
| 12 | `Width` | Width of FHR histogram |
| 13 | `Min` | Minimum FHR histogram value |
| 14 | `Max` | Maximum FHR histogram value |
| 15 | `Nmax` | Number of histogram peaks |
| 16 | `Nzeros` | Number of histogram zeros |
| 17 | `Mode` | Histogram mode |
| 18 | `Mean` | Histogram mean |
| 19 | `Median` | Histogram median |
| 20 | `Variance` | Histogram variance |
| 21 | `Tendency` | Histogram tendency |

### Target Variable

`NSP`

| Value | Class |
|---:|---|
| `1` | Normal |
| `2` | Suspect |
| `3` | Pathologic |

The UCI dataset also contains a `CLASS` variable representing the FHR pattern class code. For this project, `CLASS` is treated as a **non-predictive column** and removed before model training.

---

## Dataset Preparation

The original UCI dataset contains 2,126 observations.

The project creates a fixed train/test split before model development.

| Dataset | Samples | Approx. Proportion |
|---|---:|---:|
| Training | 1,700 | 80% |
| Testing | 426 | 20% |
| Total | 2,126 | 100% |

The test dataset is stored in the repository as:

text
test_data.csv

---

# c. **GitHub Repository**  https://github.com/BalamuruganWILP/Classification-Model_E2E-workflow 


# d. Models Used & Evaluation Comparison

Six distinct classification frameworks were trained and evaluated on identical splits of the dataset. Performance metrics were tracked across six core statistical parameters:
TBD

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8850** | **0.9614** | **0.8893** | **0.8850** | **0.8855** | **0.6834** |
| **Decision Tree** | **0.9014** | **0.8580** | **0.8983** | **0.9014** | **0.8994** | **0.7252** |
| **kNN** | **0.8709** | **0.9393**| **0.8605** | **0.8709** | **0.8626** | **0.6191** |
| **Naive Bayes** | **0.7981** | **0.8694** | **0.8619** | **0.7981** | **0.8174** | **0.5710** |
| **Random Forest (Ensemble)** | **0.9249** | **0.9787** | **0.9218** | **0.9249** | **0.9220** | **0.7873** |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Provides a strong linear baseline with **88.50% accuracy**. Its recall is relatively strong overall, but its performance on the minority classes is lower than the tree-based ensemble model. |
| **Decision Tree** | Performs better than Logistic Regression, achieving **90.14% accuracy** and **80.92% macro recall**. The model is able to capture non-linear relationships between the CTG features. |
| **kNN** | Achieves **87.09% accuracy**, but its **69.85% macro recall** is lower than Logistic Regression and Decision Tree. Its distance-based nature makes appropriate feature scaling important. |
| **Naive Bayes** | Has the lowest overall accuracy at **79.81%**. Although its macro recall is **73.48%**, its lower precision and F1 score indicate weaker overall classification performance for this dataset. |
| **Random Forest (Ensemble)** | Provides the strongest overall baseline performance, achieving **92.49% accuracy**, **82.85% macro recall**, **84.99% macro F1**, and **78.73% MCC**. It performs particularly well because it can capture non-linear relationships and interactions among the CTG features. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** – it achieves the best baseline performance across Accuracy, Macro Recall, Macro F1 and MCC, making it the strongest overall model among the five evaluated models. |
---


# Cardiotocography Classification Analysis & Streamlit Deployment

# Project Links

| Resource | Link |
|---|---|
| **GitHub Repository** | https://github.com/BalamuruganWILP/Classification-Model_E2E-workflow |
| **Live Streamlit Application** | `https://bala2d36ml2assignment.streamlit.app/` |
| **Dataset Source** | https://archive.ics.uci.edu/dataset/193/cardiotocography |

### Live Application

The Streamlit application provides an interactive frontend for evaluating the trained classification models.

The application supports:

- Built-in test dataset
- User-uploaded CSV test data
- Model selection
- Evaluation metrics
- Confusion matrix
- Classification report
- Comparison of all trained models
- Random Forest feature importance

**Live App:** `https://bala2d36ml2assignment.streamlit.app/`

---




## 💻 Streamlit Web Application Features

The deployed interface allows real-time interactive model inference with the following components:
* **Dataset Upload Option (CSV):** Allows users to drop their custom `test_data.csv` partitions directly into the UI engine.
* **Model Selection Dropdown:** Dynamically switches execution layers between any of the 6 trained architectures.
* **Display of Evaluation Metrics:** Immediately renders computed performance parameters based on the uploaded data.
* **Confusion Matrix & Classification Report:** Visually presents structural class breakdowns for granular diagnostic evaluation.

# Cardiotocography Classification – End-to-End ML Workflow

## Machine Learning Model Development, Evaluation and Streamlit Deployment

This repository contains the complete end-to-end implementation for the Machine Learning classification assignment using the **UCI Cardiotocography (CTG) dataset**.

The project covers:

- Dataset acquisition
- Data validation and preprocessing
- Feature and target separation
- Class imbalance analysis
- Train/test dataset preparation
- Feature scaling
- Model development
- Model evaluation
- Model serialization using `.pkl`
- Streamlit-based interactive evaluation
- GitHub-based project organization
- Deployment-ready application structure

The objective is to build and evaluate multiple classification models for predicting the fetal state represented by the `NSP` target variable and expose the trained models through an interactive Streamlit application.

---
