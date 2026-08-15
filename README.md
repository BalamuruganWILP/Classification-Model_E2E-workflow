# Classification-Model_E2E-workflow

# Cardiotocography Classification Analysis & Streamlit Deployment

This repository contains the source code, trained models, test data, and deployment configurations for **Assignment 2** of the **M.Tech (AIML/ DSE) Machine Learning** course. The objective is to build, evaluate, and deploy an end-to-end multi-class classification system using cardiotocogram recordings.

* **Live Streamlit App:** [TBD]  
* **GitHub Repository:** [[GithubRepo](https://github.com/BalamuruganWILP/Classification-Model_E2E-workflow.git)]  

---

## a. Problem Statement
Fetal hypoxia and birth asphyxia remain prominent concerns during labor and delivery. Early automated processing of Cardiotocograms (CTGs) can provide reliable decision-making tools for obstetricians. This project addresses a **multi-class classification problem** using measurements derived from fetal heart rate (FHR) and uterine contractions (UC). The task is to accurately classify the fetal state into one of three clinical conditions:
1. **Normal (N)**
2. **Suspect (S)**
3. **Pathologic (P)**

---

## b. Dataset Description
* **Dataset Name:** UCI Cardiotocography Dataset  
* **Total Instances:** 2,126 fetal cardiotocograms automatically processed.  
* **Total Features Selected:** 21 predictive features.  
* **Target Variable:** `NSP` (Fetal state class code: N=normal; S=suspect; P=pathologic).  
* **Key Predictive Features Used:**
  * `LB`: FHR baseline (beats per minute)
  * `AC`: Number of accelerations per second
  * `FM`: Number of fetal movements per second
  * `UC`: Number of uterine contractions per second
  * `DL` / `DS` / `DP`: Rates of light, severe, and prolonged decelerations
  * `ASTV` / `ALTV`: Percentage of time with abnormal short-term and long-term variability
  * `MSTV` / `MLTV`: Mean value of short-term and long-term variability

---

## c. GitHub Repository Link
* **Repository URL:** [[Cardiotocography](https://archive.ics.uci.edu/dataset/193/cardiotocography)]

### Project Folder Directory Structure
TBD

---

## d. Models Used & Evaluation Comparison

Six distinct classification frameworks were trained and evaluated on identical splits of the dataset. Performance metrics were tracked across six core statistical parameters:
TBD

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | *0.0000* | *0.0000* | *0.0000* | *0.0000* | *0.0000* | *0.0000* |
| **Decision Tree** | *0.0000* | *0.0000* | *0.0000* | *0.0000* | *0.0000* | *0.0000* |
| **kNN** | *0.0000* | *0.0000* | *0.0000* | *0.0000* | *0.0000* | *0.0000* |
| **Naive Bayes** | *0.0000* | *0.0000* | *0.0000* | *0.0000* | *0.0000* | *0.0000* |
| **Random Forest (Ensemble)** | *0.0000* | *0.0000* | *0.0000* | *0.0000* | *0.0000* | *0.0000* |

*(Note: Replace placeholder 0.0000 values with your explicit 6-decimal experimental metrics from BITS Virtual Lab).*

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** |. |
| **Decision Tree** | . |
| **kNN** | `. |
| **Naive Bayes** | . |
| **Random Forest (Ensemble)**| . |
| **Overall Winner for your dataset?** | **[TBD]** . |

---

## 💻 Streamlit Web Application Features

The deployed interface allows real-time interactive model inference with the following components:
* **Dataset Upload Option (CSV):** Allows users to drop their custom `test_data.csv` partitions directly into the UI engine.
* **Model Selection Dropdown:** Dynamically switches execution layers between any of the 6 trained architectures.
* **Display of Evaluation Metrics:** Immediately renders computed performance parameters based on the uploaded data.
* **Confusion Matrix & Classification Report:** Visually presents structural class breakdowns for granular diagnostic evaluation.

