# AI-Based Disaster Risk Prediction Using SHAP Analysis

## Project Description

This project develops an AI-based disaster risk prediction system
using World Risk Index data.

The system uses machine learning techniques to classify
disaster-risk levels and provide model explanations using SHAP.

The project will also include a Streamlit dashboard for
visualizing predictions, risk factors, and model results.

---

## Project Objectives

1. Load and preprocess World Risk Index data.
2. Perform exploratory data analysis.
3. Prepare features for machine learning.
4. Create disaster-risk categories.
5. Train machine learning models.
6. Use XGBoost for risk prediction.
7. Evaluate model performance.
8. Use SHAP for model explainability.
9. Perform temporal analysis and prediction.
10. Build an interactive Streamlit dashboard.

---

## Main Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib
- Jupyter
- OpenPyXL

---

## Dataset

The project will use World Risk Index data containing
country-level disaster risk observations across multiple years.

The dataset will be investigated and documented during
the data-preparation phase.

---

## Project Components

### 1. Data Preprocessing

Cleaning, validation, handling missing values, and preparing
the dataset for machine learning.

### 2. Exploratory Data Analysis

Analysis of disaster-risk patterns using statistical
summaries and visualizations.

### 3. Risk Classification

Creation of appropriate disaster-risk categories based
on the project dataset and documented methodology.

### 4. Machine Learning

Machine learning models will be trained to predict
disaster-risk categories.

### 5. XGBoost

XGBoost will be evaluated as the primary prediction model.

### 6. Model Evaluation

The models will be evaluated using appropriate
classification metrics.

### 7. SHAP Explainability

SHAP will be used to understand which features contribute
to model predictions.

### 8. Temporal Analysis

Changes in disaster-risk patterns across years will
be investigated.

### 9. Streamlit Dashboard

An interactive dashboard will present:

- Risk predictions
- Risk categories
- Important features
- Model explanations
- Data visualizations
- Temporal trends

---

## Project Structure

```text
open_lab/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── src/
│
├── app.py
├── requirements.txt
└── README.md
