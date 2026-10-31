# AI-Based Disaster Risk Prediction Using SHAP Analysis

## Project Description

This project develops an AI-based disaster risk prediction system using World Risk Index data.

The system uses machine learning techniques to classify disaster-risk levels and provide model explanations using SHAP (SHapley Additive exPlanations).

The project also includes an interactive Streamlit dashboard for visualizing predictions, risk factors, model performance, and explainability results.

The main goal is to predict disaster-risk levels, understand the factors influencing disaster risk, and present the results through an interactive web-based dashboard.

---

## Project Objectives

1. Load and preprocess World Risk Index data.
2. Perform exploratory data analysis (EDA).
3. Prepare features for machine learning.
4. Create disaster-risk categories.
5. Train machine learning models.
6. Use XGBoost for disaster-risk prediction.
7. Evaluate model performance using classification metrics.
8. Use SHAP for model explainability.
9. Perform temporal analysis and prediction.
10. Build an interactive Streamlit dashboard.
11. Visualize disaster-risk patterns, predictions, and important risk factors.

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
- Jupyter Notebook
- OpenPyXL

---

## Dataset

The project uses World Risk Index data containing country-level disaster-risk observations across multiple years.

The dataset will be investigated and documented during the data-preparation phase.

The data will be used to identify disaster-risk patterns, prepare machine learning features, and develop predictive models.

---

## Project Components

### 1. Data Preprocessing

- Loading the World Risk Index dataset.
- Checking data types and structure.
- Handling missing values.
- Removing duplicate records.
- Cleaning and validating the dataset.
- Preparing the data for machine learning.

### 2. Exploratory Data Analysis (EDA)

- Statistical analysis of the dataset.
- Distribution of disaster-risk indicators.
- Country-level risk comparisons.
- Correlation analysis.
- Visualization of disaster-risk patterns.

### 3. Risk Classification

- Creating appropriate disaster-risk categories.
- Defining classification thresholds based on the dataset.
- Preparing target variables for machine learning.

### 4. Machine Learning

- Training machine learning classification models.
- Comparing model performance.
- Identifying the most suitable model for disaster-risk prediction.

### 5. XGBoost

XGBoost will be evaluated as the primary prediction model.

It will be used to classify disaster-risk levels and identify important predictive features.

### 6. Model Evaluation

The trained models will be evaluated using appropriate classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 7. SHAP Explainability

SHAP will be used to explain machine learning predictions.

The analysis will include:

- SHAP Summary Plot.
- SHAP Feature Importance.
- SHAP Dependence Plot.
- Individual Prediction Explanations.

This will help identify which features contribute most to disaster-risk predictions.

### 8. Temporal Analysis

- Investigating disaster-risk changes across different years.
- Analyzing country-level risk trends.
- Identifying changes in disaster-risk patterns.
- Exploring temporal prediction possibilities.

### 9. Streamlit Dashboard

An interactive Streamlit dashboard will present the results of the disaster-risk prediction system.

The dashboard will include:

- Disaster-risk predictions.
- Risk categories.
- Important features.
- Model performance metrics.
- SHAP explainability visualizations.
- Data visualizations.
- Country-level comparisons.
- Temporal trends.

---

## Streamlit Application

The project includes a Streamlit web application that provides an interactive interface for exploring disaster-risk predictions and model results.

### Running the Streamlit Application

To start the application, open the terminal in the project directory and execute:

```bash
streamlit run app.py
```

After executing the command, Streamlit starts the application server and provides access URLs.

### Application Access URLs

The current development environment provides the following URLs:

**Local URL:**
```text
http://localhost:8501
```

**Network URL:**
```text
http://10.113.24.29:8501
```

The Local URL can be used to access the application from the same machine.

The Network URL can be used to access the application from another device connected to the same network, provided network access and firewall settings allow it.

**Note:** The Network URL may change depending on the development environment and network configuration.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd open_lab
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available through the Local URL provided by Streamlit.

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
```

### Folder Description

| Folder/File | Description |
|---|---|
| `data/raw/` | Original World Risk Index dataset |
| `data/processed/` | Cleaned and processed datasets |
| `models/` | Trained machine learning models |
| `notebooks/` | Jupyter notebooks for analysis and experiments |
| `outputs/figures/` | Generated graphs and visualizations |
| `outputs/reports/` | Model evaluation and analysis reports |
| `src/` | Python source code and reusable modules |
| `app.py` | Main Streamlit application |
| `requirements.txt` | Required Python libraries |
| `README.md` | Project documentation |

---

## Expected Outcomes

The project aims to deliver:

1. A cleaned and processed World Risk Index dataset.
2. Exploratory data analysis and visualizations.
3. Machine learning models for disaster-risk classification.
4. An XGBoost-based disaster-risk prediction model.
5. Model evaluation and performance comparisons.
6. SHAP-based model explainability.
7. Temporal disaster-risk analysis.
8. An interactive Streamlit dashboard.
9. Visual insights into the factors influencing disaster risk.

---

## Project Goal

The ultimate goal of this project is to develop an explainable AI-based disaster risk prediction system that can classify disaster-risk levels, identify the most influential risk factors, analyze changes over time, and present meaningful insights through an interactive Streamlit dashboard.

**The project combines Machine Learning, XGBoost, SHAP Explainability, and Streamlit to support better understanding and analysis of disaster risk.**
