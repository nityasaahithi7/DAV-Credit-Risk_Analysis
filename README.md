# 🌿 Credit Risk Analysis System

An end-to-end data wrangling, machine learning, and interactive web application for predicting loan default risk.

🚀 **Live Interactive Application:** [https://dav-credit-risk-analysis.streamlit.app](https://dav-credit-risk-analysis.streamlit.app)

---

## 📌 Project Overview

This project analyzes credit risk data through exploratory data analysis, data wrangling, machine learning classification, and a custom interactive Streamlit web interface. 

The objective is to handle real-world dataset inconsistencies (missing values, duplicated records, and unrealistic domain outliers) and compare predictive performance between linear and ensemble classification algorithms.

---

## 📊 Dataset Specifications

- **Total Records:** 32,581
- **Features:** 12 columns
- **Target Variable:** `loan_status` (Binary: `0 = Non-Default`, `1 = Default`)
- **Key Features Included:** Applicant age, income, home ownership, employment length, loan intent, loan amount, interest rate, and historical default record.

---

## 🛠️ Data Wrangling & Preprocessing Pipeline

1. **Duplicate Handling:** Identified and dropped 165 duplicate applicant records.
2. **Missing Value Imputation:** Applied median imputation to `person_emp_length` and `loan_int_rate` to preserve dataset size without introducing mean distortion.
3. **Domain Outlier Filtering:** Removed unrealistic entries (`person_age > 100` and `person_emp_length > 60`).
4. **Categorical Encoding:** One-hot encoded categorical variables (`person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`).
5. **Stratified Split:** Performed an 80/20 train-test split maintaining target class distribution.
6. **Feature Standardization:** Scaled numerical feature distributions using `StandardScaler` for the Logistic Regression model.

---

## 🤖 Machine Learning Models

### 1. Logistic Regression
- **Purpose:** Linear baseline classification model optimized via scaled gradient boundaries.
- **Evaluation:** Precision, Recall, Accuracy, F1 Score, and Confusion Matrix.

### 2. Random Forest Classifier
- **Purpose:** Ensemble decision tree classifier capturing complex non-linear feature interactions.
- **Evaluation:** Precision, Recall, Accuracy, F1 Score, Confusion Matrix, and Feature Importance Analysis.

---

## 🎨 Interactive Dashboard Architecture

Built with **Streamlit** and styled with a custom **Sage Green & Pink Beige** aesthetic:
- **Clean Selection Cards:** Minimalist algorithm selection interface.
- **On-Demand Execution:** Loads, trains, and evaluates models interactively upon user request.
- **Visual Analytics:** Real-time generation of confusion matrices and cleaned dataset previews.

---

## 📁 Repository Structure

```text
DAV-Credit-Risk_Analysis/
│
├── .streamlit/
│   └── config.toml          # Custom theme configuration
├── data/
│   └── credit_risk_dataset.csv
├── notebooks/
│   └── credit_risk_wrangling.ipynb  # Full Colab EDA notebook
├── app.py                   # Streamlit web application
├── models.py                # Model training logic
├── preprocessing.py         # Data cleaning & pipeline functions
├── requirements.txt         # Dependencies
├── README.md
└── .gitignore