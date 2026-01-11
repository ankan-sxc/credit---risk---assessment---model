# Explainable Credit Risk Modelling System

This project implements an end-to-end, explainable credit risk scoring system designed to estimate the probability of loan default using machine learning. The system integrates data preprocessing, model development, explainable AI techniques, cloud deployment, and business intelligence reporting.

---

## Project Overview

Credit risk assessment is a core function of financial institutions. The objective of this project is to predict the **Probability of Default (PD)** for loan applicants while maintaining transparency and interpretability of model decisions.

The project follows a production-style workflow, separating model training and deployment, and emphasizes explainability to align with regulatory and ethical requirements.

---

## Dataset

- **Source:** UCI Machine Learning Repository  
- **Dataset:** Statlog German Credit Dataset (categorical version)  
- **Observations:** 1,000 loan applicants  
- **Target Variable:**  
  - 0 → Non-default  
  - 1 → Default  

The full dataset containing both categorical and numerical features was used. Categorical variables were encoded using One-Hot Encoding as part of a preprocessing pipeline.

---

## Methodology

### Data Preparation
- Imported the raw categorical dataset using official documentation
- Converted credit risk labels into a binary default indicator
- Performed exploratory data analysis to understand feature distributions and relationships
- Applied feature scaling and encoding using a ColumnTransformer pipeline

### Model Development
- Trained an **XGBoost classifier** to capture non-linear relationships in credit data
- Performed hyperparameter tuning using cross-validation with ROC-AUC as the evaluation metric
- Evaluated model performance using ROC-AUC, precision, recall, and confusion matrix

### Explainability
- Integrated **SHAP (SHapley Additive Explanations)** to provide:
  - Global feature importance
  - Applicant-level local explanations
- Ensured transparency and auditability of model predictions

---

## Deployment

A Streamlit-based web application was developed for real-time credit risk assessment.

**Key Features:**
- Real-time Probability of Default prediction
- Risk categorization (Low / Medium / High)
- Applicant-level SHAP explanations
- Cloud deployment for public access

**Live Application:**  
👉 _[PASTE YOUR STREAMLIT CLOUD LINK HERE]_

---

## Business Intelligence Dashboard

A Power BI dashboard was created to support portfolio-level risk monitoring, including:
- Default rate analysis
- Approval and rejection trends
- Risk segmentation across demographic and financial attributes

---

## Project Structure
credit-risk-project/
│
├── data/
│   ├── raw.csv
│   ├── processed.csv
│
├── notebooks/
│   └── credit_risk.ipynb
│
├── app/
│   ├── streamlit_app.py
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── feature_names.pkl
│   └── explainer.pkl
│
├── powerbi/
│   └── credit_risk_dashboard.pbix
│
├── requirements.txt
└── README.md

---

## Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- SHAP  
- Streamlit  
- Power BI  

---

## Author

**Name:** ANKAN MAITI  
**Degree:** M.Sc Data Science, St.Xavier's College(Autonomus),Kolkata

---

## Future Enhancements

- Expected Loss modeling (PD × LGD × EAD)
- Fairness and bias evaluation
- Model drift detection
- API-based deployment
