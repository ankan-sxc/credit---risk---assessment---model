# ============================
# Explainable Credit Risk App
# Production-Safe Streamlit
# ============================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(
    page_title="Explainable Credit Risk System",
    layout="wide"
)

st.title("🏦 Explainable Credit Risk Scoring System")
st.caption("XGBoost + SHAP | Production Deployment")

# ----------------------------
# Safe Artifact Loading
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    return model, preprocessor, feature_names

model, preprocessor, feature_names = load_artifacts()

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("📋 Loan Application Details")

duration = st.sidebar.slider("Loan Duration (months)", 6, 72, 24)
credit_amount = st.sidebar.slider("Credit Amount", 250, 20000, 5000)
age = st.sidebar.slider("Age", 18, 75, 35)
installment_rate = st.sidebar.slider("Installment Rate (%)", 1, 4, 2)
existing_credits = st.sidebar.slider("Existing Credits", 1, 4, 1)

# ----------------------------
# Build Input Row (Schema-Safe)
# ----------------------------
input_df = pd.DataFrame([{
    "duration": duration,
    "credit_amount": credit_amount,
    "age": age,
    "installment_rate": installment_rate,
    "existing_credits": existing_credits
}])

# Fill missing columns safely
for col in preprocessor.feature_names_in_:
    if col not in input_df.columns:
        # use safe placeholder for categorical
        input_df[col] = "unknown"

# Correct column order
input_df = input_df[preprocessor.feature_names_in_]

# ----------------------------
# Prediction & Explanation
# ----------------------------
if st.button("🚀 Predict Credit Risk"):
    try:
        # Preprocess
        X_proc = preprocessor.transform(input_df)

        # Predict
        pd_prob = model.predict_proba(X_proc)[0, 1]

        st.subheader("📊 Prediction Result")
        st.metric("Probability of Default", f"{pd_prob:.2%}")

        if pd_prob > 0.6:
            st.error("❌ High Risk – Loan Rejected")
        elif pd_prob > 0.35:
            st.warning("⚠️ Medium Risk – Manual Review")
        else:
            st.success("✅ Low Risk – Loan Approved")

        # ----------------------------
        # SHAP Explanation (SAFE)
        # ----------------------------
        st.subheader("🔍 Local Explanation (SHAP)")

        explainer = shap.Explainer(model)
        shap_values = explainer(X_proc)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0].values,
                base_values=shap_values[0].base_values,
                data=X_proc[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig)

    except Exception as e:
        st.exception(e)
