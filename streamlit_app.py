import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb

st.set_page_config(page_title="Ola Driver Churn Dashboard 🚖", layout="wide")

# Title
st.title("🚖 Ola Driver Churn Prediction Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    st.warning("⚠️ Please upload the Ola driver dataset CSV to continue.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 ML Prediction", "💡 Insights"])

# ---------------- Tab 1: EDA ----------------

with tab1:
    st.header("Exploratory Data Analysis 📊")
    st.write("### Dataset Preview")
    st.write(df.head(10))
    st.write("""
    "Problem Statement:" Ola, a leading ride-sharing platform, aims to proactively identify drivers who are at risk of attrition (churn). Driver churn
    impacts operational efficiency, customer satisfaction, and recruitment costs. Given a historical dataset containing driver demographics,
    performance metrics, and employment history, the objective is to build a predictive model using ensemble learning techniques to forecast
    whether a driver is likely to churn.
    This model should be able to generalize well on unseen data and provide interpretable, actionable insights that can support Ola’s driver
    retention strategy.

    "Key Objectives:"

    "Predictive Modeling:"
    Classify whether a driver will churn (Churn = 1) or stay (Churn = 0) using historical features.
    Ensemble Learning Focus:
    Apply ensemble methods (e.g., Random Forest, Gradient Boosting, XGBoost, etc.) to improve model accuracy and robustness.
    Business Impact:
    Enable early intervention strategies to retain high-risk drivers.
    Reduce churn rate and associated costs.
    Model Interpretability:
    Identify key drivers of churn (e.g., income, rating, city, age, etc.).
    Support HR and operational teams with explainable metrics.""")
    



