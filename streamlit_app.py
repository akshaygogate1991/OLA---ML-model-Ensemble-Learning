import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

st.set_page_config(page_title="Ola Driver Churn Dashboard 🚖", layout="wide")

st.title("🚖 Ola Driver Churn Prediction Dashboard")

uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 ML Prediction", "💡 Insights"])

