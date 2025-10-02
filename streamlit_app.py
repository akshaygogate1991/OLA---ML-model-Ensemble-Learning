pip install matplotlib seaborn
import streamlit as st
import pandas as pd

st.title("Ola Driver Churn Prediction 🚖")

st.write("Upload dataset and check churn insights")

uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib   # if you saved models
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

st.set_page_config(page_title="Ola Driver Churn Dashboard 🚖", layout="wide")

# Title
st.title("🚖 Ola Driver Churn Prediction Dashboard")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 ML Prediction", "💡 Insights"])

