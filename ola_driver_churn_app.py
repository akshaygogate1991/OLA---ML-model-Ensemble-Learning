import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(page_title="Ola Driver Churn Dashboard 🚖", layout="wide")
st.title("🚖 Ola Driver Churn Prediction Dashboard")

# ----------------------------- GOOGLE SHEET DETAILS -----------------------------
sheet_id = "1tZgqv4JIsIL_orhMGsjvYak8yubM50GiA1P45
