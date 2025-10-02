import streamlit as st
import pandas as pd

st.title("Ola Driver Churn Prediction 🚖")

st.write("Upload dataset and check churn insights")

uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
