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
    st.write(df.info())
    
    df['MMM-YY'] = pd.to_datetime(df['MMM-YY'], errors='coerce')
    df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], errors='coerce')
    df['LastWorkingDate'] = pd.to_datetime(df['LastWorkingDate'], errors='coerce')

    df['Churn'] = df['LastWorkingDate'].notnull().astype(int)
    st.write("Create Churn column: 1 if LastWorkingDate is present, else 0 ")

    df1=df
    df1['year'] = df1['LastWorkingDate'].dt.year
    df1['month'] = df1['LastWorkingDate'].dt.month
    st.write(df1['month'].value_counts())
    st.write("Most of the drive leaves the company in month July September and November")

    st.write((df1['year'].value_counts()))
    st.write("Approximately drive churn rate is same for year 2019 and 2020")

    # Use LastWorkingDate if available, else use last MMM-YY record for each Driver_ID
    df1['EndDate'] = df['LastWorkingDate']
    df1['EndDate'] = df['EndDate'].fillna(df['MMM-YY'])
    
    # Calculate working duration in years
    df1['Tenure_Years'] = (df['EndDate'] - df['Dateofjoining']).dt.days / 365
    # Group by Driver_ID to get total income and final tenure
    driver_summary = df1.groupby('Driver_ID').agg({
    'Tenure_Years': 'max', # Max tenure value per driver
    'Income': 'sum' # Total income over all months
    }).reset_index()
    
    # Round tenure for better readability    
    driver_summary['Tenure_Years'] = driver_summary['Tenure_Years'].round(2).sort_values(ascending=False)
    
    # Sort by total income in descending order
    driver_summary = driver_summary.sort_values(by='Income', ascending=False)
    st.write(driver_summary.head())
    st.write("drives which left the company as earn maximum upto 45lakhs to 35lakhs")
    
    # check age of drivers leaves the company
    st.write(df1[df1["Churn"]==1]['Age'].value_counts())
    st.write("Drivers in the age of 30 to 34 left the company most")

    #Graphical reprsentation for continous and categorical varibales
    # Set style
    sns.set(style='whitegrid')
    # Continuous columns
    continuous_cols = ['Age','Income','Total Business Value']
    # Plot distributions for continuous variables
    for col in continuous_cols:
        plt.figure(figsize=(8,4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of{col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        st.write(plt.show())
    # Categorical columns
    categorical_cols = ['Gender','City','Education_Level','Joining Designation','Grade','Quarterly Rating']
    # Plot countplots for categorical variables
    for col in categorical_cols:
        plt.figure(figsize=(8,4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Countplot of{col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.write(plt.show())
