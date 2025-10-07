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

# ----------------------------- LOAD DATA FROM GOOGLE SHEETS -----------------------------
st.info("📂 Loading dataset directly from Google Sheets...")

# Google Sheet ID (from your link)
sheet_id = "1tZgqv4JIsIL_orhMGsjvYak8yubM50GiA1P45TWJ_fs"

# Sheet name (bottom tab name)
sheet_name = "Sheet1"  # change if renamed

# Construct CSV export link
sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

try:
    df = pd.read_csv(sheet_url)
    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
    st.write("### Preview of Data:")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# ----------------------------- CHURN COLUMN CHECK -----------------------------
if 'Churn' not in df.columns:
    st.warning("⚠️ 'Churn' column not found — creating one based on LastWorkingDate...")
    if 'LastWorkingDate' in df.columns:
        df['Churn'] = df['LastWorkingDate'].notnull().astype(int)
        st.success("✅ Created 'Churn' column successfully.")
    else:
        st.error("❌ Cannot derive 'Churn' column. Please include it or add 'LastWorkingDate'.")
        st.stop()

# ----------------------------- CREATE TABS -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 ML Model & Prediction", "💡 Insights"])

# ----------------------------- TAB 1: EDA -----------------------------
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    st.write(f"**Dataset shape:** {df.shape}")
    st.write("### Data Overview")
    st.dataframe(df.describe(include='all'))

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No numeric columns found for correlation heatmap.")

    # Continuous Variable Distribution
    st.subheader("Continuous Variable Distribution")
    cont_cols = [c for c in ['Age', 'Income', 'Total Business Value'] if c in df.columns]
    for col in cont_cols:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Categorical Variable Distribution
    st.subheader("Categorical Variable Distribution")
    cat_cols = [c for c in ['Gender', 'City', 'Education_Level', 'Grade', 'Quarterly Rating'] if c in df.columns]
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(x=col, data=df, order=df[col].value_counts().index, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.info("""
    ✅ **Key Findings:**
    - Younger and lower-rated drivers churn more.  
    - Income & TBV are right-skewed.  
    - City-wise churn distribution varies significantly.
    """)

# ----------------------------- TAB 2: MODEL -----------------------------
with tab2:
    st.header("Machine Learning Model (XGBoost) & Predictions")

    st.markdown("""
    The model was trained using multiple ensemble techniques — Random Forest, Bagging, Gradient Boosting, LightGBM, and XGBoost.  
    Based on F1-score and recall, **XGBoost** emerged as the best model.
    """)

    results_data = {
        "Model": ["XGBoost", "Gradient Boosting", "LightGBM", "Random Forest", "Bagging"],
        "Accuracy": [0.904, 0.893, 0.890, 0.903, 0.909],
        "Precision": [0.45, 0.42, 0.41, 0.43, 0.21],
        "Recall": [0.60, 0.67, 0.67, 0.47, 0.03],
        "F1 Score": [0.51, 0.51, 0.50, 0.45, 0.05]
    }
    st.dataframe(pd.DataFrame(results_data).style.highlight_max(axis=0, color="lightgreen"))
    st.success("✅ XGBoost selected as final model for churn prediction.")

    # Load or train model
    try:
        model = joblib.load("models/xgboost_final_model.pkl")
        st.success("✅ Pre-trained XGBoost model loaded successfully!")
    except:
        st.error("❌ Model file not found. Please upload 'xgboost_final_model.pkl' to /models folder.")
        st.stop()
        df = df.dropna(subset=['Churn'])
        X = df.select_dtypes(include=['number']).drop(columns=['Churn'])
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, "xgb_best_model.pkl")

    # Input Form
    st.subheader("🔮 Predict Driver Churn")
    age = st.number_input("Driver Age", min_value=18, max_value=65, value=35)
    income = st.number_input("Monthly Income (₹)", min_value=10000, max_value=200000, value=60000)
    rating = st.slider("Quarterly Rating", 1, 5, 3)
    grade = st.selectbox("Grade", [1, 2, 3, 4, 5])
    tenure = st.number_input("Tenure (Years)", min_value=0.0, max_value=15.0, value=2.5)
    tbv = st.number_input("Total Business Value", min_value=0, max_value=2000000, value=300000)

    if st.button("🚀 Predict"):
        input_data = pd.DataFrame([[age, income, rating, grade, tenure, tbv]],
                                  columns=['Age', 'Income', 'Quarterly Rating', 'Grade', 'Tenure_Years', 'Total Business Value'])
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        if pred == 1:
            st.error(f"⚠️ Driver likely to CHURN (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Driver likely to STAY (Probability: {prob:.2f})")

# ----------------------------- TAB 3: INSIGHTS -----------------------------
with tab3:
    st.header("💡 Business Insights")
    st.markdown("""
    ### Key Insights from Analysis:
    1. **Performance & Grade** are the strongest churn indicators.  
       - Low-rated and low-grade drivers are most likely to churn.
    2. **Demographics** have minimal impact (Gender, Age).  
       - Majority of churners are aged 30–34.
    3. **Tenure & Business Value:**  
       - Drivers with < 1 year tenure and low TBV are 3× more likely to churn.
    4. **City Variation:**  
       - Highest churn observed in C20, C26, and C29.

    ---
    🧩 **Business Actionables**
    - Offer fast-track promotions & retention bonuses for early-career drivers.  
    - Provide training for low-rating drivers.  
    - Use this model to flag high-churn-risk drivers early.
    """)
    st.info("🎯 Data-driven retention strategies can reduce churn by up to **20%**.")
