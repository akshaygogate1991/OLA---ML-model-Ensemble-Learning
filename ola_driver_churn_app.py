import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(page_title="Ola Driver Churn Dashboard 🚖", layout="wide")
st.title("🚖 Ola Driver Churn Prediction Dashboard")

# ----------------------------- GOOGLE SHEETS LINK -----------------------------
sheet_id = "1tZgqv4JIsIL_orhMGsjvYak8yubM50GiA1P45TWJ_fs"
sheet_name = "Sheet1"
sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# ----------------------------- CREATE TABS -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 ML Model & Prediction", "💡 Insights"])

# =====================================================================
# ----------------------------- TAB 1: EDA -----------------------------
# =====================================================================
# ----------------------------- TAB 1: EDA -----------------------------
with tab1:
    st.header("📊 Exploratory Data Analysis (EDA)")

    st.markdown("""
    ### 🧠 Problem Statement
    Ola aims to proactively identify drivers who are at risk of **attrition (churn)**. 
    The goal is to use **ensemble machine learning models** to predict churn and uncover actionable insights.
    """)

    st.markdown("""
    **Key Business Objectives:**
    - 🎯 Predict whether a driver will churn (Churn = 1) or stay (Churn = 0)  
    - 🧩 Use ensemble models (Random Forest, Gradient Boosting, XGBoost, LightGBM) for robust predictions  
    - 💡 Identify top churn drivers (Income, Rating, City, Tenure, etc.)  
    - 🏆 Help Ola’s HR & Ops teams design retention strategies  
    """)

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # -----------------------------
    # Data Preprocessing
    # -----------------------------
    df['MMM-YY'] = pd.to_datetime(df['MMM-YY'], errors='coerce')
    df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], errors='coerce')
    df['LastWorkingDate'] = pd.to_datetime(df['LastWorkingDate'], errors='coerce')

    df['Churn'] = df['LastWorkingDate'].notnull().astype(int)
    st.success("✅ Created new column: **Churn = 1 if LastWorkingDate present, else 0**")

    df1 = df.copy()
    df1['year'] = df1['LastWorkingDate'].dt.year
    df1['month'] = df1['LastWorkingDate'].dt.month

    # -----------------------------
    # 1️⃣ Monthly Churn Trend
    # -----------------------------
    st.markdown("### 📅 Monthly Driver Churn Trend")
    df_month = df1['month'].value_counts().reset_index()
    df_month.columns = ['Month', 'Count']

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_month.style.highlight_max(subset=['Count'], color='lightgreen'), use_container_width=True)
    with col2:
        st.bar_chart(df_month.set_index('Month'))

    top_months = df_month.sort_values(by='Count', ascending=False).head(3)['Month'].tolist()
    st.info(f"🚗 Most drivers left during **months {top_months}** — focus retention strategies there.")

    # -----------------------------
    # 2️⃣ Yearly Churn Overview
    # -----------------------------
    st.markdown("### 📆 Yearly Driver Churn Overview")
    df_year = df1['year'].value_counts().reset_index()
    df_year.columns = ['Year', 'Count']

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_year.style.highlight_max(subset=['Count'], color='lightblue'))
    with col2:
        st.bar_chart(df_year.set_index('Year'))

    st.info("📉 Churn is relatively consistent across 2019 and 2020.")

    # -----------------------------
    # 3️⃣ Tenure vs Total Income
    # -----------------------------
    st.markdown("### 💰 Tenure vs Total Income Analysis")

    df1 = df.copy()
    df1['EndDate'] = df1['LastWorkingDate'].fillna(df1['MMM-YY'])
    df1['Tenure_Years'] = (df1['EndDate'] - df1['Dateofjoining']).dt.days / 365
    df['Tenure_Years'] = df1['Tenure_Years'].values

    st.success(f"✅ 'Tenure_Years' column added successfully! New Shape: {df.shape}")
    st.write("📋 Columns in DataFrame:")
    st.write(df.columns.tolist())

    driver_summary = df1.groupby('Driver_ID').agg({'Tenure_Years': 'max', 'Income': 'sum'}).reset_index()
    driver_summary['Tenure_Years'] = driver_summary['Tenure_Years'].round(2)
    st.dataframe(driver_summary.head(10).style.background_gradient(cmap='Greens'))
    st.caption("🧾 Top drivers who left had total earnings between ₹35–45 lakhs.")

    # -----------------------------
    # 4️⃣ Age Distribution
    # -----------------------------
    st.markdown("### 👥 Age Distribution of Drivers Who Left")
    churn_age = df1[df1["Churn"] == 1]['Age'].value_counts().reset_index()
    churn_age.columns = ['Age', 'Count']

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(churn_age.head(10))
    with col2:
        st.bar_chart(churn_age.set_index('Age'))

    st.info("🧓 Most churn occurs in age group **30–34 years** — likely mid-career transitions.")

    # -----------------------------
    # 5️⃣ Distribution Insights
    # -----------------------------
    st.markdown("### 📈 Key Continuous Variable Distributions")
    sns.set(style='whitegrid')
    cont_cols = ['Age', 'Income', 'Total Business Value']

    cols = st.columns(3)
    for i, col in enumerate(cont_cols):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color='skyblue')
            ax.set_title(f'{col} Distribution')
            st.pyplot(fig)

    st.info("""
    🔹 **Age:** Mostly 28–38 years → target early retention  
    🔹 **Income:** Concentrated ₹35K–₹75K → mid-tier earners dominate  
    🔹 **TBV:** Highly skewed → few top performers generate bulk of revenue
    """)

    # -----------------------------
    # 6️⃣ City-Wise Analysis
    # -----------------------------
    st.markdown("### 🏙️ City-wise Churn & Performance Analysis")
    city_churn_rate = df.groupby('City')['Churn'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    city_churn_rate.plot(kind='bar', color='coral', ax=ax)
    ax.set_title("Churn Rate by City")
    ax.set_ylabel("Proportion of Drivers Left")
    st.pyplot(fig)

    st.caption("📊 Top churn in C20, C26, C29 — investigate driver support & operations there.")

    # -----------------------------
    # 7️⃣ Correlation Heatmap
    # -----------------------------
    st.markdown("### 🔥 Correlation Heatmap (Numeric Features)")
    numeric_df = df.select_dtypes(include=['number'])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    st.success("📌 Key Insight: Quarterly Rating & Grade are strongest churn indicators.")


# =====================================================================
# ----------------------------- TAB 2: MODEL -----------------------------
# =====================================================================
with tab2:
    st.header("🤖 Machine Learning Model (XGBoost) & Predictions")

    st.markdown("""
    The model was trained using multiple ensemble methods — Random Forest, Bagging, Gradient Boosting, LightGBM, and XGBoost.  
    Based on **F1-score** and **Recall**, XGBoost was selected as the final model.
    """)

    # Model performance summary
    results_data = {
        "Model": ["XGBoost", "Gradient Boosting", "LightGBM", "Random Forest", "Bagging"],
        "Accuracy": [0.904, 0.893, 0.890, 0.903, 0.909],
        "Precision": [0.45, 0.42, 0.41, 0.43, 0.21],
        "Recall": [0.60, 0.67, 0.67, 0.47, 0.03],
        "F1 Score": [0.51, 0.51, 0.50, 0.45, 0.05]
    }
    st.dataframe(pd.DataFrame(results_data).style.highlight_max(axis=0, color="lightgreen"))
    st.success("✅ XGBoost selected as final churn prediction model.")

    # Load model and encoders
    try:
        model = joblib.load("models/xgboost_final_model.pkl")
        le = joblib.load("models/label_encoder.pkl")
        target_enc = joblib.load("models/target_encoder.pkl")
        st.success("✅ Model and encoders loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model/encoders: {e}")
        st.stop()

    # ----------------------------- Prediction Form -----------------------------
    st.subheader("🔮 Predict Driver Churn")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Driver Age", 18, 65, 35)
        gender = st.selectbox("Gender (0 = Male, 1 = Female)", [0, 1])
        city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Pune"])
        grade = st.selectbox("Grade", [1, 2, 3, 4, 5])
    with col2:
        income = st.number_input("Monthly Income (₹)", 10000, 200000, 60000)
        rating = st.slider("Quarterly Rating", 1, 5, 3)
        tenure = st.number_input("Tenure (Years)", 0.0, 15.0, 2.5)
        tbv = st.number_input("Total Business Value", 0, 2000000, 300000)

    with st.expander("⚙️ Advanced Inputs (Optional)"):
        performance = st.slider("Driver Performance Score", 0, 10, 5)
        city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

    # ----------------------------- Predict -----------------------------
    if st.button("🚀 Predict"):
        try:
            df_input = pd.DataFrame([{
                'Age': age, 'Gender': gender, 'City': city, 'Income': income,
                'Quarterly Rating': rating, 'Grade': grade,
                'Tenure_Years': tenure, 'Total Business Value': tbv,
                'Performance_Score': performance, 'City_Tier': city_tier
            }])

            # Feature engineering
            df_input['High_Business_Value_Flag'] = (df_input['Total Business Value'] >= 0.9 * df_input['Total Business Value'].max()).astype(int)
            df_input['Low_Income_Flag'] = (df_input['Income'] <= 40000).astype(int)
            df_input['Senior_Driver_Flag'] = (df_input['Age'] > 50).astype(int)
            df_input['Recent_Joiner_Flag'] = (df_input['Tenure_Years'] < 1).astype(int)
            df_input['Low_Rating_Flag'] = (df_input['Quarterly Rating'] <= 2).astype(int)

            bins = [0, 30, 50, 100]
            labels = ['Young', 'Middle-aged', 'Senior']
            df_input['Age_Group'] = pd.cut(df_input['Age'], bins=bins, labels=labels, include_lowest=True)

            df_input['City'] = target_enc.transform(df_input['City'])
            df_input['Age_Group'] = le.transform(df_input['Age_Group'])

            if hasattr(model, 'feature_names_in_'):
                df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

            pred = model.predict(df_input)[0]
            prob = model.predict_proba(df_input)[0][1]

            st.subheader("🔍 Prediction Result")
            if pred == 1:
                st.error(f"⚠️ Driver likely to CHURN (Probability: {prob:.2f})")
            else:
                st.success(f"✅ Driver likely to STAY (Probability: {prob:.2f})")

            st.markdown("### 🔢 Feature Values Used:")
            st.dataframe(df_input)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# =====================================================================
# ----------------------------- TAB 3: INSIGHTS -----------------------------
# =====================================================================
with tab3:
    st.markdown("## 🔍 Business Insights & Recommendations")

    st.markdown("""
    ### ⚖️ Trade-Off Analysis

    **a. Recruiting More Educated Drivers**

    **Pros:**
    - May lead to better professionalism, communication, and adherence to protocols.  
    - Educated drivers may handle tech platforms more efficiently (app usage, navigation).

    **Cons:**
    - Higher salaries or incentives may be required.  
    - Limited pool in certain Tier 2/3 cities.  
    - Education level alone may not guarantee better customer experience without soft-skill training.

    **Conclusion:**  
    Recruiting highly educated drivers may not justify the ROI unless it’s paired with **performance-linked incentives** and **training**.

    ---

    **b. Investing in Driver Training vs. Customer Satisfaction**

    **Benefits of Training:**
    - Standardizes driver behavior and improves ride quality.  
    - Enhances quarterly ratings — which correlate positively with total business value.

    **Trade-Off:**
    - Training costs (venue, materials, lost driving hours) can be significant.  
    - Not all trained drivers may apply the learning effectively.

    **Conclusion:**  
    A **targeted training program** (for low-performing drivers or specific cities) offers better ROI than company-wide training.

    ---

    ### 💡 Recommendations

    **a. Specific Strategies Derived from Model Insights**

    **🎯 Targeted Training Programs**
    - Focus on drivers with **Quarterly Rating < 3** and **low business value**.  
    - Prioritize **cities with low average ratings** but **high ride volume** for maximum impact.

    **👥 Improved Recruitment Processes**
    - Use model insights (e.g., **Age 25–35**, **Grade A**, **high prior ratings**) for screening.  
    - Recruit pilot batches from cities showing strong historical driver performance.

    **💰 Incentive Schemes**
    - Reward **top 10% drivers per city** using a combined metric (Quarterly Rating + Total Business Value).  
    - Offer early ride access, bonuses, or priority support for consistent top performers.

    ---

    **b. Evidence from Analysis**

    **🌆 City-Based Growth**
    - If a city shows the most improvement in driver ratings over the year, increase investment in **recruitment and training** there.

    **👨‍💼 Demographic Targeting**
    - Example: **Age group 30–40** may yield the **highest total business value per ride**.  
    - Use this for **targeted ads** or **partnerships** (e.g., with driving schools).

    ---

    ### 🔁 Feedback Loop for Continuous Improvement

    **a. Periodic Review Process**
    - Conduct **quarterly model validation**:
        - Re-evaluate feature importance.  
        - Retrain models with updated data (ratings, churn, new rides).  
        - Track KPIs using dashboards (Power BI / Tableau).

    **b. Collecting Ongoing Feedback**

    **Driver Feedback:**
    - Quarterly **anonymous surveys** on satisfaction, app usability, and customer interaction.

    **Customer Feedback:**
    - Post-ride **surveys** (ratings + open text).  
    - Apply **NLP analysis** to detect common themes (e.g., “late pickup”, “rude driver”).

    **Model Adjustments:**
    - Integrate new variables like **real-time location accuracy** or **ride cancellation rates**.

    ---

    ### 🧭 Final Note
    Balancing **business cost** and **customer satisfaction** requires a **feedback-driven ecosystem** — one that continuously learns from:
    - Quantitative model results, and  
    - Qualitative human insights.
    """)

    st.info("📈 These insights can guide data-backed policy design, targeted investments, and performance-linked incentives.")




