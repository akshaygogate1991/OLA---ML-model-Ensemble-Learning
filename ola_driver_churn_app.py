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

# ----------------------------- CREATE TABS -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 ML Model & Prediction", "💡 Insights"])


# =====================================================================
# ----------------------------- TAB 1: EDA -----------------------------
# =====================================================================

with tab1:
    st.header("📊 Exploratory Data Analysis (EDA)")
    # ----------------------------- LOAD DATA FROM GOOGLE SHEETS -----------------------------
    st.info("📂 Loading dataset directly from Google Sheets...")

    # Google Sheet ID (from your link)
    sheet_id = "1tZgqv4JIsIL_orhMGsjvYak8yubM50GiA1P45TWJ_fs"

    # Sheet name (bottom tab name)
    sheet_name = "Sheet1"  # change if renamed

    # Construct CSV export link
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    # Try loading data from Google Sheets
    try:
        df = pd.read_csv(sheet_url)
        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
        st.write("### Preview of Data:")
        st.dataframe(df.head(), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

    # ----------------------------- CHURN COLUMN CHECK -----------------------------
    if 'Churn' not in df.columns:
        st.warning("⚠️ 'Churn' column not found — creating one based on LastWorkingDate...")

        if 'LastWorkingDate' in df.columns:
            df['Churn'] = df['LastWorkingDate'].notnull().astype(int)
            st.success("✅ Created new column: **Churn = 1 if LastWorkingDate present, else 0**")
        else:
            st.error("❌ Cannot derive 'Churn' column. Please include it or add 'LastWorkingDate'.")
            st.stop()
    else:
        st.info("✅ 'Churn' column already exists in the dataset.")


    st.markdown("""
    ### 🧠 Problem Statement
    Ola aims to proactively identify drivers who are at risk of **attrition (churn)**.
    The goal is to use **ensemble machine learning models** to predict churn and uncover actionable insights.
    """)

    st.markdown("""
    **Key Business Objectives:**
    - 🎯 Predict whether a driver will churn (Churn = 1) or stay (Churn = 0)
    - 🧩 Use ensemble models (Random Forest, Gradient Boosting, XGBoost, LightGBM)
    - 💡 Identify top churn drivers (Income, Rating, City, Tenure, etc.)
    - 🏆 Help Ola’s HR & Ops teams design retention strategies
    """)

    # ----------------------------- DATA PREVIEW -----------------------------
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Convert dates
    df['MMM-YY'] = pd.to_datetime(df['MMM-YY'], errors='coerce')
    df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], errors='coerce')
    df['LastWorkingDate'] = pd.to_datetime(df['LastWorkingDate'], errors='coerce')

    # Extract year and month
    df1 = df.copy()
    df1['year'] = df1['LastWorkingDate'].dt.year
    df1['month'] = df1['LastWorkingDate'].dt.month

    # ============================= 1️⃣ MONTHLY CHURN =============================
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

    # ============================= 2️⃣ YEARLY CHURN =============================
    st.markdown("### 📆 Yearly Driver Churn Overview")
    df_year = df1['year'].value_counts().reset_index()
    df_year.columns = ['Year', 'Count']
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_year.style.highlight_max(subset=['Count'], color='lightblue'))
    with col2:
        st.bar_chart(df_year.set_index('Year'))
    st.info("📉 Churn is relatively consistent across 2019 and 2020.")

    # ============================= 3️⃣ TENURE & INCOME =============================
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

    # ============================= 4️⃣ AGE DISTRIBUTION =============================
    st.markdown("### 👥 Age Distribution of Drivers Who Left")
    churn_age = df1[df1["Churn"] == 1]['Age'].value_counts().reset_index()
    churn_age.columns = ['Age', 'Count']
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(churn_age.head(10))
    with col2:
        st.bar_chart(churn_age.set_index('Age'))
    st.info("🧓 Most churn occurs in age group **30–34 years** — likely mid-career transitions.")

    # ============================= 5️⃣ DISTRIBUTION INSIGHTS =============================
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

    # ============================= 6️⃣ CITY-WISE ANALYSIS =============================
    st.markdown("### 🏙️ City-wise Churn & Performance Analysis")
    city_churn_rate = df.groupby('City')['Churn'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    city_churn_rate.plot(kind='bar', color='coral', ax=ax)
    ax.set_title("Churn Rate by City")
    ax.set_ylabel("Proportion of Drivers Left")
    st.pyplot(fig)
    st.caption("📊 Top churn in C20, C26, C29 — investigate driver support & operations there.")

    # ============================= 7️⃣ CORRELATION HEATMAP =============================
    st.markdown("### 🔥 Correlation Heatmap (Numeric Features)")
    numeric_df = df.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    st.success("📌 Key Insight: Quarterly Rating & Grade are strongest churn indicators.")

    # ============================= 8️⃣ BIVARIATE ANALYSIS =============================
    st.markdown("---")
    st.header("🔍 Bivariate Analysis — Relationship Between Key Variables")
    st.markdown("""
    This section explores how two variables relate to each other.
    We analyze **age vs income**, **grade vs income**, and **performance vs churn**.
    """)

    # AGE vs INCOME
    st.subheader("📉 Age vs Income Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x='Age', y='Income', alpha=0.6, color='royalblue', edgecolor='white')
    ax.set_title("Age vs Income", fontsize=12)
    st.pyplot(fig, use_container_width=True)
    st.info("""
    🧭 **Insights**
    - Income rises with age until ~40 years, then flattens.
    - Core segment: Ages 28–40, ₹40K–₹100K income.
    - Older drivers (>45) earn less — possible early retirement.
    """)

    # GRADE vs INCOME
    st.subheader("💼 Income Distribution by Grade")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Grade', y='Income', palette='Blues', ax=ax)
    ax.set_title("Income by Grade")
    st.pyplot(fig, use_container_width=True)
    st.info("""
    💡 **Key Observations**
    - Median income increases with grade.
    - Grade 1 ≈ ₹40K → Grade 5 ≈ ₹135K+.
    - Higher grades show wider variance — performance-based pay.
    """)

    # GRADE vs CHURN
    st.subheader("📊 Churn Rate by Grade")
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.crosstab(df['Grade'], df['Churn'], normalize='index').plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax)
    ax.set_title("Churn Rate by Grade")
    ax.legend(title='Churn', labels=['Active (0)', 'Left (1)'])
    st.pyplot(fig, use_container_width=True)
    st.success("""
    📈 **Interpretation**
    - Grade 1–2 show highest churn.
    - Higher grades = lower churn → stability & loyalty.
    """)

    # GENDER vs CHURN
    st.subheader("🚻 Churn by Gender")
    fig, ax = plt.subplots(figsize=(5, 3))
    pd.crosstab(df['Gender'], df['Churn'], normalize='index').plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
    ax.set_title("Churn by Gender")
    ax.legend(title='Churn', labels=['Active (0)', 'Left (1)'])
    st.pyplot(fig, use_container_width=True)
    st.info("👫 Churn proportion nearly identical across genders — no bias observed.")

    # INCOME vs CHURN
    st.subheader("💰 Income by Churn Status")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Churn', y='Income', palette='pastel', ax=ax)
    ax.set_title("Income vs Churn")
    ax.set_xticklabels(['Active', 'Left'])
    st.pyplot(fig, use_container_width=True)
    st.info("💸 Drivers who left tend to earn slightly less; overlap large → income not sole factor.")

    # RATING vs CHURN
    st.subheader("⭐ Quarterly Rating by Churn Status")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Churn', y='Quarterly Rating', palette='Set2', ax=ax)
    ax.set_title("Quarterly Rating vs Churn")
    ax.set_xticklabels(['Active', 'Left'])
    st.pyplot(fig, use_container_width=True)
    st.success("🌟 Low-rated drivers churn significantly more — supports performance hypothesis.")

    # AGE vs CHURN
    st.subheader("🧓 Age vs Churn Status")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Churn', y='Age', ax=ax, palette='cool')
    ax.set_title("Age vs Churn")
    ax.set_xticklabels(['Active', 'Left'])
    st.pyplot(fig, use_container_width=True)
    st.info("👤 Median ages similar across churned/active drivers — weak direct impact.")

    # TBV vs CHURN
    st.subheader("📦 Business Value vs Churn Status")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Churn', y='Total Business Value', ax=ax, palette='Purples')
    ax.set_title("Business Value vs Churn")
    ax.set_xticklabels(['Active', 'Left'])
    st.pyplot(fig, use_container_width=True)
    st.success("💼 Active drivers have higher TBV — retention focus on low TBV segments.")

    # ============================= 
    # 🔥 9. CORRELATION INSIGHTS 
    # ============================= 
    st.markdown("### 🔥 Correlation Analysis Between Key Features")
    numeric_df = df.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig, use_container_width=True)

    st.info("""
    📊 **Summary:**
    - **Quarterly Rating & Grade** show strong negative correlation with churn.
    - **Income & TBV** correlate highly (0.78) — high earners generate more business value.
    - Demographic factors (Gender, Age) have minimal correlation → low modeling priority.
    """)

    st.write("""
    The data strongly suggests that employee performance and seniority are the primary drivers of churn.
    Employees with low quarterly ratings and those in lower grades are significantly more likely to leave the company.
    In contrast, demographic factors like gender and age appear to have little to no impact on an employee's decision to leave.

    **Performance is a Key Predictor of Churn**
    The most significant factor related to employee churn is the Quarterly Rating.
    - **Correlation Heatmap:** Shows a moderate negative correlation of -0.26 between Quarterly Rating and Churn.
    - **Quarterly Rating Boxplot:** Confirms that low-rated drivers churn the most.
    - **Observation:** Drivers who stayed (Churn=0) generated higher business value, supporting the hypothesis that high performers are more loyal.

    **Seniority and Grade Matter**
    - **Churn Rate by Grade:** Lower grades show highest churn; rate drops with higher grades.
    - **Correlation:** Grade and Income are strongly correlated (~0.78), forming a seniority-performance cluster.

    **Demographics Show Little Impact**
    - **Gender:** Nearly identical churn proportion.
    - **Age:** Slightly lower for active drivers; not a strong churn indicator.
    - **Income:** Weak correlation (-0.1) → not a primary churn factor.
    """)

    st.write("📋 Available Columns in DataFrame:")
    st.write(df.columns.tolist())

    # ============================= 
    # 🧹 DATA CLEANING & FEATURE ENGINEERING 
    # ============================= 
    st.markdown("---")
    st.header("🧹 Data Cleaning, Outlier Handling & Feature Engineering")

    # 1️⃣ Duplicate Check
    st.subheader("🧾 Checking for Duplicate Records")
    duplicate_rows = df[df.duplicated()]
    dup_count = duplicate_rows.shape[0]
    st.metric(label="Duplicate Rows Found", value=dup_count)
    if dup_count > 0:
        st.warning(f"⚠️ Found {dup_count} duplicate records. Consider removing them before modeling.")
    else:
        st.success("✅ No duplicate records found.")

    # 2️⃣ Missing Value Analysis
    st.subheader("🔍 Missing Value Check")
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    if not missing_summary.empty:
        st.write("### Columns with Missing Values")
        st.dataframe(missing_summary.to_frame(name="Missing Count").style.background_gradient(cmap="Reds"))
    else:
        st.success("✅ No missing values detected in dataset.")

    # 3️⃣ Missing Value Imputation (KNN)
    st.subheader("🧠 Handling Missing Values with KNN Imputer")
    from sklearn.impute import KNNImputer
    subset_cols = ['Age', 'Gender']
    imputer = KNNImputer(n_neighbors=5)
    df_subset = df[subset_cols]
    df_imputed = pd.DataFrame(imputer.fit_transform(df_subset), columns=subset_cols)
    df['Age'] = df_imputed['Age']
    df['Gender'] = df_imputed['Gender']

    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if missing_summary.empty:
        st.success("✅ Missing values successfully imputed for Age & Gender using KNN.")
    else:
        st.warning("⚠️ Some missing values remain after imputation.")

    # 4️⃣ Outlier Detection & Treatment (IQR)
    st.subheader("📦 Outlier Detection & Capping")
    cols = ['Age', 'Income']
    st.write("### Before Outlier Treatment")
    col1, col2 = st.columns(2)
    for i, col in enumerate(cols):
        with [col1, col2][i]:
            fig, ax = plt.subplots(figsize=(5, 2))
            sns.boxplot(x=df[col], color='orange', ax=ax)
            ax.set_title(f"Before Capping: {col}")
            st.pyplot(fig)

    def cap_outliers_iqr(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
        return df

    for col in cols:
        df = cap_outliers_iqr(df, col)

    st.write("### After Outlier Treatment")
    col1, col2 = st.columns(2)
    for i, col in enumerate(cols):
        with [col1, col2][i]:
            fig, ax = plt.subplots(figsize=(5, 2))
            sns.boxplot(x=df[col], color='green', ax=ax)
            ax.set_title(f"After Capping: {col}")
            st.pyplot(fig)

    st.info("""
    ✅ Outliers were capped using the **Interquartile Range (IQR)** method.
    This helps stabilize model performance and prevents bias from extreme values.
    """)

    # 5️⃣ Skewness Check (Feature Normality)
    st.subheader("📊 Skewness of Continuous Variables")
    continuous_vars = ['Age', 'Income', 'Total Business Value', 'Tenure_Years']
    col1, col2 = st.columns(2)
    for col in continuous_vars:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(df[col].dropna(), kde=True, bins=30, ax=ax)
            ax.set_title(f"{col} | Skewness: {round(df[col].skew(), 2)}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig, use_container_width=True)
        else:
            st.warning(f"⚠️ Column '{col}' not found in dataset. Skipping...")

    st.info("""
    📈 **Feature Skewness Summary:**
    - Age: 0.42 → Mildly right-skewed  
    - Income: 0.62 → Moderately right-skewed  
    - Total Business Value: 6.97 → Highly right-skewed  
    - Tenure_Years: 1.14 → Significantly right-skewed  
    👉 Consider log or power transformations for highly skewed features before training.
    """)

    # 6️⃣ Flag Feature Creation
    st.subheader("🚩 Feature Engineering — Driver Flags")
    st.write("✅ Columns before flag creation:", df.columns.tolist())

    if 'Tenure_Years' not in df.columns or df['Tenure_Years'].isnull().all():
        st.warning("⚠️ 'Tenure_Years' missing or empty — recomputing it now...")
        try:
            df['EndDate'] = df['LastWorkingDate'].fillna(df['MMM-YY'])
            df['EndDate'] = pd.to_datetime(df['EndDate'], errors='coerce')
            df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], errors='coerce')
            df['Tenure_Years'] = (df['EndDate'] - df['Dateofjoining']).dt.days / 365
            df['Tenure_Years'] = df['Tenure_Years'].fillna(0).round(2)
            st.success("✅ Recomputed 'Tenure_Years' column successfully!")
        except Exception as e:
            st.error(f"❌ Failed to recompute 'Tenure_Years': {e}")
            st.stop()
    else:
        st.info("ℹ️ 'Tenure_Years' already available — proceeding to flags.")

    if 'Tenure_Years' not in df.columns:
        st.error("❌ 'Tenure_Years' column still missing after recomputation. Stopping.")
        st.stop()

    try:
        threshold_bv = df['Total Business Value'].quantile(0.90)
        threshold_income = df['Income'].quantile(0.10)
        df['High_Business_Value_Flag'] = (df['Total Business Value'] >= threshold_bv).astype(int)
        df['Low_Income_Flag'] = (df['Income'] <= threshold_income).astype(int)
        df['Recent_Joiner_Flag'] = (df['Tenure_Years'] < 1).astype(int)
        df['Senior_Driver_Flag'] = (df['Age'] > 50).astype(int)
        df['Low_Rating_Flag'] = (df['Quarterly Rating'] <= 2).astype(int)

        st.dataframe(
            df[['Tenure_Years', 'High_Business_Value_Flag', 'Low_Income_Flag',
                'Senior_Driver_Flag', 'Recent_Joiner_Flag', 'Low_Rating_Flag']].head(10),
            use_container_width=True
        )
        st.success("✅ All feature flags created successfully!")

    except KeyError as e:
        st.error(f"❌ KeyError: {e} — one of the required columns is missing.")
        st.write("Columns present right now:", df.columns.tolist())
    except Exception as e:
        st.error(f"❌ Unexpected error while creating flags: {e}")
        st.write("Columns present right now:", df.columns.tolist())

    st.write("✅ Final Columns After Flag Creation:")
    st.write(df.columns.tolist())
    st.write("🧾 Sample of Tenure_Years column:")
    st.write(df[['Tenure_Years']].head())

    # 7️⃣ City & Age Group Analysis
    st.markdown("---")
    st.header("🏙️ City & Age Group Analysis")

    st.subheader("📉 Churn Rate by City")
    city_churn_rate = df.groupby('City')['Churn'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    city_churn_rate.plot(kind='bar', color='tomato', edgecolor='black', ax=ax)
    ax.set_title("Churn Rate by City")
    ax.set_ylabel("Churn Rate")
    st.pyplot(fig, use_container_width=True)

    st.subheader("💼 Total Business Value by City")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='City', y='Total Business Value', data=df, ax=ax)
    plt.xticks(rotation=90)
    ax.set_title("Total Business Value by City")
    st.pyplot(fig, use_container_width=True)

    st.subheader("🚗 Driver Count per City")
    fig, ax = plt.subplots(figsize=(10, 6))
    df['City'].value_counts().plot(kind='barh', color='steelblue', ax=ax)
    ax.set_xlabel("Number of Drivers")
    ax.set_title("Driver Distribution Across Cities")
    st.pyplot(fig, use_container_width=True)

    st.subheader("💰 Average Income by City")
    avg_income_city = df.groupby('City')['Income'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_income_city.plot(kind='bar', color='seagreen', ax=ax)
    ax.set_title("Average Income by City")
    st.pyplot(fig, use_container_width=True)

    # 8️⃣ AGE GROUP BINNING
    st.subheader("👥 Age Group Distribution and Churn")
    bins = [0, 30, 50, df['Age'].max()]
    labels = ['Young', 'Middle-aged', 'Senior']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df['Age_Group'].value_counts().reset_index().rename(columns={'index': 'Age Group', 'Age_Group': 'Count'}))
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        df['Age_Group'].value_counts().plot(kind='bar', color='plum', ax=ax)
        ax.set_title("Driver Age Group Distribution")
        st.pyplot(fig)

    st.subheader("📊 Churn Rate by Age Group")
    age_churn_rate = df.groupby('Age_Group', observed=False)['Churn'].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    age_churn_rate.plot(kind='bar', color='salmon', ax=ax2)
    ax2.set_title("Churn Rate by Age Group")
    st.pyplot(fig2, use_container_width=True)

    st.subheader("💼 Average Business Value by Age Group")
    age_bv = df.groupby('Age_Group')['Total Business Value'].mean()
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    age_bv.plot(kind='bar', color='skyblue', ax=ax3)
    ax3.set_title("Average Business Value by Age Group")
    st.pyplot(fig3, use_container_width=True)

    st.success("""
    👥 **Insights:**
    - Middle-aged drivers dominate (~30–45 years)
    - Young drivers (<30) have highest churn
    - Senior drivers show higher loyalty but lower TBV  
    👉 Retention focus: young & mid-age groups via engagement programs.
    """)



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




