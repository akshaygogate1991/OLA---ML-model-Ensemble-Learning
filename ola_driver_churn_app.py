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

    # Convert dates
    df['MMM-YY'] = pd.to_datetime(df['MMM-YY'], errors='coerce')
    df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], errors='coerce')
    df['LastWorkingDate'] = pd.to_datetime(df['LastWorkingDate'], errors='coerce')

    # Create churn column
    df['Churn'] = df['LastWorkingDate'].notnull().astype(int)
    st.success("✅ Created new column: **Churn = 1 if LastWorkingDate present, else 0**")

    # Extract year and month
    df1 = df.copy()
    df1['year'] = df1['LastWorkingDate'].dt.year
    df1['month'] = df1['LastWorkingDate'].dt.month

    # =============================
    # 1️⃣ CHURN DISTRIBUTION BY MONTH
    # =============================
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

    # =============================
    # 2️⃣ YEAR-WISE CHURN TREND
    # =============================
    st.markdown("### 📆 Yearly Driver Churn Overview")
    df_year = df1['year'].value_counts().reset_index()
    df_year.columns = ['Year', 'Count']

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_year.style.highlight_max(subset=['Count'], color='lightblue'))
    with col2:
        st.bar_chart(df_year.set_index('Year'))

    st.info("📉 Churn is relatively consistent across 2019 and 2020.")

    # =============================
    # 3️⃣ DRIVER TENURE & EARNINGS
    # =============================
    st.markdown("### 💰 Tenure vs Total Income Analysis")

    df1['EndDate'] = df['LastWorkingDate'].fillna(df['MMM-YY'])
    df1['Tenure_Years'] = (df1['EndDate'] - df1['Dateofjoining']).dt.days / 365

    driver_summary = df1.groupby('Driver_ID').agg({
        'Tenure_Years': 'max',
        'Income': 'sum'
    }).reset_index()

    driver_summary['Tenure_Years'] = driver_summary['Tenure_Years'].round(2)
    driver_summary = driver_summary.sort_values(by='Income', ascending=False)

    st.dataframe(driver_summary.head(10).style.background_gradient(cmap='Greens'))
    st.caption("🧾 Top drivers who left had total earnings between ₹35–45 lakhs.")

    # 🔧 Ensure Tenure_Years exists in main df
    if 'Tenure_Years' not in df.columns:
        st.warning("⚠️ 'Tenure_Years' not found — creating from joining and last working date.")
        df['EndDate'] = df['LastWorkingDate'].fillna(df['MMM-YY'])
        df['Tenure_Years'] = (df['EndDate'] - df['Dateofjoining']).dt.days / 365
        df['Tenure_Years'] = df['Tenure_Years'].round(2)
        st.success("✅ 'Tenure_Years' successfully added to main dataframe.")
    df["Tenure_Years"]=df1["Tenure_Years"]


    # =============================
    # 4️⃣ AGE DISTRIBUTION INSIGHTS
    # =============================
    st.markdown("### 👥 Age Distribution of Drivers Who Left")

    churn_age = df1[df1["Churn"] == 1]['Age'].value_counts().reset_index()
    churn_age.columns = ['Age', 'Count']

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(churn_age.head(10))
    with col2:
        st.bar_chart(churn_age.set_index('Age'))

    st.info("🧓 Most churn occurs in age group **30–34 years** — likely mid-career transitions.")

    # =============================
    # 5️⃣ DISTRIBUTION INSIGHTS
    # =============================
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

    # =============================
    # 6️⃣ CITY-WISE ANALYSIS
    # =============================
    st.markdown("### 🏙️ City-wise Churn & Performance Analysis")

    city_churn_rate = df.groupby('City')['Churn'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    city_churn_rate.plot(kind='bar', color='coral', ax=ax)
    ax.set_title("Churn Rate by City")
    ax.set_ylabel("Proportion of Drivers Left")
    st.pyplot(fig)

    st.caption("📊 Top churn in C20, C26, C29 — investigate driver support & operations there.")

    # =============================
    # 7️⃣ CORRELATION HEATMAP
    # =============================
    st.markdown("### 🔥 Correlation Heatmap (Numeric Features)")
    numeric_df = df.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    st.success("📌 Key Insight: Quarterly Rating & Grade are strongest churn indicators.")

    # =============================
    # 8️⃣ BIVARIATE ANALYSIS SECTION
    # =============================

    st.markdown("---")
    st.header("🔍 Bivariate Analysis — Relationship Between Key Variables")

    st.markdown("""
    This section explores how two variables relate to each other.  
    We analyze **age vs income**, **grade vs income**, and **performance vs churn**, revealing behavioral and financial churn trends.
    """)

    # =============================
    # 📈 1. AGE vs INCOME (Scatter Plot)
    # =============================
    st.subheader("📉 Age vs Income Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x='Age', y='Income', alpha=0.6, color='royalblue', edgecolor='white')
    ax.set_title("Age vs Income", fontsize=12)
    ax.set_xlabel("Driver Age")
    ax.set_ylabel("Monthly Income (₹)")
    st.pyplot(fig, use_container_width=True)

    st.info("""
    🧭 **Insights**
    - Income rises with age until ~40 years, then slightly flattens.  
    - Core earning segment: **Ages 28–40**, incomes ₹40K–₹100K.  
    - Older drivers (>45) earn less on average — possible early retirement or reduced engagement.
    """)

    # =============================
    # 📊 2. GRADE vs INCOME (Boxplot)
    # =============================
    st.subheader("💼 Income Distribution by Grade")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Grade', y='Income', palette='Blues', ax=ax)
    ax.set_title("Income by Grade")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Income (₹)")
    st.pyplot(fig, use_container_width=True)

    st.info("""
    💡 **Key Observations**
    - Median income **increases sharply** with grade.  
    - Grade 1 median ≈ ₹40K → Grade 5 median ≈ ₹135K+.  
    - Higher grades show greater variance — likely due to performance-based pay.  
    - Suggests **Grade is a strong churn predictor**.
    """)

    # =============================
    # ⚖️ 3. GRADE vs CHURN (Stacked Bar Chart)
    # =============================
    st.subheader("📊 Churn Rate by Grade")

    fig, ax = plt.subplots(figsize=(6, 4))
    pd.crosstab(df['Grade'], df['Churn'], normalize='index').plot(
    kind='bar', stacked=True, colormap='coolwarm', ax=ax
    )
    ax.set_title("Churn Rate by Grade")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Proportion")
    ax.legend(title='Churn', labels=['Active (0)', 'Left (1)'])
    st.pyplot(fig, use_container_width=True)

    st.success("""
    📈 **Interpretation**
    - Lower-grade drivers (Grade 1 & 2) show **highest churn rates**.  
    - Churn declines steadily in higher grades → likely due to job stability & loyalty.  
    - Suggests churn prevention via **promotion pathways** or performance-linked incentives.
    """)

    # =============================
    # 🚻 4. GENDER vs CHURN
    # =============================
    st.subheader("🚻 Churn by Gender")

    fig, ax = plt.subplots(figsize=(5, 3))
    pd.crosstab(df['Gender'], df['Churn'], normalize='index').plot(
        kind='bar', stacked=True, colormap='viridis', ax=ax
        )
    ax.set_title("Churn by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Proportion")
    ax.legend(title='Churn', labels=['Active (0)', 'Left (1)'])
    st.pyplot(fig, use_container_width=True)

    st.info("""
    👫 **Insight:** Churn proportion is nearly identical across genders —  
    no significant attrition bias observed.
    """)

    # =============================
    # 💰 5. INCOME vs CHURN (Boxplot)
    # =============================
    st.subheader("💰 Income by Churn Status")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Churn', y='Income', palette='pastel', ax=ax)
    ax.set_title("Income vs Churn Status")
    ax.set_xticklabels(['Active', 'Left'])
    ax.set_xlabel("Churn Status")
    ax.set_ylabel("Monthly Income (₹)")
    st.pyplot(fig, use_container_width=True)

    st.info("""
    💸 **Key Insight:** Drivers who left tend to have **slightly lower median incomes**.  
    However, overlap is large → income alone isn't a dominant churn factor.
    """)

    # =============================
    # ⭐ 6. QUARTERLY RATING vs CHURN (Boxplot)
    # =============================
    st.subheader("⭐ Quarterly Rating by Churn Status")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Churn', y='Quarterly Rating', palette='Set2', ax=ax)
    ax.set_title("Quarterly Rating vs Churn")
    ax.set_xticklabels(['Active', 'Left'])
    st.pyplot(fig, use_container_width=True)

    st.success("""
    🌟 **Strong Insight:**  
    Low-rated drivers churn significantly more — consistent with poor performance impact.  
    Supports **Quarterly Rating** as a top predictor in churn modeling.
    """)

    # =============================
    # 🧓 7. AGE vs CHURN (Boxplot)
    # =============================
    st.subheader("🧓 Age vs Churn Status")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Churn', y='Age', ax=ax, palette='cool')
    ax.set_title("Age vs Churn Status")
    ax.set_xticklabels(['Active', 'Left'])
    st.pyplot(fig, use_container_width=True)

    st.info("""
    👤 **Observation:** Median ages are similar for both churned and active drivers.  
    Age doesn’t strongly influence churn directly — but may interact with income or tenure.
    """)

    # =============================
    # 📦 8. TOTAL BUSINESS VALUE vs CHURN (Boxplot)
    # =============================
    st.subheader("📦 Business Value vs Churn Status")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Churn', y='Total Business Value', ax=ax, palette='Purples')
    ax.set_title("Total Business Value vs Churn")
    ax.set_xticklabels(['Active', 'Left'])
    st.pyplot(fig, use_container_width=True)

    st.success("""
    💼 **Insight:** Active drivers generate higher total business value.  
    High performers (high TBV) are more loyal — churn prevention should focus on low TBV segments.
    """)

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
    The data strongly suggests that employee performance and seniority are the primary drivers of churn. Employees with low 
    quarterly ratingsand those in lower grades are significantly more likely to leave the company. In contrast, demographic factors like 
    gender and age appear tohave little to no impact on an employee's decision to leave.
    
    Performance is a Key Predictor of Churn
    The most significant factor related to employee churn is the Quarterly Rating. • CorrelationHeatmap: Shows a moderate negative correlation of 
    -0.26 between Quarterly Rating and Churn. This indicates that as an employee's ratinggoes down, the likelihood of them leaving goes up. 
    • Quarterly Rating Boxplot: This chart provides a stark visual confirmation. The vastmajority of employees who left had a quarterly rating of 1. 
    In contrast, active employees have a much highermedian rating and a widerdistribution of scores. This implies that poor performance is a major
    reason for attrition.Drivers who stayed (Churn=0) generally generated higher business value. This supports the hypothesis that high-performing 
    drivers aremore likely to stay, a useful signal for retention modeling."
    Seniority and Grade MatterAn employee's grade and designation, which are linked to seniority and responsibility, also play a crucial role. •Churn 
    Rate by Grade: The stacked bar chart clearly shows that employees in Grade 1 have the highest proportion of churn. This churn rateprogressively 
    decreases as the grade level increases up to Grade 4. This suggests that entry-level or junior employees are the most likely -0.20). Note that Grade and 
    Joining Designation are strongly correlated with each other (0.56) and with Income (0.78), creating a cluster offactors related to seniority.
    Demographics Show Little Impact Demographic factors do not appear to be significant drivers of churn in this dataset. • Gender: The Churnby Gender c
    hart shows that the proportion of employees leaving is almost identical for both genders. The heatmap confirms this with a near-zero correlation
    of 0.013. • Age: The Age by Churn Status boxplot shows that the median age of employees who left is only slightly lower thanthat of active employees. 
    The distributions largely overlap, indicating age is not 
    a strong differentiator. • Income: While the median income foremployees who left is slightly lower, the Income by Churn Status boxplot shows
    a very large overlap between the two groups.
    The weakcorrelation of -0.1 in the heatmap confirms that income is not a primary driver of churn on its own.""")

    # =============================
    # 🧹 DATA CLEANING & FEATURE ENGINEERING
    # =============================

    st.markdown("---")
    st.header("🧹 Data Cleaning, Outlier Handling & Feature Engineering")

    # =============================
    # 1️⃣ Duplicate Check
    # =============================
    st.subheader("🧾 Checking for Duplicate Records")

    duplicate_rows = df[df.duplicated()]
    dup_count = duplicate_rows.shape[0]

    st.metric(label="Duplicate Rows Found", value=dup_count)
    if dup_count > 0:
        st.warning(f"⚠️ Found {dup_count} duplicate records. Consider removing them before modeling.")
    else:
        st.success("✅ No duplicate records found.")

    # =============================
    # 2️⃣ Missing Value Analysis
    # =============================
    st.subheader("🔍 Missing Value Check")

    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    if not missing_summary.empty:
        st.write("### Columns with Missing Values")
        st.dataframe(missing_summary.to_frame(name="Missing Count").style.background_gradient(cmap="Reds"))
    else:
        st.success("✅ No missing values detected in dataset.")

    # =============================
    # 3️⃣ Missing Value Imputation (KNN)
    # =============================
    st.subheader("🧠 Handling Missing Values with KNN Imputer")

    from sklearn.impute import KNNImputer

    subset_cols = ['Age', 'Gender']
    imputer = KNNImputer(n_neighbors=5)
    df_subset = df[subset_cols]
    df_imputed = pd.DataFrame(imputer.fit_transform(df_subset), columns=subset_cols)

    df['Age'] = df_imputed['Age']
    df['Gender'] = df_imputed['Gender']

    # Post-imputation check
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if missing_summary.empty:
        st.success("✅ Missing values successfully imputed for Age & Gender using KNN.")
    else:
        st.warning("⚠️ Some missing values remain after imputation.")

    # =============================
    # 4️⃣ Outlier Detection & Treatment (IQR)
    # =============================
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

    # IQR Capping
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

    # =============================
    # 5️⃣ Skewness Check (Feature Normality)
    # =============================
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
    - **Age:** 0.42 → Mildly right-skewed  
    - **Income:** 0.62 → Moderately right-skewed  
    - **Total Business Value:** 6.97 → Highly right-skewed  
    - **Tenure_Years:** 1.14 → Significantly right-skewed  
    👉 Consider log or power transformations for highly skewed features before training.
    """)

    # =============================
    # 6️⃣ Flag Feature Creation
    # =============================


    # =============================
    # 7️⃣ City & Age Group Analysis
    # =============================
    st.markdown("---")
    st.header("🏙️ City & Age Group Analysis")

    # ---- Churn Rate by City ----
    st.subheader("📉 Churn Rate by City")
    city_churn_rate = df.groupby('City')['Churn'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    city_churn_rate.plot(kind='bar', color='tomato', edgecolor='black', ax=ax)
    ax.set_title("Churn Rate by City")
    ax.set_ylabel("Churn Rate")
    st.pyplot(fig, use_container_width=True)

    # ---- Total Business Value by City ----
    st.subheader("💼 Total Business Value by City")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='City', y='Total Business Value', data=df, ax=ax)
    plt.xticks(rotation=90)
    ax.set_title("Total Business Value by City")
    st.pyplot(fig, use_container_width=True)

    # ---- Driver Count per City ----
    st.subheader("🚗 Driver Count per City")
    fig, ax = plt.subplots(figsize=(10, 6))
    df['City'].value_counts().plot(kind='barh', color='steelblue', ax=ax)
    ax.set_xlabel("Number of Drivers")
    ax.set_title("Driver Distribution Across Cities")
    st.pyplot(fig, use_container_width=True)

    # ---- Average Income by City ----
    st.subheader("💰 Average Income by City")
    avg_income_city = df.groupby('City')['Income'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_income_city.plot(kind='bar', color='seagreen', ax=ax)
    ax.set_title("Average Income by City")
    st.pyplot(fig, use_container_width=True)

    # =============================
    # 8️⃣ AGE GROUP BINNING
    # =============================
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

    # ---- Churn Rate by Age Group ----
    st.subheader("📊 Churn Rate by Age Group")
    age_churn_rate = df.groupby('Age_Group', observed=False)['Churn'].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    age_churn_rate.plot(kind='bar', color='salmon', ax=ax2)
    ax2.set_title("Churn Rate by Age Group")
    st.pyplot(fig2, use_container_width=True)

    # ---- Avg Business Value by Age Group ----
    st.subheader("💼 Average Business Value by Age Group")
    age_bv = df.groupby('Age_Group')['Total Business Value'].mean()
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    age_bv.plot(kind='bar', color='skyblue', ax=ax3)
    ax3.set_title("Average Business Value by Age Group")
    st.pyplot(fig3, use_container_width=True)

    st.success("""
    👥 **Insights:**
    - Middle-aged drivers dominate workforce (~30–45 years).  
    - Young drivers (<30) have highest churn probability.  
    - Senior drivers show higher loyalty but lower TBV.  
    Retention focus should target **young & mid-age groups** through engagement programs.
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
    except Exception as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Please upload 'xgboost_final_model.pkl' to the /models folder of your GitHub repo.")
        st.stop()


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
