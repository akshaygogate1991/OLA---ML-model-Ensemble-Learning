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
    # Continuous columns
    continuous_cols = ['Age', 'Income', 'Total Business Value']

    for col in continuous_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    # Categorical columns
    categorical_cols = ['Gender', 'City', 'Education_Level', 'Joining Designation', 'Grade', 'Quarterly Rating']

    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
        ax.set_title(f'Countplot of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    st.write("""
    🔹 Age Distribution The age distribution is positively skewed, with most drivers aged between 28 and 38 years.
    The modal age group is around 32–35 years, highlighting this as the most common age bracket.
    Very few drivers are older than 45, and drivers above 50 are rare.
    Implication: Ola’s driver base is relatively young. Retention strategies should focus on early-career engagement, career progression, and long-term incentives.


    🔹 Income Distribution Income is also right-skewed, with a long tail toward higher earnings.
    Most drivers earn between ₹35,000 and ₹75,000, with a peak in the ₹50,000–₹60,000 range.
    High earners (above ₹1,25,000) are rare.
    Implication: While income varies widely, the majority fall in a mid-income bracket. Income could be a strong predictor of churn, with differentmotivations for low- vs. high-income drivers.

    🔹 Total Business Value (TBV) Extremely right-skewed with a sharp spike near zero and a long tail of high performers.
    Many drivers show minimal TBV, possibly due to short tenure or low activity.
    Implication: Apply log transformation to normalize TBV. Investigate high TBV outliers as potential top performers or data errors.

    🔹 Gender Two encoded categories: 0.0 and 1.0.
    Category 0.0 is slightly more common (11,000 drivers) than 1.0 (8,000).
    Implication: Mild imbalance—may be usable as-is, but monitor for bias in churn modeling.

    🔹 City Drivers are spread across many cities, with C20, C29, and C26 having the highest counts (~1,000 each).
    Smaller cities have ~400–500 drivers.
    Implication: A few urban centers dominate the driver population. Consider grouping low-volume cities into an "Other" category or applyingtarget encoding.

    🔹 Education Level Encoded as 0, 1, 2.
    Fairly balanced: Level 1 (6,900) slightly ahead of 2 (6,300) and 0 (~5,900).
    Implication: Drivers come from diverse educational backgrounds, likely not a strong standalone churn predictor but may interact with Gradeor Income.

    🔹 Joining Designation Highly imbalanced:
    Designation 1 dominates (~9,800 drivers). 
    Designations 2 (6,000), 
    Designation 3 (2,800) are less common.
    Designations 4 and 5 are rare (<500).
    Implication: Most drivers join at the lowest level. Consider grouping rare designations or using ordinal encoding.
    
    🔹 Grade Majority:
    in Grade 2 (6,600), followed by Grades 1 and 3 (5,000 each).Grades 4 (~2,100) and 5 (<500) are rare.
    Implication: Strong central tendency in grading. Merge sparse categories for modeling stability.
    
    🔹 Quarterly Rating Strongly skewed toward lower ratings:
    Rating 1 has ~7,600 drivers.
    Ratings 2–4 decline sharply.
    Implication: Potential link between low performance and churn. A candidate for feature interaction with TBV or Grade.

    
    🔑 Key Takeaways for Modeling: Skewed variables (Age, Income, TBV) require transformation or binning.
    High cardinality (City) and imbalanced variables (Grade, Designation, Rating) need careful encoding.
    Categorical variables show interpretable patterns, especially where performance or tenure may vary (e.g., Ratings, Grades).
    Target features like TBV, Income, and Rating may have strong predictive power for churn. """)

    st.write("# Bivariate graphical checking")
    st.write("# age vs income")
    st.write("# Scatter plot for continuous-continuous relationships")
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))
    # Scatter plot
    sns.scatterplot(data=df, x='Age', y='Income', ax=ax)
    # Titles and labels
    ax.set_title('Age vs Income')
    ax.set_xlabel('Age')
    ax.set_ylabel('Income')\
    # Layout adjustment
    plt.tight_layout()
    # Display plot in Streamlit
    st.pyplot(fig)

    
