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

    st.write("# Graphical reprsentation for continous and categorical varibales")
    # Set style
    sns.set(style='whitegrid')
    # Continuous columns
    continuous_cols = ['Age', 'Income', 'Total Business Value']

    for col in continuous_cols:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        st.pyplot(fig,use_container_width=False)
    
    # Categorical columns
    categorical_cols = ['Gender', 'City', 'Education_Level', 'Joining Designation', 'Grade', 'Quarterly Rating']

    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
        ax.set_title(f'Countplot of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig,use_container_width=False)
    st.write("""
    🔹 Age Distribution The age distribution is positively skewed, with most drivers aged between 28 and 38 years.
    The modal age group is around 32–35 years, highlighting this as the most common age bracket.
    Very few drivers are older than 45, and drivers above 50 are rare.
    Implication: Ola’s driver base is relatively young. Retention strategies should focus on early-career engagement, career progression, 
    and long-term incentives.


    🔹 Income Distribution Income is also right-skewed, with a long tail toward higher earnings.
    Most drivers earn between ₹35,000 and ₹75,000, with a peak in the ₹50,000–₹60,000 range.
    High earners (above ₹1,25,000) are rare.
    Implication: While income varies widely, the majority fall in a mid-income bracket. Income could be a strong predictor of churn, with 
    differentmotivations for low- vs. high-income drivers.

    🔹 Total Business Value (TBV) Extremely right-skewed with a sharp spike near zero and a long tail of high performers.
    Many drivers show minimal TBV, possibly due to short tenure or low activity.
    Implication: Apply log transformation to normalize TBV. Investigate high TBV outliers as potential top performers or data errors.

    🔹 Gender Two encoded categories: 0.0 and 1.0.
    Category 0.0 is slightly more common (11,000 drivers) than 1.0 (8,000).
    Implication: Mild imbalance—may be usable as-is, but monitor for bias in churn modeling.

    🔹 City Drivers are spread across many cities, with C20, C29, and C26 having the highest counts (~1,000 each).
    Smaller cities have ~400–500 drivers.
    Implication: A few urban centers dominate the driver population. Consider grouping low-volume cities into an "Other" category or 
    applyingtarget encoding.

    🔹 Education Level Encoded as 0, 1, 2.
    Fairly balanced: Level 1 (6,900) slightly ahead of 2 (6,300) and 0 (~5,900).
    Implication: Drivers come from diverse educational backgrounds, likely not a strong standalone churn predictor but may interact 
    with Gradeor Income.

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
    st.write("age vs income")
    # Scatter plot for continuous-continuous relationships
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(5, 3))
    # Scatter plot
    sns.scatterplot(data=df, x='Age', y='Income', ax=ax)
    # Titles and labels
    ax.set_title('Age vs Income')
    ax.set_xlabel('Age')
    ax.set_ylabel('Income')\
    # Layout adjustment
    plt.tight_layout()
    # Display plot in Streamlit
    st.pyplot(fig,use_container_width=False)

    st.write("""
    1. Income Pattern Across Age • Income appears to increase with age until around 35–40 years, after which it starts to plateau or slightly
    decline. • This suggests that experience contributes to higher income in early to mid-career, but there may be diminishing returns or
    attrition effects after 45.
    2. High-Density Zone • There is a high concentration of points between: o Age: 28 to 40 years o Income: ₹40,000 to ₹100,000 • This
    indicates that the core workforce of drivers falls within this age-income bracket.
    3. Outliers • A few data points show incomes above ₹150,000, particularly between the ages of 28–38. • These may be top performers,
    special assignments, or anomalies worth investigating.
    4. Older Age Group Trends • Post age 45, the density of drivers drops, and their incomes also appear more scattered and lower. • This
    could indicate: o Lower income opportunities for older drivers o Early retirement or career transitions o Health or performance
    constraints impacting income
    5. Younger Drivers (20–25) • Income levels for drivers aged below 25 are relatively low and varied. • Possibly due to: o Being new to the
    platform o Fewer hours worked o Limited access to high-earning opportunities
    
    Insights & Implications • 
    Workforce Focus: Majority of drivers earning mid-to-high income are in the 30–40 age range. This could be the
    sweet spot for engagement, retention, and promotion. • Policy Direction: o Offer career growth plans for younger drivers. o Support older
    drivers with incentives or alternative roles if income tends to decline. • Modeling Note: There may be a non-linear relationship between age
    and income. Consider polynomial features or binning age when building predictive models.""")

    st.write("# grade vs income")
    st.write("Box plot for categorical-continuous relationships")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(5, 3))
    # Boxplot
    sns.boxplot(data=df, x='Grade', y='Income', ax=ax)
    # Titles and labels
    ax.set_title('Income by Grade')
    ax.set_xlabel('Grade')
    ax.set_ylabel('Income')
    # Layout adjustment
    plt.tight_layout()
    # Display plot in Streamlit
    st.pyplot(fig,use_container_width=False)

    st.write("""
    1. Positive Correlation 
    • There is a clear upward trend: As the Grade increases from 1 to 5, the median income rises significantly. 
    • This suggests that Grade is a strong indicator of driver income, and likely tied to experience, performance, or tenure.
    2. Median Incomes by Grade 
    • Grade 1: Median income around ₹40,000 
    • Grade 2: Median near ₹60,000 
    • Grade 3: Median around ₹85,000 
    • Grade 4: Median around ₹110,000 
    • Grade 5: Median around ₹135,000+ The growth is consistent and significant across grades.
    3. Interquartile Range (IQR) and Spread 
    • The spread increases with grade, especially for Grades 3 to 5. 
    • This indicates more variability in income at higher grades — likely due to performance-based pay or variable workloads.
    4. Outliers 
    • Some outliers are present at all grades, both high and low: o Lower-grade outliers show higher incomes (possibly high
    performers or bonuses). o Higher-grade outliers with lower incomes may suggest inactivity, part-time work, or data anomalies.
    5. Lower Bound Shift 
    • The minimum incomes also shift upward with grade, indicating even the lowest earners in higher grades still make
    more than most low-grade drivers.
    
    Insights & Implications 
    • Career Incentive Structure: The strong income–grade linkage suggests that promoting drivers to higher grades is
    financially rewarding and can be used to boost retention. 
    • Modeling Suggestion: Grade should be treated as a predictor variable in churn and income models — possibly even ordinal if treated numerically.""")

    # ---- Grade vs Churn (Stacked Bar Plot) ----
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    pd.crosstab(df['Grade'], df['Churn'], normalize='index').plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax1)
    ax1.set_title('Churn Rate by Grade')
    ax1.set_ylabel('Proportion')
    ax1.set_xlabel('Grade')
    ax1.legend(title='Churn', labels=['Active (0)', 'Left (1)'])
    plt.tight_layout()
    st.pyplot(fig1,use_container_width=False)

    # Gender vs Churn (Stacked Bar Plot) ----
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    pd.crosstab(df['Gender'], df['Churn'], normalize='index').plot(kind='bar', stacked=True, colormap='viridis', ax=ax2    )
    ax2.set_title('Churn by Gender')
    ax2.set_ylabel('Proportion')
    ax2.set_xlabel('Gender')
    plt.tight_layout()
    st.pyplot(fig2,use_container_width=False)

    # ---- Income vs Churn (Boxplot) ----
    fig3, ax3 = plt.subplots(figsize=(5,3))
    sns.boxplot(data=df, x='Churn', y='Income', ax=ax3)
    ax3.set_title('Income by Churn Status')
    ax3.set_xticklabels(['Active', 'Left'])
    plt.tight_layout()
    st.pyplot(fig3,use_container_width=False)

    # ---- Quarterly Rating vs Churn (Boxplot) ----
    fig4, ax4 = plt.subplots(figsize=(5,3))
    sns.boxplot(data=df, x='Churn', y='Quarterly Rating', ax=ax4)
    ax4.set_title('Quarterly Rating by Churn Status')
    ax4.set_xticklabels(['Active', 'Left'])
    plt.tight_layout()
    st.pyplot(fig4,use_container_width=False)

    # ---- Age vs Churn (Boxplot) ----
    fig5, ax5 = plt.subplots(figsize=(5,3))
    sns.boxplot(data=df, x='Churn', y='Age', ax=ax5)
    ax5.set_title('Age by Churn Status')
    ax5.set_xticklabels(['Active', 'Left'])
    plt.tight_layout()
    st.pyplot(fig5,use_container_width=False)

    # ---- Total Business Value vs Churn (Boxplot) ----
    fig6, ax6 = plt.subplots(figsize=( 5,3))
    sns.boxplot(data=df, x='Churn', y='Total Business Value', ax=ax6)
    ax6.set_title("Business Value vs Churn Status")
    plt.tight_layout()
    st.pyplot(fig6,use_container_width=False)

    st.write("# Checking correlation of columns")
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12,10 ))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)


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

    # removing duplicates
    # Find complete duplicate rows
    duplicate_rows = df[df.duplicated()]
    from sklearn.impute import KNNImputer


    # Select only the relevant columns for imputation
    subset_cols = ['Age', 'Gender']
    df_subset = df[subset_cols]

    # Apply KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_subset), columns=subset_cols)

    # Update the original DataFrame with imputed values
    df['Age'] = df_imputed['Age']
    df['Gender'] = df_imputed['Gender']

    # Once again, check for missing values
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

  # Checking by IQR method
    def cap_outliers_iqr(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
        return df


    # Select only numeric columns
    cols = df.select_dtypes(include=['number']).columns

    # Apply IQR capping for each numeric column
    for col in cols:
        df = cap_outliers_iqr(df, col)

    # flage creation
    # High Business Value Driver
    threshold_bv = df['Total Business Value'].quantile(0.90)
    df['High_Business_Value_Flag'] = (df['Total Business Value'] >= threshold_bv).astype(int)
    # Low Income Driver
    threshold_income = df['Income'].quantile(0.10)
    df['Low_Income_Flag'] = (df['Income'] <= threshold_income).astype(int)
    # Senior Age Group Flag
    df['Senior_Driver_Flag'] = (df['Age'] > 50).astype(int)
    # Recent Joiner Flag
    df['Recent_Joiner_Flag'] = (df['Tenure_Years'] < 1).astype(int)
    # Low Rating Flag
    df['Low_Rating_Flag'] = (df['Quarterly Rating'] <= 2).astype(int)
    







   
