
# Customer Churn Prediction Project
## Week 1: Exploratory Data Analysis
## Dataset Overview
**Source:** Telco Customer Churn (Kaggle)

**Size:** 7,043 customers and 21 features

**Target Variable:** Predict customer churn (Yes/No)

## Key Findings & Insights
**Class Balance:** 26.54% of customers churned, while 73.46% remained (indicating a moderate class imbalance).

**Tenure Relationship:** Customers with shorter tenure (0-12 months) generally exhibit significantly higher churn rates.

**Billing Impact:** Higher monthly charges are strongly associated with increased customer churn probability.

**Contract Types:** Month-to-month contract customers show much higher churn compared with 1-year or 2-year contract holders.

**Service Impact:** Internet service type (particularly Fiber Optic) shows noticeable differences in churn behavior.

**Payment Preferences:** Payment method (such as Electronic check) reveals distinct patterns linked to higher turnover.

**Feature Correlation:** TotalCharges generally increases consistently with customer tenure.

## Data Cleaning & Preprocessing Notes
**Missing Values:** The TotalCharges column contains empty white spaces for brand-new customers, which must be converted to numeric values (NaN) and filled or dropped.

**Irrelevant Features:** The customerID column carries no predictive power and should be removed prior to modeling.

**Target Encoding:** The binary target column (Churn) needs to be mapped to numerical values (1/0) for future machine learning pipelines.

## Environment Setup
Install the required libraries locally or run inside a Kaggle notebook:
pip install pandas numpy matplotlib seaborn scikit-learn

