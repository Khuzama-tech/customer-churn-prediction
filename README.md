# 📉 Customer Churn Prediction Project

> *A comprehensive, end-to-end machine learning pipeline for analyzing, predicting, and segmenting customer churn.*

---

## 🚀 Week 1: Exploratory Data Analysis

### 📁 Dataset Overview
* **Source:** Telco Customer Churn (**Kaggle**)
* **Size:** 7,043 customers and 21 features
* **Target Variable:** Predict customer churn (**Yes/No**)

### 🔍 Key Findings & Insights
* **Class Balance:** **26.54%** of customers churned, while **73.46%** remained (**indicating a moderate class imbalance**).
* **Tenure Relationship:** Customers with shorter tenure (**0-12 months**) generally exhibit significantly higher churn rates.
* **Billing Impact:** Higher monthly charges are strongly associated with increased customer churn probability.
* **Contract Types:** Month-to-month contract customers show much higher churn compared with 1-year or 2-year contract holders.
* **Service Impact:** Internet service type (**particularly Fiber Optic**) shows noticeable differences in churn behavior.
* **Payment Preferences:** Payment method (**such as Electronic check**) reveals distinct patterns linked to higher turnover.
* **Feature Correlation:** TotalCharges generally increases consistently with customer tenure.

### 🧹 Data Cleaning & Preprocessing Notes
* **Missing Values:** The TotalCharges column contains empty white spaces for brand-new customers, which must be converted to numeric values (**NaN**) and filled or dropped.
* **Irrelevant Features:** The customerID column carries no predictive power and should be removed prior to modeling.
* **Target Encoding:** The binary target column (**Churn**) needs to be mapped to numerical values (1/0) for future machine learning pipelines.

### ⚙️ Environment Setup
Install the required libraries locally or run inside a Kaggle notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 🚀 Week 2: Building ML Models

| Metric / Parameter | Details |
| :--- | :--- |
| **Baseline ("Always Stay")** | Accuracy: `0.73` |
| **Best Performing Model** | Random Forest |
| **Model Metrics** | AUC: `0.84` \| Recall: `0.78` @ threshold `0.35` |
| **Top Churn Drivers** *(Permutation Importance)* | Contract type, Monthly charges, Tenure |
| **Decision Threshold** | `0.35` (Chosen to minimize false negatives and catch at-risk customers early, balancing retention costs against revenue loss) |
| **Engineered Features** | `Tenure_to_Charges_Ratio`, `Total_Services_Used` |
| **Feature Engineering Impact** | AUC: `0.79` \(\rightarrow\) `0.84` |
| **Biggest Takeaway** | Strategic feature engineering and careful threshold tuning drastically improve model sensitivity on imbalanced customer churn datasets. |

# 🚀 Week 3: Model Optimization and Unsupervised Learning

## Overview & Methodology
This module covers advanced machine learning techniques, hyperparameter optimization, model evaluation under cross-validation, and unsupervised learning workflows (K-Means clustering and Principal Component Analysis) applied to customer churn prediction.

## Key Results & Metrics
- **Split-to-Split Stability**: 
  - Minimum Accuracy: $0.780$
  - Maximum Accuracy: $0.828$
  - Standard Deviation: $0.0104$
  - Theoretical 95% Confidence Interval: ($\pm 0.021$)  (Standard Error: $0.0107$)
- **5-Fold Cross-Validation AUC:** 
  - Logistic Regression (tuned C): \(0.8464 \pm 0.0129\)
  - Random Forest (random search): \(0.8464 \pm 0.0114\)
  - XGBoost (tuned): \(0.8502 \pm 0.0117\)
- **Final Model Evaluation:** 
  - Test AUC of final optimized model (XGBoost tuned, evaluated once): **0.8483** (falling safely within the CV mean \(\pm 2\) standard deviations range of $0.8268$ to $0.8736$).

## Unsupervised Learning & Dimensionality Reduction
- **Customer Segments (\(k = 4\)):**
  - *Active High Spenders* (Cluster 1): 2,157 customers, mean tenure 18.38 months, monthly charge \(\$80.41\), 3.28 services, 43% churn rate.
  - *Short-Tenure Low Spenders* (Cluster 3): 1,918 customers, mean tenure 8.96 months, monthly charge \(\$37.71\), 1.20 services, 32% churn rate.
  - *Loyal Power Users* (Cluster 2): 1,938 customers, mean tenure 59.83 months, monthly charge \(\$92.09\), 5.06 services, 14% churn rate.
  - *Stable Budget Users* (Cluster 0): 1,030 customers, mean tenure 53.61 months, monthly charge \(\$30.96\), 1.48 services, 5% churn rate.
- **Principal Component Analysis (PCA):** 
  - 15 of 30 components explain 90% of the cumulative variance.
  - PC1 top loadings (\(\sim 0.302\)) are dominated by identical positive loadings across dummy variables indicating a lack of internet service (`InternetService_No`, `OnlineSecurity_No internet service`, `TechSupport_No internet service`), revealing strong multicollinearity.

## Key Takeaway
- **Biggest Lesson:** Tuning and advanced modeling yield minor performance gains on tabular datasets like Telco Churn because simpler models already capture most primary linear patterns. Proper cross-validation and evaluation protocols ensure models generalize robustly without overfitting to specific random splits.
