# Predictive-Customer-Analytics-for-E-commerce
Project Documentation: Predictive Customer Analytics for E-commerce
Author: Data Analytics Department
Date: 03 October 2025
Version: 1.2

1. Executive Summary
Objective: To build a comprehensive, data-driven framework for an e-commerce business to understand and predict customer behaviour, focusing on Customer Lifetime Value (CLV) and customer churn.

Methodology Used:

RFM (Recency, Frequency, Monetary) and advanced behavioral feature engineering from transactional data.

K-Means clustering for customer segmentation.

Probabilistic models (BG/NBD, Gamma-Gamma) for CLV prediction.

XGBoost machine learning model for churn prediction, enhanced with class imbalance handling.

Key Findings:

Customer base successfully segmented into 4 actionable personas: Champions, At-Risk, Potential Loyalists, and New Customers.

Successfully forecasted the 12-month CLV for each customer, identifying top-tier clients.

Built a high-accuracy churn model (AUC ≈ 1.00) and used SHAP analysis to identify Recency as the main churn predictor.

Business Impact: This project provides actionable intelligence for customer retention, helps optimize marketing spend by targeting high-value segments, and proactively mitigates revenue loss from customer churn.

2. Project Context
Core Problem: E-commerce companies need to identify their most valuable customers and predict which ones are likely to leave. This project provides a quantitative, data-driven solution.

Dataset:

Name: Online Retail II Dataset

Source: UCI Machine Learning Repository

Contents: UK-based online retail transactions from 2010-2011.

Tech Stack:

Data Handling: pandas, numpy

Machine Learning: scikit-learn, lifetimes, xgboost

Model Interpretation: shap

Visualization: matplotlib, seaborn

3. Phase 1: Data Preprocessing & Feature Engineering
Note: Model quality is dependent on the quality of the data. High-quality features are essential for accurate predictions.

Data Cleaning Checklist:

[x] Dropped rows with missing CustomerID.

[x] Filtered out cancelled orders (invoices starting with 'C').

[x] Removed transactions with zero or negative price/quantity.

[x] Created a TotalPrice column (Quantity * Price).

RFM Feature Engineering:

Recency (R): Days since last purchase (Lower is better).

Frequency (F): Total number of unique transactions (Higher is better).

Monetary (M): Total money spent (Higher is better).

# Code functionality for RFM calculation
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
    'Invoice': 'nunique',
    'TotalPrice': 'sum'
})

Advanced Feature Engineering: To add more depth, we also calculated:

Tenure: The total number of days between a customer's first and last purchase.

AvgTimeBetweenPurchases: The average time between a customer's consecutive purchases.

4. Phase 2: Customer Segmentation
Goal: To group similar customers into distinct segments based on their purchasing behaviour.

Methodology: K-Means Clustering:

An unsupervised algorithm that partitions data into a pre-defined number of 'k' clusters.

Data Preparation: The RFM data was log-transformed (to handle skewness) and then standardized using StandardScaler to ensure all features contributed equally.

Finding 'k': The Elbow Method was used to find the optimal number of clusters, which was determined to be four.

Customer Personas Created:

Champions: Our best customers. They buy recently, frequently, and spend the most.

At-Risk: High-value customers who used to be frequent buyers but have not purchased in a long time.

Potential Loyalists: Consistent customers with moderate recency and frequency.

New Customers: First-time or very recent buyers with low frequency.

5. Phase 3: Customer Lifetime Value (CLV) Prediction
Goal: To forecast the future monetary value of each customer.

Methodology: Probabilistic Models (lifetimes library):

BG/NBD Model: Predicts the number of future purchases a customer is likely to make, based on their historical recency and frequency.

Gamma-Gamma Model: Predicts the average monetary value of those future purchases.

Implementation:

# Code functionality for fitting the models and predicting CLV
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(clv_data['frequency'], clv_data['recency'], clv_data['T'])

ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(clv_data['frequency'], clv_data['monetary_value'])

clv_data['predicted_clv_12_months'] = ggf.customer_lifetime_value(...)

Result: A 12-month CLV forecast was generated, enabling the business to quantify the future value of its customer base and identify high-priority clients.

6. Phase 4: Customer Churn Prediction
Goal: To proactively build a model that identifies customers who are likely to stop purchasing.

Methodology: XGBoost Classifier:

Churn Definition: A customer is flagged as "churned" if their Recency is greater than 180 days.

Model Choice: XGBoost was selected for its high predictive accuracy.

Handling Class Imbalance: The scale_pos_weight parameter was used to give more weight to the minority "churn" class during model training, ensuring the model does not become biased.

Evaluation:

The model achieved a near-perfect AUC (Area Under the Curve) score of ~1.00, indicating an excellent ability to distinguish between churning and non-churning customers.

Model Interpretation with SHAP:

SHAP (SHapley Additive exPlanations) was used to explain the XGBoost model's decisions.

Key Insight: The analysis confirmed that Recency is by far the most dominant factor influencing churn predictions. Customers who have not purchased in a long time are very likely to be predicted as churned.

7. Actionable Insights & Next Steps
Business Recommendations:

For At-Risk Customers: Launch targeted win-back campaigns (e.g., personalized emails with discounts), especially for those who had a high monetary value in the past.

For Champions: Implement a loyalty or VIP program to maintain engagement and reward their loyalty.

For Potential Loyalists: Nurture these customers with timely promotions to increase their purchase frequency and move them into the "Champions" segment.

Future Work:

Deploy the models as a live API for real-time predictions.

Integrate the segmentation and churn scores into a Customer Relationship Management (CRM) system for the marketing team.

Experiment with different churn definitions (e.g., 90 days instead of 180) to see how it impacts the model.
