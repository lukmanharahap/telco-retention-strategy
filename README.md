# Telco Retention Strategy

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Random%20Forest-orange?logo=scikit-learn)](https://scikit-learn.org/)

> **Executive Summary:** An end-to-end Machine Learning solution designed to reduce customer churn. The system identifies high-risk customers, quantifies financial loss ($1.67M annual risk), and simulates the ROI of retention strategies using an interactive dashboard.

---

## Objectives
* Identify customers with high churn risk.
* Understand churn drivers that are actionable.
* Support prioritization for retention efforts.

---

## Background
The company faces a critical revenue leakage problem with a **26.5% churn rate**, translating to an estimated **$1.67 Million annual revenue loss**. 
Traditional "mass-marketing" retention strategies are inefficient (high cost, low precision). This project shifts the strategy to **ML-Driven Targeted Intervention** to focusing resources only on high-value and high-risk customers.

### Key Insights
Based on EDA (Exploratory Data Analysis), the root causes of churn are:
1.  **Month-to-Month Contract:** Customers on monthly contracts are **20x more likely to churn** than those on 2-year contracts.
2.  **Fiber Optic Dissatisfaction:** Despite being a premium service, Fiber Optic users have a **41.9% churn rate**.
3.  **High-Value Vulnerability:** Churn is highest among customers paying **$70-$100/month** within their first year (Tenure < 12 months).

---

## The Solution
This project deploys a **Random Forest Classifier** wrapped in a **Streamlit Web App** that serves as a Decision Support System for marketing managers.

**🔗 [Live Demo Link]** *([Dashboard](https://telco-retention-strategy.streamlit.app/))*

### Key Features
1.  **Real-Time Risk Scoring:** Inputs customer data to predict Churn Probability.
2.  **Financial ROI Simulator:** Calculates the net profit of retention campaigns based on Intervention Cost vs. Customer LTV (Lifetime Value).
3.  **What-If Analysis:** Recommends the best action.
4.  **Logic Guardrails:** Input validation ensures simulations remain within realistic business constraints (e.g., pricing ranges based on historical data).

---

## Technical Implementation

### 1. Data Pipeline & Feature Engineering
* **Handling Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set to prevent model bias towards the majority class.
* **Feature Preservation:** Using careful One-Hot Encoding to ensure high interpretability.
* **Input Guardrails:** Implemented dynamic min/max constraints in the dashboard based on real data distribution to prevent out-of-distribution predictions.

### 2. Modeling & Evaluation
* **Algorithm:** Random Forest Classifier (n_estimators=100).
* **Why Random Forest?** Chosen for its robustness against noise, ability to handle non-linear relationships in customer behavior, and feature importance interpretability.
* **Performance:**
    * **Recall (Churn Capture):** 62% (Successfully identifies ~6 out of 10 potential churners).
    * **Precision:** Optimized to balance intervention costs.

### 3. Business Simulation Logic
The dashboard calculates uplift using the formula:
$$\text{Net Profit} = (\text{LTV} \times \text{Recall} \times \text{Success Rate}) - (\text{Total Intervention Cost})$$  
Assume that the retention cost is $65 per customer that reflects 1 month free with a **Success Rate of 50%**, the simulation shows:
* **Do Nothing Strategy:** Results in a potential loss of **$290,650** due to unmitigated churn.
* **Mass Campaign Strategy:** Results in a net profit of **$53,740** due to high costs and low precision.
* **ML-Targeted Strategy:** Results in a net profit of **$62,652**.
Results indicate that the ML-Targeted Strategy generates an estimated **+$8,912** compared to the Mass Campaign Strategy.

---

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lukmanharahap/telco-retention-strategy.git
    cd telco-rentention-strategy
    ```

2.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

---

## Author
**Lukman Harahap**
* Connect on [LinkedIn](https://www.linkedin.com/in/lukmanharahap/) | [Email](mailto:lukmanharahap28@gmail.com)