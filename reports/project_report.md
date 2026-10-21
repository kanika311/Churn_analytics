# TelcoPulse AI: Customer Churn Intelligence & Retention Platform
## Comprehensive Analytics & Machine Learning Project Report

**Domain:** Telecommunications & Customer Lifecycle Analytics  
**Methodology:** CRISP-DM (Cross-Industry Standard Process for Data Mining)  
**Tools Stack:** Python, Pandas, Scikit-learn, SQLite, Plotly, Streamlit  

---

## 1. Abstract
Customer attrition poses a severe threat to recurring revenue in telecommunications. This project develops an end-to-end predictive analytics and retention intelligence system (**TelcoPulse AI**) on 7,043 customer accounts. Through rigorous data cleaning, feature engineering (lifecycle cohorting, ecosystem stickiness index, spend velocity), and supervised learning across multiple classifiers, we established that **balanced Logistic Regression** delivered the superior operational trade-off, achieving an **ROC-AUC of 0.82** and **75.4% Recall** on minority churners. Integrating this model into an enterprise-grade Streamlit cockpit empowers decision-makers with real-time risk scoring, cohort slicers, a "What-If" retention strategy simulator, and an interactive SQL analytics studio.

---

## 2. Problem Statement & Business Objectives
1. **Identify Root Causes of Churn:** Uncover the primary behavioral, contractual, and service features driving customer departures.
2. **Predict Flight Risk with High Sensitivity:** Build a robust classifier prioritizing **Recall** to minimize costly False Negatives (unidentified churners).
3. **Preserve Recurring Revenue via Proactive Intervention:** Provide an interactive simulator that calculates the Net Financial Gain of targeted retention offers (contract upgrades, discounts, bundled tech support).
4. **Democratize Data Access:** Enable real-time ad-hoc query capabilities via an in-memory SQL execution studio.

---

## 3. Dataset Architecture
The dataset encompasses **7,043 customer records** with **20 original attributes** plus engineered features:

| Feature Category | Variables | Description |
|---|---|---|
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Customer household and demographic profile |
| **Account Lifecycle** | `tenure`, `tenure_group`, `Contract` | Duration with provider and contract commitment level |
| **Services Ecosystem** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `num_services` | Service adoption, connectivity type, and bundled add-ons |
| **Billing & Finance** | `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `avg_monthly_spend` | Invoicing method, payment automation, and spend velocity |
| **Target Variable** | `Churn` | Binary indicator (1 = Churned, 0 = Retained) |

---

## 4. Data Engineering Pipeline

### 4.1 Data Cleaning & Imputation
- **TotalCharges Whitespace Resolution:** Converted `TotalCharges` from string to float. 11 records with zero tenure had whitespace entries; these were imputed with 0.0 to reflect non-billed new accounts.
- **Deduplication & Cardinality Check:** Removed any duplicate records and isolated `customerID` for individual account tracking.

### 4.2 Feature Engineering
1. **`tenure_group`:** Binned continuous tenure into actionable lifecycle stages: `0-12m`, `13-24m`, `25-48m`, `49-72m`.
2. **`num_services`:** Count of active security, backup, device protection, tech support, and streaming subscriptions (measuring ecosystem lock-in).
3. **`avg_monthly_spend`:** Computed as `TotalCharges / tenure` to identify anomalous billing spikes versus baseline monthly charges.

---

## 5. Exploratory Data Analysis & Strategic Insights

1. **Contract Type is the Dominant Flight Driver:**
   - **Month-to-Month contracts** experience a staggering **~42.7% churn rate**, compared to **~11.3%** for 1-year and **~2.8%** for 2-year contracts.
2. **Onboarding Fragility in Year 1:**
   - Customers in their first 12 months exhibit an attrition rate of **47.4%**. Attrition decays to **< 10%** once tenure exceeds 48 months.
3. **Service Stickiness Protects Accounts:**
   - Customers with **0 add-ons** churn at **45.2%**, whereas customers with **3 or more bundled services** churn at under **14.8%**. Tech Support and Online Security cut attrition in half.
4. **Billing Method Friction:**
   - **Electronic check** users churn at **45.3%**, whereas automated bank transfer and credit card customers churn at under **17%**.

---

## 6. Machine Learning Model Benchmarking

Models were trained on an 80/20 stratified split with standard scaling for continuous features and one-hot encoding for categorical variables. 5-fold cross-validation was used to ensure generalizability.

### Model Leaderboard
| Model | Accuracy | Precision | Recall | F1-Score | Test ROC-AUC | 5-Fold CV ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression (Balanced)** | **0.748** | **0.539** | **0.754** | **0.629** | **0.819** | **0.821** |
| **Random Forest (Balanced)** | 0.747 | 0.546 | 0.627 | 0.583 | 0.801 | 0.811 |
| **Gradient Boosting** | 0.783 | 0.675 | 0.449 | 0.539 | 0.805 | 0.817 |

### Why Logistic Regression is the Optimal Operational Model:
In commercial churn mitigation, the cost asymmetry is significant:
- **Cost of False Negative:** Losing a customer represents **$800 - $1,200** in lost Customer Lifetime Value (CLV).
- **Cost of False Positive:** Reaching out to a loyal customer with a retention perk costs **$15 - $25**.

Logistic Regression with balanced class weights caught **75.4% of actual churners** (Recall), outperforming Gradient Boosting which missed over 55% of churners due to threshold bias on the imbalanced target.

---

## 7. The TelcoPulse AI Enterprise Cockpit Architecture

The application has been transformed into a full-scale analytical dashboard featuring:

1. **Executive Command Cockpit:**
   - Real-time KPI cards: Total Cohort, Churn Rate %, Portfolio MRR, MRR at Risk / Month, Annualized Risk Exposure, High-Risk Account Count.
   - Dynamic AI Executive Synthesis synthesizing risk exposure for filtered segments.
   - Interactive Plotly visualizations: Sunburst revenue hierarchy, tenure retention decay curve, charges distribution.
2. **Behavioral & Ecosystem Analytics:**
   - Add-on service stickiness multiplier.
   - Value vs. Flight Risk 4-quadrant portfolio scatter matrix.
3. **AI Predictor & "What-If" Retention Studio:**
   - Instant single-customer risk scoring with animated Plotly gauge.
   - Live intervention testing (Contract upgrades, bill discounts, complimentary tech support, auto-pay incentives) with dynamic calculation of new churn probability and Net Financial Gain.
4. **Customer Risk Explorer & CRM Hub:**
   - Searchable, sortable customer risk table with color-coded risk tier badges.
   - Deep-dive customer dossier inspector.
   - 1-Click CSV export for integration with CRM outreach campaigns.
5. **Machine Learning Lab:**
   - Model benchmark leaderboard and ROC curves.
   - Interactive decision threshold slider with live recalculation of Confusion Matrix, Recall, and Precision.
6. **SQL Analytics Studio:**
   - Built-in SQLite in-memory database with pre-loaded analytical queries and a custom SQL editor.
   - Execution performance timer (ms) and automatic chart rendering for query outputs.
7. **Viva & Knowledge Center:**
   - Interactive accordion covering core concepts, math formulations, and business defense rationales.

---

## 8. Actionable Business Recommendations
1. **Contract Migration Program:** Incentivize Month-to-Month fiber optic subscribers to migrate to 1-Year agreements via a 15% discount for the first 3 months.
2. **Onboarding "First 90 Days" Program:** Deploy proactive customer care touchpoints during the high-attrition 0-12 month tenure window.
3. **Automated Payment Incentive:** Offer a one-time $10 credit to migrate electronic check customers to automated ACH or credit card autopay.
4. **Security & Tech Support Bundling:** Package basic tech support and antivirus protection as standard offerings for high-tier internet tiers to cultivate ecosystem stickiness.

---
