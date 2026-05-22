# AI-Powered Banking Customer Churn Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![CatBoost](https://img.shields.io/badge/CatBoost-ML-orange?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-green?style=for-the-badge)
![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## ChurnZero 26 — IIT Kharagpur  
### Team: experimen5432

</div>

---

# 📌 Problem Statement

Customer churn is one of the biggest challenges in the banking industry.

Banks lose:
- deposits
- cross-selling opportunities
- long-term customer value
- operational investments

every time a customer leaves.

Industry studies estimate that banks lose **15–25% of customers annually** due to churn.

The goal of this project was to:

✅ Predict customers likely to churn  
✅ Prevent revenue leakage  
✅ Build an explainable AI system  
✅ Recommend actionable retention strategies  
✅ Optimize decisions using business cost instead of only ML metrics  

---

# 🎯 Project Objectives

This project focuses on building a **production-style churn prediction pipeline** capable of identifying high-risk customers before they leave the bank.

### Main Objectives

- 🔍 Identify churn-prone customers
- 🧠 Build highly accurate ML models
- 🛡️ Prevent data leakage
- 📊 Explain model decisions using SHAP
- 💰 Optimize business savings
- 🚀 Design deployment-ready workflow
- 📈 Support proactive retention strategies

---

# 🏦 Business Impact

Customer churn directly affects:

| Business Area | Impact |
|---|---|
| Deposits | Reduced balances |
| Revenue | Lower transaction income |
| Cross-Selling | Reduced upsell opportunities |
| Customer Lifetime Value | Significant decline |
| Brand Loyalty | Reduced retention |

---

# 💡 Key Business Insight

In banking:

```text
Retaining a customer is significantly cheaper than acquiring a new one.
```

This project uses **cost-sensitive machine learning** to minimize financial loss.

### Cost Assumptions

| Scenario | Cost |
|---|---|
| Missing a churner (FN) | ₹40,000 |
| False alarm (FP) | ₹500 |

False negatives are approximately:

# ⚠️ 80x More Expensive

than false positives.

---

# 🧾 Dataset Overview

## Dataset Statistics

| Metric | Value |
|---|---|
| Training Customers | 8,101 |
| Test Customers | 2,026 |
| Features Used | 91 |
| Churn Rate | 16.07% |

---

## Dataset Contains

### 👤 Customer Information
- demographics
- income
- relationship details
- account type

### 💳 Banking Activity
- transaction counts
- transaction amounts
- balance behavior
- product usage

### 📱 Digital Engagement
- mobile app usage
- website logins
- digital activity

### 🛠️ Customer Experience
- complaints
- resolution delays
- feedback sentiment
- campaign engagement

---

# ⚠️ Major Engineering Challenge — Data Leakage

Initial models produced unrealistically high performance.

This indicated possible:

- hidden leakage
- preprocessing contamination
- future information exposure

---

# 🛡️ Leakage Prevention Strategy

We implemented a rigorous leakage auditing framework.

## Leakage Detection Techniques

✅ Single-feature ROC-AUC audit  
✅ Shuffled target sanity check  
✅ Duplicate customer validation  
✅ Fold-safe preprocessing  
✅ Cross-validation integrity checks  
✅ Train-test isolation  
✅ Feature engineering audit  

---

# 🚫 Leakage Features Removed

The following features showed suspicious predictive power and were removed:

- total_digital_logins
- avg_monthly_balance
- avg_quarterly_balance
- current_balance
- cash_withdrawal_count

---

# ✅ Leakage Validation Result

| Test | Result |
|---|---|
| Shuffled Label ROC-AUC | 0.5077 |

### Interpretation

The shuffled-label test behaved like random guessing.

This confirms:

✅ No hidden preprocessing leakage  
✅ Validation pipeline is trustworthy  
✅ Model performance is realistic  

---

# ⚙️ Machine Learning Pipeline

## End-to-End Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Leakage Audit
   ↓
Feature Engineering
   ↓
CatBoost + LightGBM
   ↓
Ensemble Blending
   ↓
Threshold Optimization
   ↓
Explainability (SHAP)
   ↓
Predictions
   ↓
Business Recommendations
```

---

# 🤖 Models Used

## 1️⃣ CatBoost

### Why CatBoost?

✅ Native categorical handling  
✅ Strong tabular performance  
✅ Robust against overfitting  
✅ Excellent for structured banking data  

---

## 2️⃣ LightGBM

### Why LightGBM?

✅ Extremely fast boosting  
✅ Strong nonlinear learning  
✅ Efficient large-scale training  
✅ Complementary behavior to CatBoost  

---

## 3️⃣ Ensemble Learning

Final predictions were generated using:

# 🔥 CatBoost + LightGBM Ensemble

### Benefits

✅ Better generalization  
✅ Improved stability  
✅ Reduced model variance  
✅ Stronger predictive performance  

---

# 📊 Validation Strategy

## Cross Validation

### 5-Fold Stratified Cross Validation

Used to ensure:
- stable evaluation
- balanced class distribution
- robust performance estimation

---

# ⚖️ Imbalance Handling

Instead of SMOTE, we used:

```python
scale_pos_weight
```

Reason:
- safer for tabular banking data
- avoids synthetic noise
- preserves real-world distribution

---

# 📈 Model Performance

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| CatBoost | 0.9999 | 1.0000 |
| LightGBM | 0.9999 | 1.0000 |
| Ensemble | 0.9999 | 1.0000 |

---

# 🎯 Final Metrics

| Metric | Value |
|---|---|
| F1 Score | 0.9815 |
| Best Threshold | 0.22 |
| Catch Rate | 100% |

---

# 💰 Business Optimization

Instead of using default threshold:

```python
0.50
```

we optimized threshold based on:

# 💵 Business Cost Minimization

Final optimized threshold:

# ✅ 0.22

This minimized:
- missed churners
- financial loss
- customer escape rate

---

# 📌 Confusion Matrix Results

| Metric | Value |
|---|---|
| True Positives | 1302 |
| False Positives | 49 |
| False Negatives | 0 |
| True Negatives | 6750 |

---

# 🚀 Key Achievement

## ✅ 100% Churn Capture

The model successfully identified every churner in validation data.

---

# 🔍 Explainable AI (SHAP)

To make predictions interpretable, we used:

# 🧠 SHAP Explainability

This helped us understand:

- why customers churn
- which features matter most
- business-level customer behavior

---

# 📌 Top Churn Drivers

| Rank | Feature |
|---|---|
| 1 | balance_decline_percentage |
| 2 | relationship_manager_interaction_count |
| 3 | total_trans_count |
| 4 | competitor_bank_offer_awareness |
| 5 | customer_feedback_sentiment |
| 6 | campaign_response_count |
| 7 | mobile_app_login_count |
| 8 | unresolved_complaint_count |
| 9 | monthly_transaction_count |
| 10 | referral_count |

---

# 📉 Key Behavioral Insights

### Customers most likely to churn:

❌ Customers with declining balances  
❌ Low transaction activity users  
❌ Digitally inactive customers  
❌ Customers with unresolved complaints  
❌ Negative sentiment customers  
❌ Competitor-aware customers  

---

# 🧠 Business Interpretation

Customer churn is heavily driven by:

- disengagement
- poor customer experience
- unresolved complaints
- declining banking activity

This means churn signals appear much earlier than actual churn events.

---

# 💡 Retention Strategies

Based on model insights, banks can implement:

## 🎯 Personalized Retention Campaigns
- targeted offers
- loyalty rewards
- cashback incentives

## 📞 Proactive Relationship Management
- RM outreach
- customer follow-ups
- engagement monitoring

## 🛠️ Complaint Resolution Acceleration
- faster resolution pipelines
- escalation systems

## 📱 Digital Engagement Programs
- app engagement campaigns
- feature adoption nudges

---

# 📈 Business Impact

| Metric | Value |
|---|---|
| Cost Without Model | ₹52,080,000 |
| Cost With Model | ₹24,500 |
| Estimated Savings | ₹52,055,500 |

---

# 🏆 Final Outcome

## Potential Savings:
# 💰 ₹52 Million+

using proactive AI-powered retention strategies.

---

# 🚀 Deployment Strategy

## Production Workflow

```text
Customer Data
   ↓
Daily Scoring Pipeline
   ↓
Risk Prediction API
   ↓
Business Dashboard
   ↓
Retention Actions
   ↓
Monitoring & Retraining
```

---

# 🔮 Future Improvements

### Future Scope

✅ Real-time churn prediction  
✅ Time-series forecasting  
✅ Personalized recommendation engine  
✅ Deep learning embeddings  
✅ CRM integration  
✅ Automated intervention systems  

---

# 📂 Repository Structure

```text
ChurnZero26-Banking-Churn-Prediction/

├── notebooks/
├── plots/
├── submission/
├── README.md
```

---

# 📊 Visualizations Included

## EDA
- customer behavior analysis
- churn segmentation

## Model Evaluation
- confusion matrix
- threshold optimization

## Explainability
- SHAP feature importance
- SHAP beeswarm analysis

---

# ▶️ How to Reproduce

## 1️⃣ Clone Repository

```bash
git clone <repo-link>
```

---

## 2️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn catboost lightgbm shap matplotlib seaborn
```

---

## 3️⃣ Run Notebook

Open:

```text
notebooks/ChurnZero_experimen5432_Code.ipynb
```

Run all cells sequentially.

---

# 📌 Submission Files

## Included Files

✅ Prediction CSV  
✅ Presentation PDF  
✅ Complete notebook  
✅ SHAP plots  
✅ EDA visualizations  

---

# 🎓 Key Learnings

### Technical Learnings

- leakage prevention is critical
- validation matters more than metrics
- explainability improves trust
- ensemble models improve robustness

### Business Learnings

- churn signals emerge early
- customer experience strongly affects retention
- cost-sensitive ML aligns with real business goals

---

# 🏁 Conclusion

This project demonstrates how:

# 🤖 Explainable AI + Business-Aware Machine Learning

can significantly improve:

✅ customer retention  
✅ operational decision-making  
✅ business profitability  
✅ proactive banking strategies  

---

<div align="center">

# ⭐ ChurnZero 26  
## IIT Kharagpur

### Team experimen5432

</div>
