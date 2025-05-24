# 📊 Feature Scores and Selection

This folder contains the feature importance analysis and selection results using multiple methods applied on our dataset of 183 features. The goal is to identify the most informative features contributing to the performance of our Partitioned Ensemble Classifier (PEC) model.

---

## 🔍 1. Average Feature Importance Across Partitions

We compute average feature importances from models trained on partitioned data using:

- Random Forest (`rf_feature_scores.ipynb`)
- XGBoost (`xgb_feature_scores.ipynb`)

These notebooks generate:

- `RF/partition_avg_feature_scores.csv`
- `XGB/partition_avg_feature_scores.csv`

Each CSV contains the average importance of each feature, ranked by relevance.

Additionally, both notebooks evaluate PEC's performance using the top K features incrementally (from 1 to all 183). The corresponding performance metrics (accuracy, F1 score, etc.) are reported in:

- `RF/feature_improvement_metrics.csv`
- `XGB/feature_improvement_metrics.csv`

These files illustrate how model performance evolves as more top features are added.

---

## ⚖️ 2. SHAP (Shapley Additive Explanations) Analysis

SHAP values are used to evaluate feature impact in the PEC framework. The findings show that the **top 10 features** identified using SHAP values are consistent with those derived from average feature importance.

Code:
- `rf_pec_shap.ipynb`
- `xgb_pec_shap.ipynb`

---

## 🔁 3. Recursive Feature Elimination (RFE)

We apply RFE to iteratively remove the least important feature (based on average importance across partitioned models), retraining the model each time, until only one feature remains.

Code:
- `rf_feature_selection_rfe.py`
- `xgb_feature_selection_rfe.py`

Outputs:
- `RF/RFE_RF_top_features.txt`
- `XGB/RFE_XGB_top_features.txt`

These text files list all features ranked from most important (top) to least important (bottom), based on the RFE process.

---