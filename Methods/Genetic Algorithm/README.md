# 🧬 Genetic Algorithm-Based Feature Selection

This repository contains a genetic algorithm (GA) pipeline for feature selection, model evaluation, and stability analysis. The aim is to identify stable and high-performing features across partitioned and balanced datasets using evolutionary search techniques.

---

## 📦 1. Genetic Feature Selection

Feature selection is performed using the `sklearn-genetic` library applied on balanced partitions of the dataset.

📄 **Script**:  
- `ga_feature_selection.py`

⚙️ **Process**:
- The original dataset is partitioned such that the non-druggable class is split into 8 equal parts, balanced against the druggable class.
- Each partition is used to run GA-based feature selection with the following configuration:

```python
GAFeatureSelectionCV(
    estimator=clf,
    cv=3,
    scoring="accuracy",
    population_size=20,
    generations=5,
    n_jobs=-1,
    verbose=True,
    keep_top_k=2,
    elitism=True
)
```

📤 **Output**:  
- Binary masks of selected features (`True/False`) for each partition, indicating selected features for model training.

---

## 📊 2. Feature Scores Evaluation

We train custom models—Partition-Level Classifier (PLC) and Partition-Leave-One Out Classifier (PLEOC)—on the GA-selected features for each partition.

📄 **Script**:  
- `ga_feature_scores.py`

📁 **Results**:
- `Results/Features Scores.csv`: Contains performance scores (accuracy, F1, etc.) of models trained on GA-selected features across partitions.

---

## 📈 3. Fitness Score Progression Analysis

This module analyzes how the average fitness score evolves across generations during the GA search, as well as its variance.

📄 **Script**:  
- `ga_fitness_scores_&_valuation.py`

📁 **Results**:
- `Results/Mean (Average) Fitness Scores Across Generations.csv`: Mean fitness score across generations for each partition.
- `Results/Variance of Fitness Scores Across Generations.csv`: Variance of fitness scores to measure stability across generations.

---

## 📌 4. Feature Stability with SHAP Rankings

To evaluate the consistency of feature selection, we compare GA-selected features with the top 15 features ranked using SHAP values.

📄 **Script**:  
- `ga_feature_stability_with_shap_rankings.py`

📁 **Results**:
- `Results/Feature Stability with SHAP Rankings.csv`: Overlap statistics showing how many features selected by GA match top SHAP-ranked features per partition.

---

## 📌 Notes

- The GA setup ensures both **performance optimization** and **feature robustness** across varied partitions.
- SHAP-based stability check ensures that the selected features are not only optimal per partition but also **consistent with interpretable feature importance** metrics.
