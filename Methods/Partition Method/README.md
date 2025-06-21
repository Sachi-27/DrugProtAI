# 🧠 Partition Method & Ablation Studies

This folder contains code and results related to our custom **Partition Ensemble Classifier (PEC)** approach for druggability prediction using various machine learning algorithms and feature sets. The implementation and experiments are primarily documented in the notebook:

> 📓 `partition_method_ablation_study.ipynb`

---

## ⚙️ Model Selection Ablation Study

We evaluate different machine learning algorithms under a consistent PEC setup. You can specify the model under test via the `MODEL_UNDER_TEST` variable, with the following options:

- `"xgb"` – XGBoost Classifier
- `"rf"` – Random Forest Classifier
- `"lr"` – Logistic Regression
- `"svm_linear"` – Support Vector Machine (Linear Kernel)
- `"svm_rbf"` – Support Vector Machine (RBF Kernel)
- `"knn"` – K-Nearest Neighbors
- `"nb"` – Gaussian Naive Bayes

We maintain a consistent evaluation setup with a **held-out test set of 600 proteins** (300 druggable, 300 non-druggable).

### 📊 Summary of Performance Across Models

| Model                 | Total Accuracy | Druggable Accuracy (Sensitivity) | Non-Druggable Accuracy (Specificity) |
|----------------------|----------------|----------------------------------|--------------------------------------|
| **XGBoost**          | 78.06 ± 2.03   | 78.73 ± 2.82                     | 77.38 ± 2.43                         |
| **Random Forest**    | 75.94 ± 1.55   | 77.73 ± 1.95                     | 74.15 ± 2.36                         |
| Logistic Regression  | 63.34 ± 2.07   | 65.53 ± 3.27                     | 61.15 ± 2.12                         |
| SVM (Linear Kernel)  | 50.19 ± 0.74   | 91.53 ± 20.75                    | 8.85 ± 20.82                         |
| SVM (RBF Kernel)     | 49.74 ± 0.65   | 94.28 ± 12.78                    | 5.20 ± 12.43                         |
| K-Nearest Neighbors  | 58.58 ± 1.63   | 64.28 ± 2.43                     | 52.87 ± 2.78                         |
| Naive Bayes          | 57.30 ± 1.78   | 25.50 ± 4.34                     | 89.10 ± 2.15                         |

> 🔍 **Conclusion**: XGBoost and Random Forest consistently outperform other models across key metrics.

---

## 🧬 Feature Group Comparisons

We compare the effect of using different types of features, specified via `FEATURE_GROUP`:

- `"seq_only"` – Sequence-derived features:
  - Physicochemical properties
  - GDPC encodings
  - Flexibility properties
  - Latent neural network-derived features
- `"non_seq_only"` – Non-sequence features:
  - PTM and glycosylation counts
  - PPI-based features (pairwise & network)
  - Subcellular localization
  - Protein domain information
- `"all"` – Combined feature set
- `"esm_instead_of_latent"` – Combined feature set except that latent values replaced with ESM2 embeddings 

### 📊 Performance of XGBoost PEC on Different Feature Groups

| Feature Group                                      | Total Accuracy | Sensitivity  | Specificity  |
| -------------------------------------------------- | -------------- | ------------ | ------------ |
| All Features                                       | 78.06 ± 2.03   | 78.73 ± 2.82 | 77.38 ± 2.43 |
| Seq-only                                           | 71.71 ± 1.47   | 74.25 ± 2.15 | 69.17 ± 2.50 |
| Non-seq only                                       | 75.14 ± 1.38   | 76.15 ± 2.57 | 74.13 ± 1.75 |
| All Features but ESM2 embeddings instead of Latent | 82.78 ± 1.55   | 83.95 ± 1.98 | 81.60 ± 2.47 |

### 📊 Performance of Random Forest PEC on Different Feature Groups

| Feature Group                                      | Total Accuracy | Sensitivity  | Specificity  |
| -------------------------------------------------- | -------------- | ------------ | ------------ |
| All Features                                       | 75.94 ± 1.55   | 77.73 ± 1.95 | 74.15 ± 2.36 |
| Seq-only                                           | 70.43 ± 2.19   | 71.73 ± 2.76 | 69.13 ± 2.95 |
| Non-seq only                                       | 74.33 ± 1.60   | 77.65 ± 2.80 | 71.02 ± 2.13 |
| All Features but ESM2 embeddings instead of Latent | 77.80 ± 1.89   | 79.30 ± 2.36 | 76.30 ± 2.63 |

> ✅ **Observation**: Non-sequence features significantly contribute to predictive performance and should not be excluded.

---

## 🧬 Partition Method on ESM2 Embeddings

We also extend the PEC framework to test ESM2 protein embeddings using XGBoost, under different preprocessing choices:

- `none`: Raw ESM2 embeddings (no scaling)
- `standard`: StandardScaler normalization
- `minmax`: MinMaxScaler normalization

### 📊 XGBoost PEC on ESM2 Embeddings

| Preprocessing      | Total Accuracy | Sensitivity | Specificity |
|--------------------|----------------|-------------|-------------|
| No preprocessing   | 80.99 ± 1.95   | 82.37 ± 2.61 | 79.62 ± 2.36 |
| Standard Scaling   | 81.47 ± 1.42   | 82.78 ± 2.03 | 80.15 ± 1.64 |
| MinMax Scaling     | 81.27 ± 1.60   | 82.62 ± 2.24 | 79.92 ± 2.52 |

> 📈 **Insight**: Scaling has a marginal positive effect on XGBoost performance when using high-dimensional ESM2 embeddings.

---


