import pandas as pd 
import numpy as np
import random
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

### Setting seeds for reproducibility
random.seed(42)
np.random.seed(42)


# File paths 
druggable_proteins_file_path = "druggable_proteins.txt"
investigational_proteins_file_path = "investigational_proteins.txt"
embeddings_file_path = "protein_embeddings_ESM2.csv" # To change to the path of the embeddings file
PEC_MODEL = "xgb" # Change to "rf" for Random Forest Classifier, "svc" for 
SCALER = "standard" # Change to "standard" for StandardScaler, "minmax" for MinMaxScaler

# Extract embeddings and protein ids of druggable, non druggable and investigational categories
embeddings = pd.read_csv(embeddings_file_path, index_col=0)
protein_ids = list(embeddings.index)
print("Number of protein ids:", len(protein_ids))

with open(druggable_proteins_file_path, 'r') as f:
    druggable_proteins = f.read().splitlines()
with open(investigational_proteins_file_path, 'r') as f:
    investigational_proteins = f.read().splitlines()

druggable_proteins = list(set(druggable_proteins) & set(protein_ids))
investigational_proteins = list(set(investigational_proteins) & set(protein_ids))
non_druggable_proteins = list(set(protein_ids) - set(druggable_proteins) - set(investigational_proteins))

print("Number of druggable proteins:", len(druggable_proteins))
print("Number of investigational proteins:", len(investigational_proteins))
print("Number of non-druggable proteins:", len(non_druggable_proteins))

### Extracting embeddings for druggable and non-druggable proteins
X_druggable = embeddings.loc[druggable_proteins]
X_non_druggable = embeddings.loc[non_druggable_proteins]
X_investigational = embeddings.loc[investigational_proteins]

def get_model(method):
    if method == "xgb": return xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    elif method == "rf": return RandomForestClassifier(random_state=27)
    else: raise ValueError("Invalid method")

# Performing normalization on training data and test data
X_combined = pd.concat([X_druggable, X_non_druggable])
if SCALER == "standard":
    scaler = StandardScaler()
elif SCALER == "minmax":
    scaler = MinMaxScaler()
elif SCALER == "none":
    pass
else:
    raise ValueError("Invalid scaler type. Choose 'standard', 'minmax', or 'none'.")

if SCALER != "none":
    scaler.fit(X_combined)
    X_druggable = pd.DataFrame(scaler.transform(X_druggable), columns=X_druggable.columns, index=X_druggable.index)
    X_non_druggable = pd.DataFrame(scaler.transform(X_non_druggable), columns=X_non_druggable.columns, index=X_non_druggable.index)
    X_investigational = pd.DataFrame(scaler.transform(X_investigational), columns=X_investigational.columns, index=X_investigational.index)

# Divide the non_druggable (majority) class into K partitions
K = round(len(X_non_druggable) / len(X_druggable))
X_non_druggable_partitions = np.array_split(X_non_druggable, K)

# Train K models of type PEC_MODEL
models = []
print(f"Splitting non-druggable set into {K} partitions:")
for partition_num, partition in enumerate(X_non_druggable_partitions):
    print(f"Training on partition {partition_num + 1}/{K}")
    X_train = pd.concat([X_druggable, partition])
    y_train = [1] * len(X_druggable) + [0] * len(partition)
    model = get_model(PEC_MODEL)
    model.fit(X_train, y_train)
    models.append(model)

### DI calculation on investigational set
investigational_preds = []
for model in models:
    investigational_preds.append(model.predict_proba(X_investigational)[:, 1])

investigational_preds = np.array(investigational_preds)
investigational_preds_mean = np.mean(investigational_preds, axis=0)

# save in csv protein ids and DI scores
investigational_df = pd.DataFrame({
    "protein_id": X_investigational.index,
    "DI_score": investigational_preds_mean
})

investigational_df.to_csv("results/ESM2xgbstd_DI_scores_investigational.csv", index=False)

### DI calculation on non-druggable set
non_druggable_preds = []
non_druggable_proteins_one_by_one = []

assert(len(X_non_druggable_partitions) == len(models)) # Sanity check
for partition_num, partition in enumerate(X_non_druggable_partitions):
    print(f"Calculating DI for partition {partition_num + 1}/{K}")
    non_druggable_preds_partition = []
    for model_num, model in enumerate(models):
        if model_num == partition_num: continue
        non_druggable_preds_partition.append(model.predict_proba(partition)[:, 1])
    non_druggable_preds_partition = np.array(non_druggable_preds_partition)
    non_druggable_preds_partition_mean = np.mean(non_druggable_preds_partition, axis=0)
    non_druggable_preds.extend(non_druggable_preds_partition_mean)
    non_druggable_proteins_one_by_one.extend(partition.index)

# save in csv protein ids and DI scores
non_druggable_df = pd.DataFrame({
    "protein_id": non_druggable_proteins_one_by_one,
    "DI_score": non_druggable_preds
})
non_druggable_df.to_csv("results/ESM2xgbstd_DI_scores_non_druggable.csv", index=False)

