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
output_file_path = "results/ESM2_xgb_minmax.txt"
PEC_MODEL = "xgb" # Change to "rf" for Random Forest Classifier, "svc" for 
SCALER = "minmax" # Change to "standard" for StandardScaler, "minmax" for MinMaxScaler

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

def get_model(method):
    if method == "xgb": return xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    elif method == "rf": return RandomForestClassifier(random_state=27)
    else: raise ValueError("Invalid method")

def evaluate_model(random_seed):
    # Separating a small test set of size 300 from each class
    TEST_SIZE = 300
    X_test_druggable = X_druggable.sample(TEST_SIZE, random_state=random_seed)
    X_test_non_druggable = X_non_druggable.sample(TEST_SIZE, random_state=random_seed)
    X_test = pd.concat([X_test_druggable, X_test_non_druggable])
    y_test = [1] * len(X_test_druggable) + [0] * len(X_test_non_druggable)
    y_test = np.array(y_test)

    # Remaining data for training
    X_train_druggable = X_druggable.drop(X_test_druggable.index)
    X_train_non_druggable = X_non_druggable.drop(X_test_non_druggable.index)

    # Performing normalization on training data and test data
    X_combined = pd.concat([X_train_druggable, X_train_non_druggable])
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
        X_train_druggable = pd.DataFrame(scaler.transform(X_train_druggable), columns=X_train_druggable.columns, index=X_train_druggable.index)
        X_train_non_druggable = pd.DataFrame(scaler.transform(X_train_non_druggable), columns=X_train_non_druggable.columns, index=X_train_non_druggable.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Divide the non_druggable (majority) class into K partitions
    K = round(len(X_train_non_druggable) / len(X_train_druggable))
    X_train_non_druggable_partitions = np.array_split(X_train_non_druggable, K)

    # Train K models of type PEC_MODEL
    models = []
    for partition_num, partition in enumerate(X_train_non_druggable_partitions):
        print(f"Training on partition {partition_num + 1}/{K}")
        X_train = pd.concat([X_train_druggable, partition])
        y_train = [1] * len(X_train_druggable) + [0] * len(partition)
        model = get_model(PEC_MODEL)
        model.fit(X_train, y_train)
        models.append(model)
    
    # Prediction on the test set
    test_preds = []

    for model in models:
        pred_proba = model.predict_proba(X_test)[:, 1]
        test_preds.append(pred_proba)
    test_preds = np.mean(np.array(test_preds), axis=0)
    test_preds = np.round(test_preds).astype(int)

    accuracy = np.mean(test_preds == y_test)
    sensitivity = np.sum((test_preds == 1) & (y_test == 1)) / np.sum(y_test == 1)
    specificity = np.sum((test_preds == 0) & (y_test == 0)) / np.sum(y_test == 0)

    return accuracy, sensitivity, specificity
    
random.seed(42)
random_seeds = random.sample(range(100), 20)
results = []
for seed in random_seeds:
    accuracy, sensitivity, specificity = evaluate_model(seed)
    with open(output_file_path, 'a') as f:
        f.write(f"Random Seed: {seed}, Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}\n")
    results.append([accuracy, sensitivity, specificity])
    print(f"Random Seed: {seed}, Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

# Calculate and print the average results
avg_results = np.mean(np.array(results), axis=0)
with open(output_file_path, 'a') as f:
    f.write(f"Average Results: Accuracy: {avg_results[0]:.4f}, Sensitivity: {avg_results[1]:.4f}, Specificity: {avg_results[2]:.4f}\n")
print(f"Average Results: Accuracy: {avg_results[0]:.4f}, Sensitivity: {avg_results[1]:.4f}, Specificity: {avg_results[2]:.4f}")
