# Mount google drive at /content/drive
from google.colab import drive
drive.mount('/content/drive')

# Setting seeds for reproducibility
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import ADASYN, SMOTE
import pandas as pd
import random
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


MODEL_UNDER_TEST = "xgb" 
FEATURE_GROUP = "all" 
'''
MODEL_UNDER_TEST can be one of the following:
- "xgb" for XGBoost Classifier
- "rf" for Random Forest Classifier
- "lr" for Logistic Regression
- "svm_linear" for Support Vector Machine with linear kernel
- "svm_rbf" for Support Vector Machine with RBF kernel
- "knn" for K-Nearest Neighbors
- "nb" for Naive Bayes
FEATURE_GROUP can be one of the following:
- "seq_only" for sequence-only features
- "non_seq_only" for non-sequence features
- "all" for all features
- "esm_instead_of_latent" for ESM2 embeddings instead of latent values and remaining features same as "all"
'''

data_file_path = "protein_props.json"
druggable_proteins_file_path = "druggable_proteins.txt"
investigational_proteins_file_path = "investigational_proteins.txt"

with open(data_file_path, 'r') as f:
    protein_data = json.load(f)

print("Total number of uniprot human verified proteins:", len(protein_data))

# Extracting list of druggable and approved druggable proteins
with open(druggable_proteins_file_path, 'r') as f:
    approved_druggable_proteins = f.read().splitlines()

with open(investigational_proteins_file_path, 'r') as f:
    investigational_proteins = f.read().splitlines()

druggable_proteins = approved_druggable_proteins + investigational_proteins

print("Number of druggable proteins:", len(druggable_proteins))
print("Number of druggable approved proteins:", len(approved_druggable_proteins))


# Fetching feature data for all proteins
properties = (pd.read_json("protein_props.json")).transpose()
is_druggable = [1 if i in druggable_proteins else 0 for i in properties.index]
is_approved_druggable = [1 if i in approved_druggable_proteins else 0 for i in properties.index]

properties["is_druggable"] = is_druggable
properties["is_approved_druggable"] = is_approved_druggable

PCP_properties = properties.copy()
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
amino_acid_percent = {i:[] for i in amino_acids}
for i in PCP_properties['Amino Acid Percent']:
  for aa in amino_acids:
    amino_acid_percent[aa].append(i[aa])
for aa in amino_acids:
  PCP_properties = pd.concat([PCP_properties, pd.Series(amino_acid_percent[aa], index = PCP_properties.index, name = f"Amino Acid Percent {aa}")], axis = 1)

PCP_properties[f"Molar Extinction Coefficient 1"] = pd.Series([x[0] for x in PCP_properties['Molar Extinction Coefficient']], index = PCP_properties.index)
PCP_properties[f"Molar Extinction Coefficient 2"] = pd.Series([x[1] for x in PCP_properties['Molar Extinction Coefficient']], index = PCP_properties.index)

PCP_properties[f"Secondary Structure helix"] = pd.Series([x[0] for x in PCP_properties['Secondary Structure']], index = PCP_properties.index)
PCP_properties[f"Secondary Structure turn"] = pd.Series([x[1] for x in PCP_properties['Secondary Structure']], index = PCP_properties.index)
PCP_properties[f"Secondary Structure sheet"] = pd.Series([x[2] for x in PCP_properties['Secondary Structure']], index = PCP_properties.index)

PCP_properties.drop(columns = ['Amino Acid Count','Amino Acid Percent',"Molar Extinction Coefficient","Flexibility","Secondary Structure",'Sequence'], inplace = True)
PCP_properties['Sequence Length'] = PCP_properties['Sequence Length'].astype(int)
PCP_properties[['Molecular Weight', 'GRAVY', 'Isoelectric Point', 'Instability Index', 'Aromaticity', 'Charge at 7']] = PCP_properties[['Molecular Weight', 'GRAVY', 'Isoelectric Point', 'Instability Index', 'Aromaticity', 'Charge at 7']].astype(float)

with open("features/gdpc_encodings.json", 'r') as file:
    data = json.load(file)
gpdc_encodings = pd.DataFrame(data).transpose()

ppi = pd.read_json("features/ppi.json").transpose()
ppi_network = pd.read_csv("features/ppi_network_properties.csv")
ppi_network.index = ppi_network['Unnamed: 0']
ppi_network.drop(columns = ['Unnamed: 0'], inplace = True)
ppi = pd.concat([ppi, ppi_network], axis = 1)

glycolisation = pd.read_csv("features/glycosylation.csv")
glycolisation.index = glycolisation['Unnamed: 0']
glycolisation.drop(columns = ['Unnamed: 0'], inplace = True)
ptm = pd.read_csv("features/PTM_counts.csv")
ptm.index = ptm["Unnamed: 0"]
ptm.drop(columns = ['Unnamed: 0'], inplace = True)
ptm_counts = pd.concat([ptm, glycolisation], axis = 1)

with open("features/subcellular_locations2.json", 'r') as file:
    data = json.load(file)
unique_groups = set()
for entry in data.values():
    if "general" in entry:
        for general_entry in entry["general"]:
            if "group" in general_entry: unique_groups.add(general_entry["group"])

unique_groups_list = list(unique_groups)

rows = []
for protein_id in PCP_properties.index:
    row = {group: 0 for group in unique_groups_list}
    if protein_id in data:
        for entry in data[protein_id].get("general", []):
            if "group" in entry and entry["group"] in unique_groups:
                row[entry["group"]] = 1
    row["protein_id"] = protein_id
    rows.append(row)

subcellular_data = pd.DataFrame(rows).set_index("protein_id")

domains = pd.read_csv("features/data_top20_updated.csv")
domains.index = domains['Unnamed: 0']
domains.drop(columns = ['Unnamed: 0'], inplace = True)

flexibility = pd.read_csv("features/flexibility_properties.csv")
flexibility.index = flexibility['Unnamed: 0']
flexibility.drop(columns = ['Unnamed: 0'], inplace = True)

latent_data = pd.read_csv("features/latent_values.csv").transpose()
latent_data.columns = [f"Latent_Value_{i+1}" for i in latent_data.columns]

if FEATURE_GROUP == "all":
   final_data = pd.concat([PCP_properties,gpdc_encodings, ptm_counts, ppi, subcellular_data, domains, flexibility, latent_data], axis = 1).dropna()
elif FEATURE_GROUP == "seq_only":
   final_data = pd.concat([PCP_properties, gpdc_encodings, flexibility, latent_data], axis=1).dropna()
elif FEATURE_GROUP == "non_seq_only":
   final_data = pd.concat([ptm_counts, ppi, subcellular_data, domains], axis=1).dropna()
elif FEATURE_GROUP == "esm_instead_of_latent":
    embeddings_file_path = "features/protein_embeddings_ESM2.csv"
    embeddings = pd.read_csv(embeddings_file_path, index_col=0)
    embeddings.columns = [f"Embedding_{int(i)+1}" for i in embeddings.columns]
    final_data = pd.concat([PCP_properties, gpdc_encodings, ptm_counts, ppi, subcellular_data, domains, flexibility, embeddings], axis=1).dropna()
else:
    raise ValueError("Invalid FEATURE_GROUP. Choose 'all', 'seq_only', or 'non_seq_only'.")

if 'is_druggable' not in final_data.columns:
  final_data['is_druggable'] = is_druggable
if 'is_approved_druggable' not in final_data.columns:
  final_data['is_approved_druggable'] = is_approved_druggable

features_list = final_data.columns
features_list = features_list.drop(['is_druggable','is_approved_druggable'])
features_list = list(features_list)
print(features_list)
print(len(features_list))

# Train Test Splitting
def get_data(x_sample, y_sample):
  return np.array(x_sample), np.array(y_sample)

def data_splitting(x_sample, y_sample, mode="default", scaler="none", class_size=300, random_state=123):
  druggable_indices = (y_sample == 1)  # Assuming 1 represents druggable
  non_druggable_indices = (y_sample == 0)  # Assuming 0 represents non-druggable

  druggable_X = x_sample[druggable_indices]
  druggable_y = y_sample[druggable_indices]

  non_druggable_X = x_sample[non_druggable_indices]
  non_druggable_y = y_sample[non_druggable_indices]

  druggable_X_remaining, druggable_X_test, druggable_y_remaining, druggable_y_test = train_test_split(druggable_X, druggable_y, test_size=class_size, random_state=random_state)
  non_druggable_X_remaining, non_druggable_X_test, non_druggable_y_remaining, non_druggable_y_test = train_test_split(non_druggable_X, non_druggable_y, test_size= class_size, random_state=random_state)

  X_test = pd.concat((druggable_X_test, non_druggable_X_test))
  y_test = pd.concat((druggable_y_test, non_druggable_y_test))
  X_train = pd.concat((druggable_X_remaining, non_druggable_X_remaining))
  y_train = pd.concat((druggable_y_remaining, non_druggable_y_remaining))
  X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
  if mode == "default":
    pass
  elif mode == "adasyn":
    ada = ADASYN(random_state=random_state)
    X_train, y_train = ada.fit_resample(X_train, y_train)
  elif mode == "smote":
    smt = SMOTE(random_state=random_state)
    X_train, y_train = smt.fit_resample(X_train, y_train)
  elif mode == "undersampling":
    undersampled_non_druggable_X_remaining = non_druggable_X_remaining.sample(n=len(druggable_X_remaining))
    X_train = pd.concat((druggable_X_remaining, undersampled_non_druggable_X_remaining))
    y_train = pd.concat((druggable_y_remaining, pd.Series([0] * len(undersampled_non_druggable_X_remaining), index=undersampled_non_druggable_X_remaining.index)))
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

  if scaler == "std":
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
  elif scaler == "minmax":
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
  elif scaler == "none":
    pass

  return X_train, X_test, y_train, y_test

# rem-new-data is to extract only those proteins which are either approved druggable or non-druggable
# i.e., it excludes proteins which are non-approved but druggable
new_data = final_data.copy()
new_data['new_column'] = new_data['is_druggable'] + new_data['is_approved_druggable']
rem_new_data = new_data[new_data['new_column'] != 1]
print(rem_new_data.shape, np.bincount(rem_new_data['new_column']))

def get_model(model_name):
  if model_name == "xgb": return xgb.XGBClassifier(objective='binary:logistic', random_state=42)
  if model_name == "rf": return RandomForestClassifier(random_state=27)
  if model_name == "lr": return LogisticRegression(random_state=42)
  if model_name == "svm_linear": return SVC(random_state=42, kernel='linear', max_iter=1000, probability=True)
  if model_name == "svm_rbf": return SVC(random_state=42, kernel='rbf', max_iter=1000, probability=True)
  if model_name == "knn": return KNeighborsClassifier()
  if model_name == "nb": return GaussianNB()

def complete_evaluate(random_state):
  X_train, X_test, y_train, y_test = data_splitting(rem_new_data[features_list], rem_new_data['is_druggable'], random_state=random_state)
  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
  print(np.bincount(y_train), np.bincount(y_test))
  X_train_druggable = X_train[y_train == 1]
  X_train_non_druggable = X_train[y_train == 0]

  X_train_non_druggable_partitions = np.array_split(X_train_non_druggable, round(len(X_train_non_druggable)/len(X_train_druggable)))
  print(f"Splitting into {len(X_train_non_druggable_partitions)} partitions")
  print("Sizes of partitions")
  for i, partition in enumerate(X_train_non_druggable_partitions):
    print(f"Partition {i}: {len(partition)}")
  models = []
  for partition in X_train_non_druggable_partitions:
    X_combined = np.concatenate((X_train_druggable, partition))
    y_combined = np.concatenate((np.ones(len(X_train_druggable)), np.zeros(len(partition))))
    model = get_model(MODEL_UNDER_TEST)
    model.fit(X_combined, y_combined)
    models.append(model)

  y_preds = []
  for model in models:
    y_pred = model.predict(X_test)
    y_preds.append(y_pred)

  majority_preds = np.mean(y_preds, axis=0)
  majority_preds = np.round(majority_preds)
  print(majority_preds.shape)

  y_pred_probas = []
  for model in models:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_probas.append(y_pred_proba)

  y_pred_probas = np.array(y_pred_probas)
  mean_pred_probas = np.mean(y_pred_probas, axis=0)
  average_proba_preds = np.round(mean_pred_probas)
  print(y_pred_probas.shape, mean_pred_probas.shape, average_proba_preds.shape)
  accuracy_metrics = {}
  for i, y_pred in enumerate(y_preds):
    accuracy_metrics[f"partition_{i}"]={
        "accuracy_total": accuracy_score(y_test, y_pred),
        "accuracy_druggable": accuracy_score(y_test[y_test == 1], y_pred[y_test == 1]),
        "accuracy_non_druggable": accuracy_score(y_test[y_test == 0], y_pred[y_test == 0]),
    }
    accuracy_metrics["majority_prediction"]={
        "accuracy_total": accuracy_score(y_test, majority_preds),
        "accuracy_druggable": accuracy_score(y_test[y_test == 1], majority_preds[y_test == 1]),
        "accuracy_non_druggable": accuracy_score(y_test[y_test == 0], majority_preds[y_test == 0]),
    }
    accuracy_metrics["average_probability_prediction"] = {
        "accuracy_total": accuracy_score(y_test, average_proba_preds),
        "accuracy_druggable": accuracy_score(y_test[y_test == 1], average_proba_preds[y_test == 1]),
        "accuracy_non_druggable": accuracy_score(y_test[y_test == 0], average_proba_preds[y_test == 0]),
    }


  df = pd.DataFrame(accuracy_metrics).transpose()
  return df

# randomly sample 20 from range(100)
scores_df = []
random.seed(42)
random_states = random.sample(range(100), 20)
for random_state in random_states:
  scores_df.append(complete_evaluate(random_state))
  print(f"Completed for random state {random_state}")


# Mean scores
avg_scores = sum(scores_df)/20

df_accuracy_totals = [scores_df[i]["accuracy_total"] for i in range(20)]
df_accuracy_druggables = [scores_df[i]["accuracy_druggable"] for i in range(20)]
df_accuracy_non_druggables = [scores_df[i]["accuracy_non_druggable"] for i in range(20)]
df_accuracy_totals = np.array(df_accuracy_totals)
df_accuracy_druggables = np.array(df_accuracy_druggables)
df_accuracy_non_druggables = np.array(df_accuracy_non_druggables)

# store the np.std values in the same as avg_scores
std_scores = avg_scores.copy()
std_scores["accuracy_total"] = np.std(df_accuracy_totals, axis=0)
std_scores["accuracy_druggable"] = np.std(df_accuracy_druggables, axis=0)
std_scores["accuracy_non_druggable"] = np.std(df_accuracy_non_druggables, axis=0)
std_scores

# Save files
avg_scores.to_csv("PEC_avg_scores.csv")
std_scores.to_csv("PEC_std_scores.csv")