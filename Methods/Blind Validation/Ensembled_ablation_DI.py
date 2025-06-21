# Setting seeds for reproducibility
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from imblearn.over_sampling import ADASYN, SMOTE
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


PEC_MODEL = "xgb" 
FEATURE_GROUP = "all" 
'''
PEC_MODEL can be one of the following:
- "xgb" for XGBoost Classifier
- "rf" for Random Forest Classifier
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
   final_data = pd.concat([PCP_properties, gpdc_encodings, ptm_counts, ppi, subcellular_data, domains, flexibility, latent_data], axis=1).dropna()
elif FEATURE_GROUP == "seq_only":
    final_data = pd.concat([PCP_properties, gpdc_encodings, flexibility, latent_data], axis=1).dropna()
elif FEATURE_GROUP == "non_seq_only":
    final_data = pd.concat([ptm_counts, ppi, subcellular_data, domains], axis=1).dropna()
elif FEATURE_GROUP == "esm_instead_of_latent":
    embeddings_file_path = "protein_embeddings_ESM2.csv"
    embeddings = pd.read_csv(embeddings_file_path, index_col=0)
    embeddings.columns = [f"Embedding_{int(i)+1}" for i in embeddings.columns]
    final_data = pd.concat([PCP_properties, gpdc_encodings, ptm_counts, ppi, subcellular_data, domains, flexibility, embeddings], axis=1).dropna()
else:
    raise ValueError("Invalid FEATURE_GROUP. Choose 'all', 'seq_only', 'non_seq_only', or 'esm_instead_of_latent'.")

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

# rem-new-data is to extract only those proteins which are either approved druggable or non-druggable
# i.e., it excludes proteins which are non-approved but druggable
new_data = final_data.copy()
new_data['new_column'] = new_data['is_druggable'] + new_data['is_approved_druggable']
print(np.bincount(new_data['new_column']))
rem_new_data = new_data[new_data['new_column'] != 1]
rem_new_data.shape, np.bincount(rem_new_data['new_column'])

X, y = rem_new_data[features_list], np.array(rem_new_data["is_approved_druggable"])
X.shape, y.shape

X_druggable = X[y == 1]
X_non_druggable = X[y == 0]

X_non_druggable_partitions = np.array_split(X_non_druggable, round(len(X_non_druggable)/len(X_druggable)))
print(f"Splitting into {len(X_non_druggable_partitions)} partitions")
print("Sizes of partitions")
for i, partition in enumerate(X_non_druggable_partitions):
  print(f"Partition {i}: {len(partition)}")

def get_model(model_type):
    if model_type == "xgb": return xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    if model_type == "rf": return RandomForestClassifier(random_state=27)
    else: raise ValueError("Invalid model type. Choose 'xgb' or 'rf'.")


models = []
for partition in X_non_druggable_partitions:
  X_combined = np.concatenate((np.array(X_druggable), np.array(partition)))
  y_combined = np.concatenate((np.ones(len(X_druggable)), np.zeros(len(partition))))
  pec_model = get_model(PEC_MODEL)
  pec_model.fit(X_combined, y_combined)
  models.append(pec_model)


### DI for investigational proteins
non_approved_druggable = new_data[new_data["new_column"] == 1]
X_test, y_test = get_data(non_approved_druggable[features_list], non_approved_druggable["is_approved_druggable"])
protein_names = non_approved_druggable.index
predictions = []
probabilities = []
for model in models:
  predictions.append(model.predict(X_test))
  probabilities.append(model.predict_proba(X_test)[:,1])

predictions = np.array(predictions)
probabilities = np.array(probabilities)
predictions = np.mean(predictions, axis=0)
predictions = np.round(predictions)
mean_probabilities = np.mean(probabilities, axis=0)
data = {
    "Protein": protein_names
}
for i,probs in enumerate(probabilities):
  data[f"Probability_Partition_{i+1}"] = probs
data["Mean_Probability"] = mean_probabilities
data["Majority_Prediction"] = predictions

df = pd.DataFrame(data)
df.set_index("Protein", inplace=True)
df.to_csv(f"results/{PEC_MODEL}_{FEATURE_GROUP}_DI_investigational.csv")

### DI for non druggable proteins
protein_names_all, predictions_all, probabilities_all = [], None, None
for i, partition in enumerate(X_non_druggable_partitions):
  protein_names = partition.index
  predictions, probabilities = [], []
  for j, model in enumerate(models):
    if j != i:
      predictions.append(model.predict(np.array(partition)))
      probabilities.append(model.predict_proba(np.array(partition))[:,1])
  predictions, probabilities = np.array(predictions), np.array(probabilities)
  predictions = np.mean(predictions, axis=0)
  predictions = np.round(predictions)
  print(predictions.shape, probabilities.shape)

  protein_names_all.extend(protein_names)
  if predictions_all is None:
    predictions_all = predictions
    probabilities_all = probabilities
  else:
    predictions_all = np.concatenate((predictions_all, predictions))
    probabilities_all = np.concatenate((probabilities_all, probabilities), axis=1)

  
predictions_all = np.array(predictions_all)
probabilities_all = np.array(probabilities_all)
mean_probabilities_all = np.mean(probabilities_all, axis=0)

data = {
    "Protein": protein_names_all
}
for i,probs in enumerate(probabilities_all):
  data[f"Probability_Partition_{i+1}"] = probs
data["Mean_Probability"] = mean_probabilities_all
data["Majority_Prediction"] = predictions_all

df = pd.DataFrame(data)
df.set_index("Protein", inplace=True)

df.to_csv(f"results/{PEC_MODEL}_{FEATURE_GROUP}_DI_non_druggable.csv")

### DI for druggable proteins
predictions_druggable, probabilities_druggable = [], []
for model in models:
    predictions_druggable.append(model.predict(X_druggable))
    probabilities_druggable.append(model.predict_proba(X_druggable)[:, 1])

predictions_druggable = np.array(predictions_druggable)
probabilities_druggable = np.array(probabilities_druggable)
mean_probabilities_druggable = np.mean(probabilities_druggable, axis=0)
data = {
    "Protein": X_druggable.index
}
for i, probs in enumerate(probabilities_druggable):
    data[f"Probability_Partition_{i+1}"] = probs
data["Mean_Probability"] = mean_probabilities_druggable
data["Majority_Prediction"] = np.round(np.mean(predictions_druggable, axis=0))
df = pd.DataFrame(data)
df.set_index("Protein", inplace=True)
df.to_csv(f"results/{PEC_MODEL}_{FEATURE_GROUP}_DI_druggable.csv")

print("DI scores for investigational, non-druggable, and druggable proteins have been saved in respective CSV files.")