# Setting seeds for reproducibility
import numpy as np
import json
import pandas as pd
import random
import xgboost as xgb

random.seed(42)
np.random.seed(42)

### EXTRACTING DATA ###
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
final_data = pd.concat([PCP_properties,gpdc_encodings, ptm_counts, ppi, subcellular_data, domains, flexibility, latent_data], axis = 1).dropna()
features_list = final_data.columns
features_list = features_list.drop(['is_druggable','is_approved_druggable'])
features_list = list(features_list)
print(features_list)
print(len(features_list))


# rem-new-data is to extract only those proteins which are either approved druggable or non-druggable
# i.e., it excludes proteins which are non-approved but druggable
new_data = final_data.copy()
new_data['new_column'] = new_data['is_druggable'] + new_data['is_approved_druggable']
print(np.bincount(new_data['new_column']))
rem_new_data = new_data[new_data['new_column'] != 1]


file_to_write = "feature_selection_results.txt"
'''
FEATURE SELECTION METHOD 1
We train xgb partition models and then use these models to get feature importance scores.
We then average the feature importance scores across all models to get the final feature importance scores.
We then remove the feature with the least importance score.
We then run it again and again until we are left with 1 feature.
'''
while len(features_list) > 0:
    # Getting the data
    X, y = rem_new_data[features_list], np.array(rem_new_data["is_approved_druggable"])
    print(X.shape, y.shape)
    print(np.bincount(y))

    # Splitting the data into druggable and non-druggable
    X_druggable = X[y == 1]
    X_non_druggable = X[y == 0]

    # Splitting the non-druggable data into partitions
    # We will use these partitions to train xgb models
    X_non_druggable_partitions = np.array_split(X_non_druggable, round(len(X_non_druggable)/len(X_druggable)))
    print(f"Splitting into {len(X_non_druggable_partitions)} partitions")
    print("Sizes of partitions")
    for i, partition in enumerate(X_non_druggable_partitions):
        print(f"Partition {i}: {len(partition)}")

    xgb_models = []
    for partition in X_non_druggable_partitions:
        X_combined = np.concatenate((np.array(X_druggable), np.array(partition)))
        y_combined = np.concatenate((np.ones(len(X_druggable)), np.zeros(len(partition))))
        xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
        xgb_model.fit(X_combined, y_combined)
        xgb_models.append(xgb_model)

    # Get feature importance scores
    feature_importance_scores = np.zeros(X.shape[1])
    for model in xgb_models:
        feature_importance_scores += model.feature_importances_
    feature_importance_scores /= len(xgb_models)

    # Sort features by importance
    sorted_indices = np.argsort(feature_importance_scores)[::-1]
    sorted_features = [features_list[i] for i in sorted_indices]

    # Remove the least important feature
    least_important_feature = sorted_features[-1]
    print(f"Removing feature: {least_important_feature}")
    with open(file_to_write, 'a') as f:
        f.write(f"Feature rank {len(features_list)}: {least_important_feature}\n")

    # Remove the least important feature from the list
    features_list.remove(least_important_feature)

