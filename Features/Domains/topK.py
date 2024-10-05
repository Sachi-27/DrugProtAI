import json
import pandas as pd
K = 20

domains = pd.read_csv("unique_domains.csv", delimiter="::", engine='python')
domain_names = list(domains['domain'])[:K]

with open("entire_domains_info.json") as f:
    all_data = json.load(f)

new_data = {}
for protein in all_data:
    protein_data = {}
    for desc, count in all_data[protein].items():
        if desc in domain_names: protein_data[desc] = count 
    new_data[protein] = protein_data

with open(f"files_for_ml/data_top{K}.json", "w") as f:
    json.dump(new_data, f, indent=4)