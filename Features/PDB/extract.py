import requests
import json

def get_pdb_info(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json?fields=ft_helix%2Cft_strand%2Cft_turn"
    response = requests.get(url)
    data = response.json()
    data = {uniprot_id: data["features"]}
    with open(f"PDB/{uniprot_id}.json", "w") as f:
        json.dump(data, f, indent=4)
    
if __name__ == "__main__":
    with open("../PCFs/files_for_ml/protein_props.json") as f:
        uniprot_human_proteins = list((json.load(f)).keys())
    
    for uniprot_id in uniprot_human_proteins:
        try:
            get_pdb_info(uniprot_id)
            print(f"Successful : {uniprot_id}")
        except:
            print(f"Error occured : {uniprot_id}")
            continue

