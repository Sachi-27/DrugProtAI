
import requests
import json

def get_pdb_info(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json?fields=ft_signal%2Cft_chain%2Cft_disulfid%2Cft_mod_res%2Cft_carbohyd%2Cft_peptide%2Cft_crosslnk"
    response = requests.get(url)
    data = response.json()
    data = {uniprot_id: data["features"]}
    with open(f"PTM_indiv/{uniprot_id}.json", "w") as f:
        json.dump(data, f, indent=4)
    
if __name__ == "__main__":
    with open("../PCFs/files_for_ml/protein_props.json") as f:
        uniprot_human_proteins = list((json.load(f)).keys())
    
    f = open("extract.log","a")
    for uniprot_id in uniprot_human_proteins:
        while True:
            try:
                get_pdb_info(uniprot_id)
                f.write(f"Successful : {uniprot_id}\n")
                print(f"Successful : {uniprot_id}")
                break
            except:
                print(f"Reattempting : {uniprot_id}")
                

