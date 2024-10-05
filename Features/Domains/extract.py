import requests
import json
import sys

def get_domain_info(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json?fields=ft_region%2Cft_domain%2Cft_compbias%2Cft_motif"
    response = requests.get(url)
    data = response.json()
    data = {uniprot_id: data["features"]}
    with open(f"Dom/{uniprot_id}.json", "w") as f:
        json.dump(data, f, indent=4)
    
if __name__ == "__main__":
    filename = sys.argv[1] # logger file
    with open("../PCFs/files_for_ml/protein_props.json") as f:
        uniprot_human_proteins = list((json.load(f)).keys())
    count = 1
    for uniprot_id in uniprot_human_proteins:
        try:
            get_domain_info(uniprot_id)
            f = open(filename, "a")
            f.write(str(count) + ". Successful : " + uniprot_id + "\n")
            f.close()
            print(f"{count}. Successful : {uniprot_id}")
        except:
            f = open(filename, "a")
            f.write(str(count) + ". Error occured : " + uniprot_id + "\n")
            f.close()
            print(f"{count}. Error occured : {uniprot_id}")
            continue
        count += 1

