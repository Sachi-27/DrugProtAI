
import requests
import json


def extract_ppi_info(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
    data = requests.get(url)

    interaction_info = []
    start = True
    # process line by line
    lines = data.text.split("\n")
    for line in lines:
        # if line begins with CC and contains -!- INTERACTION: 
        if start:
            if not (line.startswith("CC") and "-!- INTERACTION:" in line):
                continue
            else:
                start = False
                continue
        # if line begins with CC and contains -!- again then break
        if "-!-" in line:
            break
        # else it is an interaction data
        line2 = line.replace("CC       ", "").strip().split(";")
        line2 = [x.strip() for x in line2]
        if not (len(line2) == 5 or len(line2) == 6): 
            # print("Number of lines not 5 or 6:", line2)
            # exit(1)
            break
        entry1 = line2[0].strip()
        if(":") in line2[1]:
            parts = line2[1].strip().split(":")
            entry2_id = parts[0].strip()
            entry2_name = ":".join([x.strip() for x in parts[1:]]).strip()
        else:
            entry2_id = line2[1].strip()
            entry2_name = ""
        entry2_id = entry2_id.strip()
        entry2_name = entry2_name.strip()
        if len(line2) == 6:
            interaction_type = "xeno"
            if not (line2[2] == "Xeno"): 
                print("Len 6 but not xeno: ", line2)
                exit(2)
        else:
            interaction_type = "binary"
        if ("NbExp=" in line2[-3]):
            nb_exp = int(line2[-3].split("=")[1].strip())
        else:
            print("NbExp not found: ", line2)
            exit(3)
        
        interaction_info.append({
            "entry1": entry1,
            "entry2_id": entry2_id,
            "entry2_name": entry2_name,
            "interaction_type": interaction_type,
            "nb_exp": nb_exp
        })

    interaction_info = {uniprot_id: interaction_info}
    with open(f"PPI_indiv/{uniprot_id}.json", "w") as f:
        json.dump(interaction_info, f, indent=4)


if __name__ == "__main__":
    with open("../PCFs/files_for_ml/protein_props.json") as f:
        uniprot_human_proteins = list((json.load(f)).keys())
    
    f = open("extract.log","a")
    for i in range(len(uniprot_human_proteins)):
            uniprot_id = uniprot_human_proteins[i]
            try:
                extract_ppi_info(uniprot_id)
                f.write(f"{i}. Successful : {uniprot_id}\n")
                print(f"{i}. Successful : {uniprot_id}")
            except:
                f.write(f"{i}. Failed : {uniprot_id}\n")
                print(f"{i}. Failed : {uniprot_id}")
