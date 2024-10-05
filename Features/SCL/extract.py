import requests
import xml.etree.ElementTree as ET
import time

def fetch_subcellular_locations(uniprot_id):
    # Fetch the XML data from the URL
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.xml"
    response = requests.get(url)
    xml_data = response.text

    # Parse the XML data
    root = ET.fromstring(xml_data)

    # Find and extract content within <comment type="subcellular location"> tags
    subcellular_locations = {}
    for comment in root.findall(".//{http://uniprot.org/uniprot}comment"):
        comment_type = comment.attrib.get("type")
        if comment_type == "subcellular location":
            molecule = comment.find(".//{http://uniprot.org/uniprot}molecule")
            if molecule is not None:
                molecule = molecule.text
            else:
                molecule = "general"
            if molecule not in subcellular_locations:
                subcellular_locations[molecule] = []
            subcellularLocations = comment.findall(".//{http://uniprot.org/uniprot}subcellularLocation")
            for subcellularLocation in subcellularLocations:
                locations = subcellularLocation.findall(".//{http://uniprot.org/uniprot}location")
                subcellular_locations[molecule].append(", ".join([location.text for location in locations]))

    return subcellular_locations

import sys 
uniprot_id = sys.argv[1]
print("Fetching data for", uniprot_id)
try:
    subcellular_locations = fetch_subcellular_locations(uniprot_id)
except:
    print("Something went wrong")
    sys.exit(5)

data = {uniprot_id: subcellular_locations}
# read the content from subcellular_locations.json and update it with the new data
import json
import os

if not os.path.exists("subcellular_locations.json"):
    with open("subcellular_locations.json", "w") as f:
        json.dump(data, f, indent=4)

else:
    with open("subcellular_locations.json", "r") as f:
        existing_data = json.load(f)
    existing_data.update(data)
    # write the updated content to subcellular_locations.json
    with open("subcellular_locations.json", "w") as f:
        json.dump(existing_data, f, indent=4)

print("Data updated successfully")
