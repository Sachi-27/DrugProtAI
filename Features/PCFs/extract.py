from pyfaidx import Fasta
import Bio
from Bio.PDB import *
from Bio.SeqUtils.ProtParam import ProteinAnalysis  
from Bio.SeqUtils.ProtParam import ProtParamData
import json

#### NOTE
#### Before running, please unzip the file 'uniprotkb_taxonomy_id_9606_AND_reviewed_2024_03_08.fasta.gz'

genes = Fasta('uniprotkb_taxonomy_id_9606_AND_reviewed_2024_03_08.fasta')
sequences = [v[:] for k,v in genes.items()]
print("Number of Sequences:",len(sequences))

def sequence_analysis(sequence, pH=7):
    # Replace 'U', Selenocysteine with 'C', Cysteine
    sequence = sequence.replace('U', 'C')
    seq_info = dict()
    
    analyzed_seq = ProteinAnalysis(str(sequence))
    seq_info['Sequence'] = str(sequence)
    seq_info['Sequence Length'] = len(sequence)
    seq_info['Molecular Weight'] = analyzed_seq.molecular_weight()
    seq_info['GRAVY'] = analyzed_seq.gravy()
    seq_info['Amino Acid Count'] = analyzed_seq.count_amino_acids()
    seq_info['Amino Acid Percent'] = analyzed_seq.get_amino_acids_percent()
    seq_info['Molar Extinction Coefficient'] = analyzed_seq.molar_extinction_coefficient()
    seq_info['Isoelectric Point']= analyzed_seq.isoelectric_point()
    seq_info['Instability Index']= analyzed_seq.instability_index()
    seq_info['Aromaticity']= analyzed_seq.aromaticity()
    seq_info['Secondary Structure'] = analyzed_seq.secondary_structure_fraction()
    seq_info['Flexibility'] = analyzed_seq.flexibility()
    seq_info[f'Charge at {pH}'] = analyzed_seq.charge_at_pH(pH=pH)
    return seq_info

uniprot_protein_props = dict()
for sequence in sequences:
    uniprot_id = sequence.name.split('|')[1]
    uniprot_protein_props[uniprot_id] = sequence_analysis(sequence.seq)

json_data = json.dumps(uniprot_protein_props)
with open('files_for_ml/protein_props.json', "w") as f:
    f.write(json_data)