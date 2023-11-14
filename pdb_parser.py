from Bio.PDB import *
from Bio.Data.IUPACData import protein_letters_3to1
import os
import json
DATA_PATH = 'data'

pdb_parser = PDBParser()
id_to_residues = {}
for filename in os.listdir(f'{DATA_PATH}/UP000005640_9606_HUMAN_v4'):
    if filename.endswith(".pdb"):
        uniprot = filename.split('-')[1]  # UniProtKB Object ID
        pdb_structure = pdb_parser.get_structure('foobar', f'{DATA_PATH}/UP000005640_9606_HUMAN_v4/{filename}')
        all_residues = ""
        for res in pdb_structure.get_residues():
            all_residues += protein_letters_3to1[res.get_resname().capitalize()]
        id_to_residues[uniprot] = all_residues

with open(f"{DATA_PATH}/residues.json", "w") as f:
    json.dump(id_to_residues, f)
